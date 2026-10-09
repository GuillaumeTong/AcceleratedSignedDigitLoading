import math

import torch
from torch import Tensor


def rank_topk(x: Tensor, k: int, dim: int = -1):
    """
    Give the descending sorting rank of the values in x, maxing out at k
    """
    _, sorted_indices = torch.topk(x, k, dim=dim)
    ranks = torch.full_like(x, k, dtype=torch.long)
    arange = torch.arange(k, device=x.device).broadcast_to(sorted_indices.shape)
    ranks.scatter_(dim=dim, index=sorted_indices, src=arange)
    return ranks


def compute_residue_dependencides(residue: Tensor, one: Tensor):
    """
    Compute all elements needed for ASDL that only depend on the residue.
    `one` serves to avoid repeatedly re-creating new tensors,
    and sould be made like : `one = residue.new_ones(1)`
    """
    residue_abs = residue.abs()
    residue_max = residue_abs.max(-1, keepdim=True)[0]
    r_mantissa, r_exponent = residue_max.frexp()
    del residue_max
    r_mantissa.abs_()
    max_pots = r_exponent + torch.where(r_mantissa >= 0.75, 0, -1)
    del r_mantissa, r_exponent
    max_pots = torch.ldexp(one, max_pots)
    same_pot = residue_abs >= 0.75 * max_pots
    return residue_abs, max_pots, same_pot


def accelerated_sdl(x: Tensor, limit: float, fill="fill_sorted") -> Tensor:
    """
    Re-implemented MPGBP with p=1 (AKA SDL)
    Work directly with mantissa and exponents, without relying on complex functions like log2 and squaring
    Vectorize all operations
    Replace the P parameter with determining the PoT using the max,
      then apply it to all vector terms that match the same PoT
    """
    if x.is_cuda:
        return accelerated_sdl_v2(x, limit, fill)
    else:
        return accelerated_sdl_v1(x, limit, fill)


def accelerated_sdl_v1(x: Tensor, limit: float, fill="fill_sorted") -> Tensor:
    M_max = math.floor(x.shape[-1] * limit)
    # x_ = x.flatten(0, -2)
    residue = x.clone()
    x_hat = torch.zeros_like(x)
    total_popcount = torch.zeros(size=x.size()[:-1], dtype=torch.long, device=x.device)
    one = x.new_ones(1)
    # float32 has 8-bit exponents, and 24-bit mantissa, resulting in a possible exponent range of [-149, 127]
    # This technique should not take more steps than the possible exponent levels
    for _ in range(277):
        need_more_pots = (total_popcount < M_max) & (residue.abs().max(-1)[0] > 0)
        if not need_more_pots.any():
            return x_hat

        subresidue = residue[need_more_pots]
        subresidue_abs, max_pots, same_pot = compute_residue_dependencides(subresidue, one)
        pot_budget = M_max - total_popcount[need_more_pots]
        need_restrict = pot_budget < same_pot.sum(-1)
        if need_restrict.any():
            if fill == "fill_sorted":
                pot_budget = pot_budget[need_restrict]
                k = pot_budget.max()
                ranks = rank_topk(subresidue_abs[need_restrict], k)

                # Mask: keep same_pot only where rank < K
                mask = ranks < pot_budget.unsqueeze(-1)
                same_pot[need_restrict] &= mask
            elif fill == "underfill":
                # underfill
                if need_restrict.all():
                    return x_hat
                same_pot[need_restrict] = False
            else:
                raise ValueError(f"{fill=}")
        update = max_pots.copysign(subresidue).where(same_pot, 0)
        residue[need_more_pots] -= update
        x_hat[need_more_pots] += update
        total_popcount[need_more_pots] += same_pot.sum(-1)
    raise RuntimeError("The accelerated MPGBP has run for much longer than expected")


def accelerated_sdl_v2(x: Tensor, limit: float, fill="fill_sorted") -> Tensor:
    """
    - use where instead of indexing
    - do the last fill as one synchronized step
    - iterate blindly as much as possible
    - simplify update
    """
    M_max = math.floor(x.shape[-1] * limit)
    # x_ = x.flatten(0, -2)
    residue = x.clone()
    x_hat = torch.zeros_like(x)
    total_popcount = torch.zeros(size=x.size()[:-1], dtype=torch.long, device=x.device)
    one = x.new_ones(1)
    need_more_pots = torch.ones(size=x.size()[:-1], dtype=torch.bool, device=x.device)
    blind_i = math.ceil(limit)  # Allows blindly iterating without requiring a GPU sync
    # float32 has 8-bit exponents, and 24-bit mantissa, resulting in a possible exponent range of [-149, 127]
    # This technique should not take more steps than the possible exponent levels
    for _ in range(277):
        need_more_pots_now = (total_popcount < M_max) & (residue.abs().max(-1)[0] > 0)
        need_more_pots &= need_more_pots_now
        # if torch.is_tensor(blind_i):
        #     blind_i = blind_i.item()
        if blind_i:
            blind_i -= 1
        elif not need_more_pots.any():
            break
        residue_abs, max_pots, same_pot = compute_residue_dependencides(residue, one)
        pot_budget = M_max - total_popcount
        same_pot_sum = same_pot.sum(-1)
        need_restrict = pot_budget < same_pot_sum
        need_more_pots &= ~need_restrict
        same_pot &= need_more_pots.unsqueeze(-1)
        same_pot_sum = same_pot.sum(-1)
        del residue_abs, pot_budget, need_restrict
        update = max_pots.copysign(residue).where(same_pot, 0)
        del max_pots, same_pot
        residue -= update
        x_hat += update
        del update
        total_popcount += same_pot_sum
        del same_pot_sum
    else:
        # Only executed if no break was ever encountered
        raise RuntimeError("The accelerated MPGBP has run for much longer than expected")

    if fill == "underfill":
        return x_hat

    # need_more_pots = need_more_pots_now
    if not need_more_pots_now.any():
        return x_hat
    residue_abs, max_pots, same_pot = compute_residue_dependencides(residue, one)
    pot_budget = M_max - total_popcount
    need_restrict = pot_budget < same_pot.sum(-1)
    if fill == "fill_sorted":
        k = pot_budget.where(need_restrict, 0).max()
        ranks = rank_topk(residue_abs, k)
        del k

        # Mask: keep same_pot only where rank < K
        mask = ranks < pot_budget.unsqueeze(-1)
        del ranks
        same_pot &= mask
        del mask
    else:
        raise ValueError(f"{fill=}")
    del residue_abs, pot_budget
    same_pot &= need_restrict.unsqueeze(-1)
    update = max_pots.copysign(residue).where(same_pot, 0)
    del max_pots, same_pot
    x_hat += update
    return x_hat
