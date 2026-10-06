"""
Step 3 — Sampling.

Each cell offers at most

  Ñ_c = min(N_c, max(1, ceil(ρ N_c)))

points. ρ is sampling.max_cell_fill_ratio (paper default 0.3).
ρ = None or ρ ≥ 1 leaves the cap off, so Ñ_c = N_c.

The budget m is split in two stages.

1. Inter-level. Occupied strata share m as evenly as integer arithmetic allows.
   A stratum that cannot hold its share is clamped, and the overflow is given
   back to strata that still have capacity (enable_inter_level_backflow, default on).
   Turning the backflow off keeps the clamp and leaves the unused budget unspent.
2. Intra-level. Water-filling spreads that stratum budget across cells so the
   integer quotas differ by at most one, subject to Ñ_c.
   Leftover singles are given to random cells that still have room.

Points inside a cell are drawn uniformly without replacement.
The realized count can be smaller than m when the capped pools run out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AllocationStats:
    requested_total: int
    used_total: int
    total_available: int
    level_totals: Dict[int, int]
    level_used: Dict[int, int]
    details: Dict[str, Any]


def cap_with_fill_ratio(cap: int, rho: Optional[float]) -> int:
    cap = int(cap)
    if cap <= 0:
        return 0
    if rho is None:
        return cap
    r = float(rho)
    if r >= 1.0:
        return cap
    if r <= 0.0:
        return 1
    return max(1, min(int(np.ceil(cap * r)), cap))


def compute_two_phase_budgets(
    level_point_counts: Dict[int, int],
    n_total: int,
    *,
    enable_backflow: bool = True,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    levels = sorted(level_point_counts.keys())
    if not levels:
        return {}, {
            "strategy": "iterative_uniform_with_capacity",
            "enable_backflow": bool(enable_backflow),
            "actual_total": 0,
            "deficit": int(n_total),
            "level_totals": {},
            "overflow_before_backflow": 0,
            "backflow_redistributed": 0,
            "iterations": 0,
        }

    total_pts = {int(k): int(v) for k, v in level_point_counts.items()}
    L = len(levels)

    base = int(n_total) // L
    rem = int(n_total) - base * L
    level_totals: Dict[int, int] = {}
    for i, level in enumerate(levels):
        level_totals[int(level)] = int(base + (1 if i >= L - rem else 0))

    remaining_budget = 0
    for level in levels:
        alloc = int(level_totals[int(level)])
        cap = int(total_pts[int(level)])
        if alloc > cap:
            remaining_budget += alloc - cap
            level_totals[int(level)] = cap
    overflow_before_backflow = int(remaining_budget)

    iteration = 0
    backflow_redistributed = 0
    while enable_backflow and remaining_budget > 0:
        levels_with_capacity = [l for l in levels if total_pts[int(l)] > level_totals[int(l)]]
        if not levels_with_capacity:
            break
        K = len(levels_with_capacity)
        base_add = remaining_budget // K
        rem_add = remaining_budget - base_add * K
        if base_add == 0 and rem_add == 0:
            break

        distributed_this_round = 0
        for i, level in enumerate(sorted(levels_with_capacity)):
            want = base_add + (1 if i >= K - rem_add and base_add >= 0 else 0)
            if want <= 0:
                continue
            cap = total_pts[int(level)] - level_totals[int(level)]
            if cap <= 0:
                continue
            add = min(int(want), int(cap), int(remaining_budget - distributed_this_round))
            if add <= 0:
                continue
            level_totals[int(level)] += add
            distributed_this_round += add
            if distributed_this_round >= remaining_budget:
                break

        if distributed_this_round == 0:
            break
        remaining_budget -= distributed_this_round
        backflow_redistributed += distributed_this_round
        iteration += 1

    actual_total = int(sum(level_totals.values()))
    stats = {
        "strategy": (
            "iterative_uniform_with_capacity"
            if enable_backflow
            else "uniform_with_capacity_no_backflow"
        ),
        "enable_backflow": bool(enable_backflow),
        "base_initial": int(base),
        "remainder_initial": int(rem),
        "level_totals": dict(level_totals),
        "actual_total": int(actual_total),
        "deficit": int(n_total) - int(actual_total),
        "overflow_before_backflow": int(overflow_before_backflow),
        "backflow_redistributed": int(backflow_redistributed),
        "remaining_budget_after_iterations": int(remaining_budget),
        "iterations": int(iteration),
    }
    return level_totals, stats


def allocate_cell_budgets_waterfilling(
    capacities: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64).reshape(-1)
    n_cells = int(capacities.size)
    if n_cells == 0:
        return np.empty((0,), dtype=np.int64)
    if int(n_total) <= 0:
        return np.zeros((n_cells,), dtype=np.int64)

    total_cap = int(capacities.sum())
    if int(n_total) >= total_cap:
        return capacities.copy()

    sort_idx = np.argsort(capacities)
    caps_sorted = capacities[sort_idx]

    remaining = int(n_total)
    prev = 0
    active = n_cells

    k = 0
    while k < n_cells and active > 0:
        v = int(caps_sorted[k])
        j = k + 1
        while j < n_cells and int(caps_sorted[j]) == v:
            j += 1
        count_at_v = int(j - k)

        delta = int(v - prev)
        if delta < 0:
            delta = 0

        required = int(delta * active)
        if required <= remaining:
            remaining -= required
            prev = v
            active -= count_at_v
            k = j
            continue

        t = int(prev + (remaining // active))
        rem = int(remaining % active)

        quotas = np.minimum(capacities, t).astype(np.int64, copy=False)
        if rem > 0:
            eligible = np.flatnonzero(capacities > t)
            if eligible.size < rem:
                chosen = eligible
            else:
                chosen = rng.choice(eligible, size=int(rem), replace=False)
            quotas[chosen] += 1
        return quotas

    quotas = np.minimum(capacities, prev).astype(np.int64, copy=False)
    cur = int(quotas.sum())
    if cur > int(n_total):
        over = int(cur - int(n_total))
        idx = np.argsort(capacities)[::-1]
        for i in idx:
            if over <= 0:
                break
            take = min(int(quotas[i]), int(over))
            quotas[i] -= take
            over -= take
    return quotas


def sample_within_cells(
    *,
    cells_point_indices: Sequence[np.ndarray],
    quotas: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    out: List[np.ndarray] = []
    for idx, q in zip(cells_point_indices, quotas):
        q = int(q)
        if q <= 0:
            continue
        pool = np.asarray(idx, dtype=np.int64).reshape(-1)
        if pool.size == 0:
            continue
        if q >= pool.size:
            out.append(pool)
        elif q == 1:
            out.append(np.asarray([int(pool[int(rng.integers(0, pool.size))])], dtype=np.int64))
        else:
            out.append(rng.choice(pool, size=q, replace=False).astype(np.int64, copy=False))
    if not out:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(out, axis=0)


def allocate_cell_quotas_two_stage_equal_cell(
    *,
    cells_by_eff_level: Dict[int, List[np.ndarray]],
    m_total: int,
    rng: np.random.Generator,
    rho_fill_ratio: Optional[float] = None,
    target_levels: Optional[Sequence[int]] = None,
    enable_inter_level_backflow: bool = True,
) -> Tuple[Dict[int, np.ndarray], AllocationStats]:
    level_point_counts: Dict[int, int] = {}
    total_available = 0

    if target_levels is None:
        levels = sorted(int(lv) for lv in cells_by_eff_level.keys())
    else:
        levels = sorted(int(lv) for lv in target_levels)

    for lv in levels:
        pools = cells_by_eff_level.get(int(lv), [])
        caps = np.array([cap_with_fill_ratio(len(p), rho_fill_ratio) for p in pools], dtype=np.int64)
        cap_sum = int(np.sum(caps))
        level_point_counts[int(lv)] = cap_sum
        total_available += cap_sum

    if total_available <= 0 or int(m_total) <= 0:
        stats = AllocationStats(
            requested_total=int(m_total),
            used_total=0,
            total_available=int(total_available),
            level_totals={},
            level_used={},
            details={},
        )
        return {}, stats

    m_req = int(min(int(m_total), int(total_available)))
    level_totals, level_stats = compute_two_phase_budgets(
        level_point_counts,
        m_req,
        enable_backflow=bool(enable_inter_level_backflow),
    )

    quotas_by_level: Dict[int, np.ndarray] = {}
    level_used: Dict[int, int] = {}
    used_total = 0

    for lv in sorted(level_totals.keys()):
        pools = cells_by_eff_level.get(int(lv), [])
        caps = np.array([cap_with_fill_ratio(len(p), rho_fill_ratio) for p in pools], dtype=np.int64)
        q = allocate_cell_budgets_waterfilling(caps, int(level_totals.get(int(lv), 0)), rng)
        quotas_by_level[int(lv)] = q
        used = int(np.sum(q))
        level_used[int(lv)] = used
        used_total += used

    stats = AllocationStats(
        requested_total=int(m_req),
        used_total=int(used_total),
        total_available=int(total_available),
        level_totals=dict(level_totals),
        level_used=dict(level_used),
        details={"level_stats": level_stats},
    )
    return quotas_by_level, stats


def step3_allocate_and_sample(
    *,
    cells_by_level: Dict[int, List[np.ndarray]],
    m_total: int,
    seed: int = 0,
    rho_fill_ratio: Optional[float] = None,
    target_levels: Optional[Sequence[int]] = None,
    enable_inter_level_backflow: bool = True,
) -> Tuple[np.ndarray, AllocationStats]:
    rng = np.random.default_rng(int(seed))
    m_total = int(m_total)
    if m_total < 0:
        raise ValueError("m_total must be >= 0")

    quotas_by_level, stats = allocate_cell_quotas_two_stage_equal_cell(
        cells_by_eff_level=cells_by_level,
        m_total=int(m_total),
        rng=rng,
        rho_fill_ratio=rho_fill_ratio,
        target_levels=target_levels,
        enable_inter_level_backflow=bool(enable_inter_level_backflow),
    )

    sampled_parts: List[np.ndarray] = []
    for lv in sorted(quotas_by_level.keys()):
        pools = cells_by_level.get(int(lv), [])
        q = quotas_by_level[int(lv)]
        sampled_parts.append(sample_within_cells(cells_point_indices=pools, quotas=q.tolist(), rng=rng))

    sampled = np.concatenate(sampled_parts, axis=0) if sampled_parts else np.empty((0,), dtype=np.int64)
    return sampled, stats
