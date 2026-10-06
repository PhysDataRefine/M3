"""
Step 2 — Grouping.

Cell size is not the scale. Each multi-point cell is scored by intensity per length,

  S_c = log(ε_log + δ_c / h_c),

then cut into K equal-width bins, ℓ(c) ∈ {1, …, K}.
The bin edges come from this case's S_c, optionally clipped to
log_strata_quantiles (paper default: minimum and P99.9).
If log_strata_fixed_range is set, those absolute edges replace the per-case quantiles.
A singleton cell has no in-cell variation and is placed in stratum K+1.
ε_log is 1e-30, only so the log stays finite when δ_c = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


EPSILON_LOG: float = 1e-30


def _parse_minimal_yaml(path: str) -> dict:
    def _parse_scalar(tok: str):
        t = tok.strip()
        if t == "" or t.lower() == "null":
            return None
        if t.lower() == "true":
            return True
        if t.lower() == "false":
            return False
        if t.startswith("[") and t.endswith("]"):
            inner = t[1:-1].strip()
            if inner == "":
                return []
            return [_parse_scalar(x) for x in inner.split(",")]
        try:
            if any(c in t for c in (".", "e", "E")):
                return float(t)
            return int(t)
        except Exception:
            return t

    root: dict = {}
    stack: list[tuple[int, dict]] = [(0, root)]
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            if ":" not in line:
                continue
            k, v = line.lstrip(" ").split(":", 1)
            key = k.strip()
            val = v.strip()

            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()
            cur = stack[-1][1]

            if val == "":
                nxt: dict = {}
                cur[key] = nxt
                stack.append((indent + 2, nxt))
            else:
                cur[key] = _parse_scalar(val)
    return root


def load_supplement_params(yaml_path: str) -> dict:
    cfg = _parse_minimal_yaml(yaml_path) or {}
    octree = cfg.get("octree") or {}
    return {
        "effective_strata_bins": octree.get("effective_strata_bins"),
        "log_strata_quantiles": octree.get("log_strata_quantiles"),
        "log_strata_fixed_range": octree.get("log_strata_fixed_range"),
    }


def singleton_stratum_id(K: int) -> int:
    return int(K) + 1


def score_Sc(*, delta_c: float, h_c: float, epsilon_log: float = EPSILON_LOG) -> float:
    h = float(h_c)
    if h <= 1e-300:
        h = 1e-300
    return float(np.log(float(epsilon_log) + float(delta_c) / h))


def assign_equal_width_bins(
    S: np.ndarray,
    K: int,
    *,
    quantile_low_pct: Optional[float] = None,
    quantile_high_pct: Optional[float] = None,
    fixed_lo: Optional[float] = None,
    fixed_hi: Optional[float] = None,
) -> np.ndarray:
    x = np.asarray(S, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.empty((0,), dtype=np.int32)
    K = int(K)
    if K < 2:
        raise ValueError("K must be >= 2")

    use_fixed = fixed_lo is not None and fixed_hi is not None and float(fixed_hi) > float(fixed_lo)
    use_q = (not use_fixed) and (
        (quantile_low_pct is not None) or (quantile_high_pct is not None)
    )
    if use_fixed:
        lo = float(fixed_lo)
        hi = float(fixed_hi)
        use_q = True
    elif use_q:
        lo = float(np.percentile(x, quantile_low_pct)) if quantile_low_pct is not None else float(np.min(x))
        hi = float(np.percentile(x, quantile_high_pct)) if quantile_high_pct is not None else float(np.max(x))
        if hi <= lo + 1e-300:
            lo, hi = float(np.min(x)), float(np.max(x))
            use_q = False
    else:
        lo, hi = float(np.min(x)), float(np.max(x))

    if hi <= lo + 1e-300:
        out = np.empty(x.size, dtype=np.int32)
        out.fill((K + 1) // 2)
        return out

    xc = np.clip(x, lo, hi) if use_q else x
    bins = np.floor(np.clip((xc - lo) / (hi - lo), 0.0, 1.0 - 1e-15) * K).astype(np.int32) + 1
    np.clip(bins, 1, K, out=bins)
    return bins


@dataclass
class GroupingResult:
    S: np.ndarray
    effective_level: np.ndarray


def step2_group_cells(
    *,
    delta: Sequence[float],
    h: Sequence[float],
    n_points_in_cell: Sequence[int],
    K: int,
    epsilon_log: float = EPSILON_LOG,
    quantile_low_pct: Optional[float] = None,
    quantile_high_pct: Optional[float] = None,
    fixed_lo: Optional[float] = None,
    fixed_hi: Optional[float] = None,
) -> GroupingResult:
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    npts = np.asarray(n_points_in_cell, dtype=np.int64).reshape(-1)
    if not (delta.size == h.size == npts.size):
        raise ValueError("delta, h, and n_points_in_cell must have the same length")

    S = np.empty(delta.size, dtype=np.float64)
    for i in range(delta.size):
        S[i] = score_Sc(delta_c=float(delta[i]), h_c=float(h[i]), epsilon_log=epsilon_log)

    eff = np.empty(delta.size, dtype=np.int32)
    sing = singleton_stratum_id(int(K))
    mask_multi = npts > 1
    if np.any(mask_multi):
        eff[mask_multi] = assign_equal_width_bins(
            S[mask_multi],
            int(K),
            quantile_low_pct=quantile_low_pct,
            quantile_high_pct=quantile_high_pct,
            fixed_lo=fixed_lo,
            fixed_hi=fixed_hi,
        )
    eff[~mask_multi] = int(sing)
    return GroupingResult(S=S, effective_level=eff)


def step2_group_cells_from_yaml(
    *,
    delta: Sequence[float],
    h: Sequence[float],
    n_points_in_cell: Sequence[int],
    yaml_path: str,
    epsilon_log: float = EPSILON_LOG,
) -> GroupingResult:
    p = load_supplement_params(yaml_path)
    K = int(p.get("effective_strata_bins") or 64)
    q = p.get("log_strata_quantiles")
    qlo = q[0] if isinstance(q, list) and len(q) >= 2 else None
    qhi = q[1] if isinstance(q, list) and len(q) >= 2 else None
    fixed = p.get("log_strata_fixed_range")
    flo = fixed[0] if isinstance(fixed, list) and len(fixed) >= 2 else None
    fhi = fixed[1] if isinstance(fixed, list) and len(fixed) >= 2 else None
    return step2_group_cells(
        delta=delta,
        h=h,
        n_points_in_cell=n_points_in_cell,
        K=K,
        epsilon_log=epsilon_log,
        quantile_low_pct=qlo,
        quantile_high_pct=qhi,
        fixed_lo=flo,
        fixed_hi=fhi,
    )
