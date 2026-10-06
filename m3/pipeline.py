"""Partition, group, and sample.

``run_m3`` reads ``hyperparameters.yaml`` unless the call overrides those values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .step1_partitioning import (
    _parse_minimal_yaml,
    cell_side_length,
    compute_bbox_cube,
    step1_partition_from_yaml,
)
from .step2_grouping import step2_group_cells_from_yaml
from .step3_sampling import step3_allocate_and_sample


_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_YAML = _PKG_ROOT / "hyperparameters.yaml"
_FROM_YAML = object()


@dataclass
class M3Result:
    sampled_indices: np.ndarray
    sorted_indices: np.ndarray
    cell_start: np.ndarray
    cell_end: np.ndarray
    cell_level: np.ndarray
    cell_delta: np.ndarray
    cell_h: np.ndarray
    effective_level: np.ndarray
    S: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.cell_start.size)

    @property
    def n_sampled(self) -> int:
        return int(self.sampled_indices.size)


def _zscore(x: np.ndarray, mean: Optional[Sequence[float]], std: Optional[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mean is None or std is None:
        return arr
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    s = np.where(np.abs(s) < 1e-12, 1.0, s)
    if arr.ndim == 1:
        return (arr - m.reshape(())) / s.reshape(()) if m.size == 1 else (arr - m[0]) / s[0]
    return (arr - m.reshape(1, -1)) / s.reshape(1, -1)


def _sampling_choices(yaml_path: str) -> Dict[str, Any]:
    cfg = _parse_minimal_yaml(yaml_path) or {}
    sampling = cfg.get("sampling") or {}
    rho = sampling.get("max_cell_fill_ratio", 0.3)
    if rho is not None:
        rho = float(rho)
        if rho >= 1.0:
            rho = None
    backflow = sampling.get("enable_inter_level_backflow", True)
    if backflow is None:
        backflow = True
    return {"rho_fill_ratio": rho, "enable_inter_level_backflow": bool(backflow)}


def run_m3(
    *,
    points_xyz: np.ndarray,
    phi_scalar: np.ndarray,
    psi_vec3: Optional[np.ndarray] = None,
    budget: int,
    file_type: str = "boundary",
    seed: int = 0,
    rho_fill_ratio: Any = _FROM_YAML,
    enable_inter_level_backflow: Any = _FROM_YAML,
    yaml_path: Optional[str] = None,
    min_points_stop_split: int = 3,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
    normalize: bool = True,
    phi_mean: Optional[Sequence[float]] = None,
    phi_std: Optional[Sequence[float]] = None,
    psi_mean: Optional[Sequence[float]] = None,
    psi_std: Optional[Sequence[float]] = None,
) -> M3Result:
    yaml_path = str(yaml_path or _DEFAULT_YAML)
    choices = _sampling_choices(yaml_path)
    if rho_fill_ratio is _FROM_YAML:
        rho_fill_ratio = choices["rho_fill_ratio"]
    if enable_inter_level_backflow is _FROM_YAML:
        enable_inter_level_backflow = choices["enable_inter_level_backflow"]

    phi = np.asarray(phi_scalar, dtype=np.float64).reshape(-1)
    psi = None if psi_vec3 is None else np.asarray(psi_vec3, dtype=np.float64)
    if normalize:
        phi = _zscore(phi, phi_mean, phi_std)
        if psi is not None:
            psi = _zscore(psi, psi_mean, psi_std)

    cells, sorted_indices = step1_partition_from_yaml(
        points_xyz=points_xyz,
        phi_scalar=phi,
        psi_vec3=psi,
        file_type=file_type,
        min_points_stop_split=int(min_points_stop_split),
        yaml_path=yaml_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    cube_min, cube_max, _center, H = compute_bbox_cube(
        points_xyz, bbox_min=bbox_min, bbox_max=bbox_max
    )

    from .step1_partitioning import (
        _unit_directions_13,
        load_supplement_params,
        projection_diameter,
        scalar_range,
    )

    p = load_supplement_params(yaml_path)
    w_phi = float(p.get("weight_p") or 1.0)
    w_psi = float(p.get("weight_v") or 0.4)
    D = _unit_directions_13()
    for c in cells:
        if len(c) <= 1:
            c.delta = 0.0
            continue
        idx = sorted_indices[c.start_idx : c.end_idx]
        dp = scalar_range(phi, idx)
        dv = projection_diameter(psi, idx, D) if psi is not None else 0.0
        c.delta = float(max(w_phi * dp, w_psi * dv))

    deltas = [float(c.delta) for c in cells]
    hs = [cell_side_length(H=H, level=int(c.level)) for c in cells]
    npts = [len(c) for c in cells]

    grouping = step2_group_cells_from_yaml(
        delta=deltas,
        h=hs,
        n_points_in_cell=npts,
        yaml_path=yaml_path,
    )

    order = np.argsort(np.asarray([c.start_idx for c in cells], dtype=np.int64), kind="stable")
    cells = [cells[int(i)] for i in order]
    deltas = [deltas[int(i)] for i in order]
    hs = [hs[int(i)] for i in order]
    npts = [npts[int(i)] for i in order]
    grouping.S = grouping.S[order]
    grouping.effective_level = grouping.effective_level[order]

    cells_by_level: Dict[int, List[np.ndarray]] = {}
    for i, c in enumerate(cells):
        lv = int(grouping.effective_level[i])
        pool = sorted_indices[c.start_idx : c.end_idx].astype(np.int64, copy=False)
        cells_by_level.setdefault(lv, []).append(pool)

    sampled, alloc_stats = step3_allocate_and_sample(
        cells_by_level=cells_by_level,
        m_total=int(budget),
        seed=int(seed),
        rho_fill_ratio=rho_fill_ratio,
        enable_inter_level_backflow=bool(enable_inter_level_backflow),
    )

    return M3Result(
        sampled_indices=np.asarray(sampled, dtype=np.int64),
        sorted_indices=np.asarray(sorted_indices, dtype=np.int64),
        cell_start=np.asarray([c.start_idx for c in cells], dtype=np.int64),
        cell_end=np.asarray([c.end_idx for c in cells], dtype=np.int64),
        cell_level=np.asarray([c.level for c in cells], dtype=np.int32),
        cell_delta=np.asarray(deltas, dtype=np.float64),
        cell_h=np.asarray(hs, dtype=np.float64),
        effective_level=np.asarray(grouping.effective_level, dtype=np.int32),
        S=np.asarray(grouping.S, dtype=np.float64),
        stats={
            "n_points": int(np.asarray(points_xyz).shape[0]),
            "n_cells": len(cells),
            "budget": int(budget),
            "n_sampled": int(sampled.size),
            "file_type": str(file_type),
            "H": float(H),
            "bbox_min": cube_min.tolist(),
            "bbox_max": cube_max.tolist(),
            "allocation": {
                "requested_total": alloc_stats.requested_total,
                "used_total": alloc_stats.used_total,
                "total_available": alloc_stats.total_available,
                "level_totals": dict(alloc_stats.level_totals),
                "level_used": dict(alloc_stats.level_used),
                "enable_inter_level_backflow": bool(
                    (alloc_stats.details.get("level_stats") or {}).get("enable_backflow", True)
                ),
            },
        },
    )
