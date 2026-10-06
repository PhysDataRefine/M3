"""
Step 1 — Partition.

Points are Morton-sorted inside one cube of half-width H and split as an octree.
A cell c stops when any of these holds:

  |P_c| ≤ κ          (min_points_stop_split, default 3)
  depth ≥ G_max
  δ_c ≤ ε_refine

with

  δ_c = max(w_φ Δφ_c, w_ψ Δψ_c)
  h_c = (2H) / 2^g

Δφ_c is the range of the scalar field in the cell.
Δψ_c is the largest projection range of the vector field over 13 directions
(the axis, face-diagonal, and space-diagonal directions, up to sign).
A cell with one point is stored with δ_c = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Cell:
    level: int
    morton_prefix: int
    start_idx: int
    end_idx: int
    delta: float = 0.0

    def __len__(self) -> int:
        return int(max(0, self.end_idx - self.start_idx))


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
        "boundary_max_level": octree.get("boundary_max_level"),
        "volume_max_level": octree.get("volume_max_level"),
        "boundary_epsilon_refine": octree.get("boundary_epsilon_refine"),
        "volume_epsilon_refine": octree.get("volume_epsilon_refine"),
        "weight_p": octree.get("weight_p"),
        "weight_v": octree.get("weight_v"),
    }


def _unit_directions_13() -> np.ndarray:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [inv_sqrt2, inv_sqrt2, 0.0],
            [inv_sqrt2, -inv_sqrt2, 0.0],
            [inv_sqrt2, 0.0, inv_sqrt2],
            [inv_sqrt2, 0.0, -inv_sqrt2],
            [0.0, inv_sqrt2, inv_sqrt2],
            [0.0, inv_sqrt2, -inv_sqrt2],
            [inv_sqrt3, inv_sqrt3, inv_sqrt3],
            [inv_sqrt3, inv_sqrt3, -inv_sqrt3],
            [inv_sqrt3, -inv_sqrt3, inv_sqrt3],
            [-inv_sqrt3, inv_sqrt3, inv_sqrt3],
        ],
        dtype=np.float64,
    )


def morton_encode_3d(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, *, max_bits: int) -> np.ndarray:
    if max_bits > 21:
        raise ValueError("max_bits must be <= 21 for 64-bit Morton codes")
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    iz = np.asarray(iz, dtype=np.int64)
    if not (ix.shape == iy.shape == iz.shape):
        raise ValueError("ix,iy,iz must have the same shape")

    def _split_by_3(x: np.ndarray) -> np.ndarray:
        x = x & 0x1FFFFF
        x = (x | (x << 32)) & 0x1F00000000FFFF
        x = (x | (x << 16)) & 0x1F0000FF0000FF
        x = (x | (x << 8)) & 0x100F00F00F00F00F
        x = (x | (x << 4)) & 0x10C30C30C30C30C3
        x = (x | (x << 2)) & 0x1249249249249249
        return x

    xx = _split_by_3(ix)
    yy = _split_by_3(iy)
    zz = _split_by_3(iz)
    return (xx | (yy << 1) | (zz << 2)).astype(np.int64, copy=False)


def compute_bbox_cube(
    points_xyz: np.ndarray,
    *,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    P = np.asarray(points_xyz, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 3:
        raise ValueError("points_xyz must have shape (N,3)")
    if bbox_min is None or bbox_max is None:
        mn = P[:, :3].min(axis=0)
        mx = P[:, :3].max(axis=0)
    else:
        mn = np.asarray(bbox_min, dtype=np.float64).reshape(3)
        mx = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    center = 0.5 * (mn + mx)
    H = 0.5 * float(np.max(mx - mn))
    if H <= 1e-12:
        H = 1.0
    return center - H, center + H, center, float(H)


def cell_side_length(*, H: float, level: int) -> float:
    lv = int(level)
    if lv < 0:
        raise ValueError("level must be >= 0")
    return float((2.0 * float(H)) / float(1 << lv))


def morton_codes_from_points(
    points_xyz: np.ndarray,
    *,
    max_level: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    P = np.asarray(points_xyz, dtype=np.float64)
    mn = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    mx = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    extent = mx - mn
    if np.any(extent <= 0):
        raise ValueError("Invalid bbox: max must be > min on all axes")
    u = (P[:, :3] - mn) / extent
    u = np.clip(u, 0.0, 1.0 - 1e-10)
    scale = 1 << int(max_level)
    coords = np.floor(u * scale).astype(np.int64)
    coords = np.clip(coords, 0, scale - 1)
    return morton_encode_3d(coords[:, 0], coords[:, 1], coords[:, 2], max_bits=int(max_level))


def scalar_range(values: np.ndarray, idx: np.ndarray) -> float:
    if idx.size <= 0:
        return 0.0
    v = values[idx]
    return float(np.max(v) - np.min(v))


def projection_diameter(vec3: np.ndarray, idx: np.ndarray, D: np.ndarray) -> float:
    if idx.size <= 0:
        return 0.0
    V = vec3[idx, :3].astype(np.float64, copy=False)
    proj = V @ D.T
    return float(np.max(np.max(proj, axis=0) - np.min(proj, axis=0)))


def _partition_first_level(sorted_codes: np.ndarray, sorted_indices: np.ndarray, *, shift: int) -> List[Cell]:
    child_ids = ((sorted_codes >> shift) & 0x7).astype(np.int64)
    cells: List[Cell] = []
    for child in range(8):
        left = int(np.searchsorted(child_ids, child, side="left"))
        right = int(np.searchsorted(child_ids, child, side="right"))
        if left < right:
            cells.append(Cell(level=1, morton_prefix=int(child), start_idx=left, end_idx=right))
    return cells


def _subdivide_cell(
    cell: Cell,
    *,
    sorted_codes: np.ndarray,
    sorted_indices: np.ndarray,
    max_level: int,
) -> List[Cell]:
    next_level = cell.level + 1
    if next_level > max_level:
        return []
    shift = 3 * (int(max_level) - int(next_level))

    codes_slice = sorted_codes[cell.start_idx : cell.end_idx]
    child_ids = ((codes_slice >> shift) & 0x7).astype(np.int64)
    order = np.argsort(child_ids, kind="stable")
    child_ids_sorted = child_ids[order]

    sl = slice(cell.start_idx, cell.end_idx)
    sorted_codes[sl] = codes_slice[order]
    sorted_indices[sl] = sorted_indices[sl][order]

    out: List[Cell] = []
    base_prefix = int(cell.morton_prefix)
    for child in range(8):
        n = int(np.sum(child_ids_sorted == child))
        if n == 0:
            continue
        left = int(np.searchsorted(child_ids_sorted, child, side="left"))
        right = left + n
        prefix = (base_prefix << 3) | int(child)
        out.append(
            Cell(
                level=next_level,
                morton_prefix=prefix,
                start_idx=cell.start_idx + left,
                end_idx=cell.start_idx + right,
            )
        )
    return out


def step1_partition(
    *,
    points_xyz: np.ndarray,
    phi_scalar: np.ndarray,
    psi_vec3: Optional[np.ndarray],
    max_level: int,
    min_points_stop_split: int,
    epsilon_refine: float,
    weight_phi: float = 1.0,
    weight_psi: float = 1.0,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
    directions_D: Optional[np.ndarray] = None,
) -> Tuple[List[Cell], np.ndarray]:
    P = np.asarray(points_xyz, dtype=np.float64)
    phi = np.asarray(phi_scalar, dtype=np.float64).reshape(-1)
    if P.shape[0] != phi.shape[0]:
        raise ValueError("points_xyz and phi_scalar must have the same length")
    if psi_vec3 is not None:
        psi = np.asarray(psi_vec3, dtype=np.float64)
        if psi.shape[0] != P.shape[0] or psi.shape[1] < 3:
            raise ValueError("psi_vec3 must have shape (N,3)")
    else:
        psi = None

    bbox_min2, bbox_max2, _center, _H = compute_bbox_cube(P, bbox_min=bbox_min, bbox_max=bbox_max)
    codes = morton_codes_from_points(P, max_level=int(max_level), bbox_min=bbox_min2, bbox_max=bbox_max2)

    shift_lvl1 = 3 * (int(max_level) - 1)
    child_ids_root = ((codes >> shift_lvl1) & 0x7).astype(np.int64)
    sorted_indices = np.argsort(child_ids_root, kind="stable").astype(np.int64)
    sorted_codes = codes[sorted_indices].astype(np.int64, copy=False)

    if directions_D is None:
        D = _unit_directions_13()
    else:
        D = np.asarray(directions_D, dtype=np.float64)
        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError("directions_D must have shape (n_dirs,3)")

    queue: List[Cell] = _partition_first_level(sorted_codes, sorted_indices, shift=shift_lvl1)
    terminal: List[Cell] = []

    while queue:
        c = queue.pop(0)
        n = len(c)
        if n <= int(min_points_stop_split) or c.level >= int(max_level):
            terminal.append(c)
            continue

        idx = sorted_indices[c.start_idx : c.end_idx]
        dp = scalar_range(phi, idx)
        dv = projection_diameter(psi, idx, D) if (psi is not None) else 0.0
        delta_c = float(max(float(weight_phi) * dp, float(weight_psi) * dv))
        c.delta = delta_c

        if delta_c <= float(epsilon_refine):
            terminal.append(c)
            continue

        children = _subdivide_cell(
            c, sorted_codes=sorted_codes, sorted_indices=sorted_indices, max_level=int(max_level)
        )
        if not children:
            terminal.append(c)
            continue

        for ch in children:
            if len(ch) <= int(min_points_stop_split):
                terminal.append(ch)
            else:
                queue.append(ch)

    return terminal, sorted_indices


def step1_partition_from_yaml(
    *,
    points_xyz: np.ndarray,
    phi_scalar: np.ndarray,
    psi_vec3: Optional[np.ndarray],
    file_type: str,
    min_points_stop_split: int,
    yaml_path: str,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
    directions_D: Optional[np.ndarray] = None,
) -> Tuple[List[Cell], np.ndarray]:
    p = load_supplement_params(yaml_path)
    ft = str(file_type).lower()
    if ft not in ("boundary", "volume"):
        raise ValueError("file_type must be 'boundary' or 'volume'")

    if ft == "boundary":
        max_level = int(p.get("boundary_max_level") or 8)
        epsilon_refine = float(p.get("boundary_epsilon_refine") or 0.05)
    else:
        max_level = int(p.get("volume_max_level") or 13)
        epsilon_refine = float(p.get("volume_epsilon_refine") or 0.005)

    weight_phi = float(p.get("weight_p") or 1.0)
    weight_psi = float(p.get("weight_v") or 0.4)

    return step1_partition(
        points_xyz=points_xyz,
        phi_scalar=phi_scalar,
        psi_vec3=psi_vec3,
        max_level=max_level,
        min_points_stop_split=int(min_points_stop_split),
        epsilon_refine=epsilon_refine,
        weight_phi=weight_phi,
        weight_psi=weight_psi,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        directions_D=directions_D,
    )
