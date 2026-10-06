"""Reference quantities for AhmedML, DrivAerML, and SHIFT-Wing.

``run_m3`` does not read them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

_THIS_DIR = Path(__file__).resolve().parent
_DATASETS_YAML = _THIS_DIR / "datasets.yaml"


@dataclass(frozen=True)
class DatasetNormConfig:
    n_cases: Optional[int] = None
    size_tb: Optional[float] = None
    solver_family: Optional[str] = None
    streamwise_axis: Optional[str] = None
    vertical_axis: Optional[str] = None
    length_ref: Optional[float] = None
    length_ref_name: Optional[str] = None
    area_ref: Optional[float] = None
    area_ref_name: Optional[str] = None
    u_inf: Optional[float] = None
    reynolds_ref: Optional[float] = None
    reynolds_ref_name: Optional[str] = None
    rho_ref: Optional[float] = None
    nu_ref: Optional[float] = None
    temperature_ref: Optional[float] = None
    mach_ref: Optional[float] = None
    p_ref: Optional[float] = None
    q_ref: Optional[float] = None
    raw_pos_min: Optional[float] = None
    raw_pos_max: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_scalar(tok: str) -> Any:
    t = tok.strip()
    if t == "" or t.lower() == "null":
        return None
    if t.lower() == "true":
        return True
    if t.lower() == "false":
        return False
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]
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


def _parse_minimal_yaml(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(0, root)]
    with path.open("r", encoding="utf-8") as f:
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
                nxt: Dict[str, Any] = {}
                cur[key] = nxt
                stack.append((indent + 2, nxt))
            else:
                cur[key] = _parse_scalar(val)
    return root


def _load_yaml_payload() -> Dict[str, Any]:
    return _parse_minimal_yaml(_DATASETS_YAML)


_PAYLOAD = _load_yaml_payload()
_FIELD_NAMES = {f.name for f in fields(DatasetNormConfig)}

DATASET_NORM_CONFIGS: Dict[str, DatasetNormConfig] = {
    str(k): DatasetNormConfig(**{name: val for name, val in v.items() if name in _FIELD_NAMES})
    for k, v in (_PAYLOAD.get("datasets") or {}).items()
}


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower()
    return key.replace(" ", "_").replace("-", "_")


def get_dataset_norm_config(name: str) -> DatasetNormConfig:
    key = normalize_dataset_key(name)
    if key not in DATASET_NORM_CONFIGS:
        available = ", ".join(sorted(DATASET_NORM_CONFIGS.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available dataset keys: {available}")
    return DATASET_NORM_CONFIGS[key]


def list_dataset_norm_configs() -> Dict[str, DatasetNormConfig]:
    return dict(DATASET_NORM_CONFIGS)
