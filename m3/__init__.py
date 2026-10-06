"""M³: Multi-scale Measure Modeling."""

from .config import (
    DATASET_NORM_CONFIGS,
    DatasetNormConfig,
    get_dataset_norm_config,
    list_dataset_norm_configs,
)
from .pipeline import M3Result, run_m3

__all__ = [
    "run_m3",
    "M3Result",
    "DATASET_NORM_CONFIGS",
    "DatasetNormConfig",
    "get_dataset_norm_config",
    "list_dataset_norm_configs",
]
__version__ = "0.1.0"
