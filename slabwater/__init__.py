"""Independent elevation catalog and opened-outlet storage plan verifier.

Synthetic discrete model only -- not for field hydraulic decisions.
"""

from .catalog import Catalog
from .model import (
    BatchError,
    BatchResult,
    Cell,
    Outlet,
    Slice,
    SourceRef,
    TileSnapshot,
    WaterResult,
    canonical_json,
    sha256_hex,
)
from .publisher import PublishError, PublishResult, Publisher, VerifyResult
from .solver import BottleneckCache, SolveResult, SolverStats, enumerate_subsets, solve

__all__ = [
    "BatchError",
    "BatchResult",
    "BottleneckCache",
    "Catalog",
    "Cell",
    "Outlet",
    "PublishError",
    "PublishResult",
    "Publisher",
    "Slice",
    "SolveResult",
    "SolverStats",
    "SourceRef",
    "TileSnapshot",
    "VerifyResult",
    "WaterResult",
    "canonical_json",
    "enumerate_subsets",
    "sha256_hex",
    "solve",
]
