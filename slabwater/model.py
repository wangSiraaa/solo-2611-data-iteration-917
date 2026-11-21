"""Immutable value objects shared by the catalog, hydro engine and publisher.

All quantities in this package are integer synthetic-model quantities.  This is
not a hydraulic model; it is a discrete verifier for the rules described in the
task statement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

Cell = tuple[int, int]


def is_int(value: Any) -> bool:
    """True for real integers (``bool`` is rejected even though it is an int)."""

    return isinstance(value, int) and not isinstance(value, bool)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Outlet:
    """A boundary outlet: a cell plus the sill height that must be overtopped."""

    id: str
    cell: Cell
    sill: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "cell": [self.cell[0], self.cell[1]], "sill": self.sill}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Outlet":
        cell = raw.get("cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError("outlet cell must be a [row, col] pair")
        row, col = cell
        if not is_int(row) or not is_int(col):
            raise ValueError("outlet cell coordinates must be integers")
        if not isinstance(raw.get("id"), str) or not raw.get("id"):
            raise ValueError("outlet id must be a non-empty string")
        sill = raw.get("sill")
        if not is_int(sill):
            raise ValueError("outlet sill must be an integer")
        return cls(id=raw["id"], cell=(row, col), sill=sill)

    @classmethod
    def from_parts(cls, id: str, cell: Sequence[int], sill: int) -> "Outlet":
        return cls(id=id, cell=(int(cell[0]), int(cell[1])), sill=sill)


@dataclass(frozen=True)
class SourceRef:
    """Provenance of one tile's contribution to a resolved cell."""

    tile_id: str
    tile_revision: int
    local_row: int
    local_col: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "tile_revision": self.tile_revision,
            "local_row": self.local_row,
            "local_col": self.local_col,
            "height": self.height,
        }


@dataclass(frozen=True)
class TileSnapshot:
    """A tile at a frozen catalog version."""

    tile_id: str
    revision: int
    row: int
    col: int
    nrows: int
    ncols: int
    values: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "revision": self.revision,
            "row": self.row,
            "col": self.col,
            "nrows": self.nrows,
            "ncols": self.ncols,
            "values": [list(row) for row in self.values],
        }

    def iter_cells(self):
        for lr, row in enumerate(self.values):
            for lc, value in enumerate(row):
                yield self.row + lr, self.col + lc, lr, lc, value


@dataclass(frozen=True)
class Slice:
    """A frozen, fully-covered elevation surface at one catalog version."""

    version: int
    height: int
    width: int
    ceiling: int
    heights: tuple[tuple[int, ...], ...]
    elevation_hash: str
    sources: Mapping[Cell, tuple[SourceRef, ...]]
    tiles: tuple[TileSnapshot, ...]

    def at(self, row: int, col: int) -> int:
        return self.heights[row][col]

    def cell(self) -> Iterable[tuple[int, int, int]]:
        for r, row in enumerate(self.heights):
            for c, value in enumerate(row):
                yield r, c, value

    def is_boundary(self, row: int, col: int) -> bool:
        return row in (0, self.height - 1) or col in (0, self.width - 1)

    def ground_total(self) -> int:
        return sum(value for _, _, value in self.cell())

    def elevation_source(self, include_cells: bool = True) -> dict[str, Any]:
        source: dict[str, Any] = {
            "version": self.version,
            "height": self.height,
            "width": self.width,
            "ceiling": self.ceiling,
            "elevation_hash": self.elevation_hash,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }
        if include_cells:
            cells = []
            for r, row in enumerate(self.heights):
                for c, value in enumerate(row):
                    refs = self.sources.get((r, c), ())
                    cells.append(
                        {
                            "cell": [r, c],
                            "height": value,
                            "sources": [ref.to_dict() for ref in refs],
                        }
                    )
            source["cells"] = cells
        return source


@dataclass(frozen=True)
class BatchError:
    code: str
    message: str
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details or {})}


@dataclass(frozen=True)
class BatchResult:
    ok: bool
    version: int | None
    errors: tuple[BatchError, ...] = ()
    overlap_cells: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "errors": [error.to_dict() for error in self.errors],
            "overlap_cells": self.overlap_cells,
        }


@dataclass(frozen=True)
class WaterResult:
    opened: tuple[Outlet, ...]
    levels: tuple[tuple[int, ...], ...]
    storage: tuple[tuple[int, ...], ...]
    total_storage: int
    dijkstra_runs: int = 0
    cell_updates: int = 0

    def storage_at(self, row: int, col: int) -> int:
        return self.storage[row][col]

    def level_at(self, row: int, col: int) -> int:
        return self.levels[row][col]

    def to_report(self) -> dict[str, Any]:
        return {
            "opened": [outlet.id for outlet in self.opened],
            "levels": [list(row) for row in self.levels],
            "storage": [list(row) for row in self.storage],
            "total_storage": self.total_storage,
        }
