"""SQLite-backed, append-only catalog of tiled elevation surfaces.

The catalog is deliberately independent of the outlet-plan publisher: it knows
how to validate a batch of tiles against a complete fixed-size surface and how
to expose each frozen version's per-cell elevation with provenance.  Nothing in
this module knows about water, outlets or plans.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .model import (
    BatchError,
    BatchResult,
    Cell,
    Slice,
    SourceRef,
    TileSnapshot,
    is_int,
    sha256_hex,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    tile_id  TEXT    NOT NULL,
    revision INTEGER NOT NULL,
    row      INTEGER NOT NULL,
    col      INTEGER NOT NULL,
    nrows    INTEGER NOT NULL,
    ncols    INTEGER NOT NULL,
    grid     TEXT    NOT NULL,
    PRIMARY KEY (tile_id, revision)
);
CREATE TABLE IF NOT EXISTS versions (
    version        INTEGER PRIMARY KEY,
    ceiling        INTEGER NOT NULL,
    elevation_hash TEXT    NOT NULL,
    heights        TEXT    NOT NULL,
    sources        TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS version_tiles (
    version  INTEGER NOT NULL,
    tile_id  TEXT    NOT NULL,
    revision INTEGER NOT NULL,
    PRIMARY KEY (version, tile_id)
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version  INTEGER,
    ok       INTEGER NOT NULL,
    payload  TEXT    NOT NULL,
    errors   TEXT    NOT NULL,
    at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


class Catalog:
    """A catalog of immutable elevation versions over a fixed H x W surface."""

    def __init__(self, path: str | Path, height: int, width: int, ceiling: int):
        if not (is_int(height) and is_int(width) and is_int(ceiling)):
            raise ValueError("height, width and ceiling must be integers")
        if height <= 0 or width <= 0:
            raise ValueError("surface dimensions must be positive")
        if ceiling < 0:
            raise ValueError("ceiling must be non-negative")
        self.path = str(path)
        self.height = height
        self.width = width
        self.ceiling = ceiling
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def open(cls, path: str | Path) -> "Catalog":
        """Re-open an existing catalog, recovering dimensions and ceiling."""

        conn = sqlite3.connect(str(path), isolation_level=None)
        try:
            row = conn.execute(
                "SELECT height, width, ceiling FROM catalog_meta WHERE id=1"
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        finally:
            conn.close()
        if row is None:
            raise ValueError("catalog does not exist or lacks metadata")
        return cls(path, row[0], row[1], row[2])

    def _meta(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS catalog_meta ("
            "id INTEGER PRIMARY KEY CHECK (id=1), height INTEGER, width INTEGER, ceiling INTEGER)"
        )
        cur = self._conn.execute("SELECT height, width, ceiling FROM catalog_meta WHERE id=1").fetchone()
        if cur is None:
            self._conn.execute(
                "INSERT INTO catalog_meta(id,height,width,ceiling) VALUES (1,?,?,?)",
                (self.height, self.width, self.ceiling),
            )
        elif (cur[0], cur[1], cur[2]) != (self.height, self.width, self.ceiling):
            raise ValueError(
                f"catalog metadata mismatch: stored {(cur[0], cur[1], cur[2])}, "
                f"requested {(self.height, self.width, self.ceiling)}"
            )

    def _ensure_meta(self) -> None:
        self._meta()

    # -- transactions ------------------------------------------------------
    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self._ensure_meta()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # -- reads -------------------------------------------------------------
    def latest_version(self) -> int | None:
        self._ensure_meta()
        row = self._conn.execute("SELECT MAX(version) FROM versions").fetchone()
        return row[0]

    def list_versions(self) -> list[int]:
        self._ensure_meta()
        return [r[0] for r in self._conn.execute("SELECT version FROM versions ORDER BY version")]

    def get_slice(self, version: int | None = None) -> Slice:
        self._ensure_meta()
        if version is None:
            version = self.latest_version()
            if version is None:
                raise KeyError("catalog has no versions")
        row = self._conn.execute(
            "SELECT ceiling, elevation_hash, heights, sources FROM versions WHERE version=?",
            (version,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no such version: {version}")
        ceiling, elevation_hash, heights_json, sources_json = row
        heights = tuple(tuple(int(v) for v in r) for r in json.loads(heights_json))
        raw_sources = json.loads(sources_json)
        sources: dict[Cell, tuple[SourceRef, ...]] = {}
        for key, refs in raw_sources.items():
            row_i, col_i = (int(x) for x in key.split(","))
            sources[(row_i, col_i)] = tuple(
                SourceRef(
                    tile_id=ref["tile_id"],
                    tile_revision=ref["tile_revision"],
                    local_row=ref["local_row"],
                    local_col=ref["local_col"],
                    height=ref["height"],
                )
                for ref in refs
            )
        tile_rows = self._conn.execute(
            "SELECT tile_id, revision FROM version_tiles WHERE version=? ORDER BY tile_id",
            (version,),
        ).fetchall()
        tiles = tuple(self._tile_snapshot(tile_id, revision) for tile_id, revision in tile_rows)
        snapshot = Slice(
            version=version,
            height=self.height,
            width=self.width,
            ceiling=ceiling,
            heights=heights,
            elevation_hash=elevation_hash,
            sources=sources,
            tiles=tiles,
        )
        return snapshot

    def _tile_snapshot(self, tile_id: str, revision: int) -> TileSnapshot:
        row = self._conn.execute(
            "SELECT row, col, nrows, ncols, grid FROM tiles WHERE tile_id=? AND revision=?",
            (tile_id, revision),
        ).fetchone()
        if row is None:
            raise KeyError(f"missing tile {tile_id}@{revision}")
        r, c, nrows, ncols, values_json = row
        values = tuple(tuple(int(v) for v in line) for line in json.loads(values_json))
        return TileSnapshot(tile_id, revision, r, c, nrows, ncols, values)

    def source_at(self, version: int | None, row: int, col: int) -> tuple[SourceRef, ...]:
        return self.get_slice(version).sources.get((row, col), ())

    # -- writes ------------------------------------------------------------
    def apply_batch(self, tiles: Sequence[Mapping[str, Any]]) -> BatchResult:
        """Validate and append one batch, replacing any tiles named in it.

        The batch is all-or-nothing: on any validation failure no version is
        created and previously published products remain untouched.
        """

        normalized, errors = self._normalize_batch(tiles)
        with self._write() as conn:
            # Freeze the active tile set *before* writing anything: replacement
            # is by tile_id, unnamed tiles carry over from the latest version.
            active = self._active_tiles(conn, normalized)
            try:
                heights, sources, overlap_cells = self._resolve(active)
            except _ValidationError as exc:
                errors = errors + exc.errors
                self._record_batch(conn, None, False, tiles, errors)
                return BatchResult(ok=False, version=None, errors=tuple(errors))
            if errors:
                self._record_batch(conn, None, False, tiles, errors)
                return BatchResult(ok=False, version=None, errors=tuple(errors))
            elevation_hash = sha256_hex(
                {
                    "height": self.height,
                    "width": self.width,
                    "ceiling": self.ceiling,
                    "heights": [list(r) for r in heights],
                }
            )
            for tile in normalized:
                conn.execute(
                    "INSERT OR IGNORE INTO tiles(tile_id,revision,row,col,nrows,ncols,grid) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        tile.tile_id,
                        tile.revision,
                        tile.row,
                        tile.col,
                        tile.nrows,
                        tile.ncols,
                        json.dumps([list(r) for r in tile.values]),
                    ),
                )
            version = (conn.execute("SELECT COALESCE(MAX(version),0)+1 FROM versions").fetchone()[0])
            sources_serial = {
                f"{r},{c}": [ref.to_dict() for ref in refs] for (r, c), refs in sources.items()
            }
            conn.execute(
                "INSERT INTO versions(version,ceiling,elevation_hash,heights,sources) VALUES (?,?,?,?,?)",
                (
                    version,
                    self.ceiling,
                    elevation_hash,
                    json.dumps([list(r) for r in heights]),
                    json.dumps(sources_serial, sort_keys=True),
                ),
            )
            for tile_id, tile in sorted(active.items()):
                conn.execute(
                    "INSERT INTO version_tiles(version,tile_id,revision) VALUES (?,?,?)",
                    (version, tile_id, tile.revision),
                )
            self._record_batch(conn, version, True, tiles, [])
            return BatchResult(ok=True, version=version, errors=(), overlap_cells=overlap_cells)

    def _record_batch(
        self,
        conn: sqlite3.Connection,
        version: int | None,
        ok: bool,
        payload: Sequence[Mapping[str, Any]],
        errors: Sequence[BatchError],
    ) -> None:
        conn.execute(
            "INSERT INTO batches(version,ok,payload,errors) VALUES (?,?,?,?)",
            (
                version,
                1 if ok else 0,
                json.dumps(list(payload), sort_keys=True),
                json.dumps([e.to_dict() for e in errors], sort_keys=True),
            ),
        )

    # -- validation helpers ------------------------------------------------
    def _normalize_batch(
        self, tiles: Sequence[Mapping[str, Any]]
    ) -> tuple[list[TileSnapshot], list[BatchError]]:
        errors: list[BatchError] = []
        seen: dict[str, int] = {}
        normalized: list[TileSnapshot] = []
        for index, raw in enumerate(tiles):
            if not isinstance(raw, Mapping):
                errors.append(BatchError("bad_tile", f"tile[{index}] is not an object"))
                continue
            tile_id = raw.get("tile_id")
            revision = raw.get("revision")
            row = raw.get("row")
            col = raw.get("col")
            values = raw.get("values")
            if not isinstance(tile_id, str) or not tile_id:
                errors.append(BatchError("bad_tile", f"tile[{index}] has invalid tile_id"))
                continue
            if not is_int(revision) or revision <= 0:
                errors.append(
                    BatchError("bad_revision", f"tile {tile_id} revision must be a positive integer")
                )
                continue
            if not is_int(row) or not is_int(col):
                errors.append(BatchError("bad_tile", f"tile {tile_id} origin must be integers"))
                continue
            if not isinstance(values, (list, tuple)) or not values:
                errors.append(BatchError("bad_tile", f"tile {tile_id} values must be a non-empty matrix"))
                continue
            matrix: list[tuple[int, ...]] = []
            ok = True
            ncols = None
            for line in values:
                if not isinstance(line, (list, tuple)) or not line:
                    errors.append(BatchError("bad_tile", f"tile {tile_id} has an empty row"))
                    ok = False
                    break
                if ncols is None:
                    ncols = len(line)
                elif len(line) != ncols:
                    errors.append(
                        BatchError("bad_tile", f"tile {tile_id} is not rectangular (ragged rows)")
                    )
                    ok = False
                    break
                parsed = []
                for value in line:
                    if not is_int(value):
                        errors.append(
                            BatchError("bad_height", f"tile {tile_id} contains a non-integer height")
                        )
                        ok = False
                        break
                    parsed.append(value)
                if not ok:
                    break
                matrix.append(tuple(parsed))
            if not ok:
                continue
            nrows = len(matrix)
            assert ncols is not None
            if row < 0 or col < 0 or row + nrows > self.height or col + ncols > self.width:
                errors.append(
                    BatchError(
                        "out_of_bounds",
                        f"tile {tile_id} rectangle rows [{row},{row + nrows}) cols "
                        f"[{col},{col + ncols}) exceeds the {self.height}x{self.width} surface",
                        {"tile_id": tile_id},
                    )
                )
                continue
            if tile_id in seen:
                errors.append(
                    BatchError(
                        "duplicate_tile_in_batch",
                        f"tile {tile_id} appears twice in one batch",
                        {"tile_id": tile_id},
                    )
                )
                continue
            seen[tile_id] = index
            snapshot = TileSnapshot(tile_id, revision, row, col, nrows, ncols, tuple(matrix))
            previous = self._conn.execute(
                "SELECT nrows,ncols,row,col,grid FROM tiles WHERE tile_id=? AND revision=?",
                (tile_id, revision),
            ).fetchone()
            if previous is not None:
                prev_values = [[int(v) for v in line] for line in json.loads(previous[4])]
                if (
                    previous[0] != nrows
                    or previous[1] != ncols
                    or previous[2] != row
                    or previous[3] != col
                    or prev_values != [list(r) for r in matrix]
                ):
                    errors.append(
                        BatchError(
                            "immutable_revision",
                            f"tile {tile_id}@{revision} already exists with different content",
                            {"tile_id": tile_id, "revision": revision},
                        )
                    )
                    continue
            normalized.append(snapshot)
        return normalized, errors

    def _active_tiles(
        self,
        conn: sqlite3.Connection,
        batch: Sequence[TileSnapshot] | Mapping[str, TileSnapshot],
    ) -> dict[str, TileSnapshot]:
        batch_map = (
            {tile.tile_id: tile for tile in batch}
            if not isinstance(batch, Mapping)
            else dict(batch)
        )
        latest = conn.execute("SELECT MAX(version) FROM versions").fetchone()[0]
        active: dict[str, TileSnapshot] = {}
        if latest is not None:
            rows = conn.execute(
                "SELECT tile_id, revision FROM version_tiles WHERE version=?", (latest,)
            ).fetchall()
            for tile_id, revision in rows:
                active[tile_id] = self._tile_snapshot(tile_id, revision)
        active.update(batch_map)
        return active

    def _resolve(
        self, active: Mapping[str, TileSnapshot]
    ) -> tuple[tuple[tuple[int, ...], ...], dict[Cell, tuple[SourceRef, ...]], int]:
        coverage: dict[Cell, list[SourceRef]] = {}
        conflicts: list[dict[str, Any]] = []
        for tile in active.values():
            for row, col, lr, lc, value in tile.iter_cells():
                if not (0 <= row < self.height and 0 <= col < self.width):
                    raise _ValidationError(
                        [
                            BatchError(
                                "out_of_bounds",
                                f"tile {tile.tile_id}@{tile.revision} covers out-of-bounds cell ({row},{col})",
                                {"tile_id": tile.tile_id, "cell": [row, col]},
                            )
                        ]
                    )
                coverage.setdefault((row, col), []).append(
                    SourceRef(tile.tile_id, tile.revision, lr, lc, value)
                )
        gaps = [
            [r, c]
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in coverage
        ]
        for (row, col), refs in coverage.items():
            heights = {ref.height for ref in refs}
            if len(heights) > 1:
                conflicts.append(
                    {
                        "cell": [row, col],
                        "heights": sorted(heights),
                        "sources": [
                            {
                                "tile_id": ref.tile_id,
                                "revision": ref.tile_revision,
                                "height": ref.height,
                            }
                            for ref in refs
                        ],
                    }
                )
        errors: list[BatchError] = []
        if gaps:
            errors.append(
                BatchError(
                    "coverage_gap",
                    f"{len(gaps)} cell(s) are not covered by any tile",
                    {"cells": gaps},
                )
            )
        if conflicts:
            errors.append(
                BatchError(
                    "overlap_mismatch",
                    f"{len(conflicts)} overlapping cell(s) disagree on height",
                    {"conflicts": conflicts},
                )
            )
        above_ceiling = [
            {"cell": [ref_cell[0], ref_cell[1]], "height": ref.height}
            for ref_cell, refs in coverage.items()
            for ref in refs
            if ref.height > self.ceiling
        ]
        if above_ceiling:
            errors.append(
                BatchError(
                    "height_above_ceiling",
                    f"{len(above_ceiling)} cell contribution(s) exceed the ceiling C={self.ceiling}",
                    {"cells": above_ceiling},
                )
            )
        if errors:
            raise _ValidationError(errors)
        heights = tuple(
            tuple(next(iter({ref.height for ref in coverage[(r, c)]})) for c in range(self.width))
            for r in range(self.height)
        )
        sources = {
            cell: tuple(sorted(refs, key=lambda ref: (ref.tile_id, ref.tile_revision)))
            for cell, refs in coverage.items()
        }
        overlap_cells = sum(1 for refs in coverage.values() if len(refs) > 1)
        return heights, sources, overlap_cells


class _ValidationError(Exception):
    def __init__(self, errors: Sequence[BatchError]):
        super().__init__("; ".join(e.message for e in errors))
        self.errors = list(errors)
