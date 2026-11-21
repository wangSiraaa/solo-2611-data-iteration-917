"""Hydro solver for the discrete "fill and drain through sill outlets" model.

Rules implemented (all integers, four-neighbour movement):

* A route from a cell to an opened outlet has drainage threshold
  ``max(every cell height on the route, outlet sill)``.
* A cell's water level is ``min(C, min(threshold over every opened outlet))``;
  with no opened outlet it is ``C``.
* Storage is ``level - height`` per cell.

Two evaluation strategies are offered:

``recompute``
    Run a multi-source minimax Dijkstra for every candidate set.  Simple and
    self-evidently correct, but repeats work across sets.

``reuse``
    Compute one minimax "bottleneck" map per outlet once, then combine maps by
    taking the per-cell minimum.  Cheaper for many candidate sets.  The maps are
    valid only for one frozen elevation surface; a catalog revision changes the
    surface hash and invalidates them.

Dijkstra here is minimax, not shortest-path-sum: an edge extends a route at the
maximum height seen so far, which is exactly the sill overtopping rule.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .model import Cell, Outlet, Slice, WaterResult

_INF = 2**62
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class SolverStats:
    strategy: str
    dijkstra_runs: int = 0
    cell_updates: int = 0
    bottleneck_cache_hits: int = 0
    bottleneck_cache_misses: int = 0
    bottleneck_invalidations: int = 0
    subsets_evaluated: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "dijkstra_runs": self.dijkstra_runs,
            "cell_updates": self.cell_updates,
            "bottleneck_cache_hits": self.bottleneck_cache_hits,
            "bottleneck_cache_misses": self.bottleneck_cache_misses,
            "bottleneck_invalidations": self.bottleneck_invalidations,
            "subsets_evaluated": self.subsets_evaluated,
        }


@dataclass
class SubsetAttempt:
    opened: tuple[str, ...]
    feasible: bool
    total_storage: int
    storage_at_cells: dict[Cell, int]
    violations: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "opened": list(self.opened),
            "feasible": self.feasible,
            "total_storage": self.total_storage,
            "storage_at_cells": [
                {"cell": [r, c], "storage": v}
                for (r, c), v in sorted(self.storage_at_cells.items())
            ],
            "violations": self.violations,
        }


@dataclass
class SolveResult:
    ok: bool
    reason: str | None
    opened: tuple[Outlet, ...]
    water: WaterResult | None
    attempts: list[SubsetAttempt]
    stats: SolverStats
    exhaustive: bool
    candidate_subsets: int

    def evidence(self, include_attempts: bool = True) -> dict[str, object]:
        data: dict[str, object] = {
            "ok": self.ok,
            "reason": self.reason,
            "opened": [outlet.id for outlet in self.opened],
            "stats": self.stats.to_dict(),
            "exhaustive": self.exhaustive,
            "candidate_subsets": self.candidate_subsets,
        }
        if include_attempts:
            data["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return data


class BottleneckCache:
    """Per-outlet minimax maps, bound to one elevation hash at a time.

    A catalog revision produces a new ``elevation_hash``; the cache notices the
    change on the next lookup and drops the stale maps.  Values returned are
    tuples and are safe to hold, but they describe *some* surface only -- always
    check ``current_hash`` before reusing them elsewhere.
    """

    def __init__(self) -> None:
        self._hash: str | None = None
        self._maps: dict[tuple[str, Cell, int], tuple[int, ...]] = {}
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    @property
    def current_hash(self) -> str | None:
        return self._hash

    def get(self, surface: Slice, outlet: Outlet) -> tuple[int, ...]:
        if surface.elevation_hash != self._hash:
            if self._hash is not None:
                self.invalidations += 1
            self._hash = surface.elevation_hash
            self._maps.clear()
        key = (outlet.id, outlet.cell, outlet.sill)
        cached = self._maps.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        values, _ = _run_minimax(surface, [outlet])
        self._maps[key] = values
        return values

    def clear(self) -> None:
        self._maps.clear()
        self._hash = None


def _run_minimax(surface: Slice, outlets: Sequence[Outlet]) -> tuple[tuple[int, ...], int]:
    """Multi-source minimax over ``outlets``; returns flat map and update count."""

    size = surface.height * surface.width
    dist = [_INF] * size
    heap: list[tuple[int, int]] = []
    for outlet in outlets:
        r, c = outlet.cell
        start = max(surface.heights[r][c], outlet.sill)
        idx = r * surface.width + c
        if start < dist[idx]:
            dist[idx] = start
            heapq.heappush(heap, (start, idx))
    updates = 0
    w = surface.width
    while heap:
        d, idx = heapq.heappop(heap)
        if d > dist[idx]:
            continue
        r, c = divmod(idx, w)
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < surface.height and 0 <= nc < surface.width:
                h = surface.heights[nr][nc]
                nd = d if d >= h else h
                nidx = nr * w + nc
                if nd < dist[nidx]:
                    dist[nidx] = nd
                    updates += 1
                    heapq.heappush(heap, (nd, nidx))
    return tuple(dist), updates


def _materialize(
    surface: Slice, dist: Sequence[int], opened: Sequence[Outlet]
) -> WaterResult:
    """Turn a minimax map into levels/storage, clamping at the ceiling.

    ``dist`` is ``+inf`` for cells unreachable from every opened outlet; those
    cells sit at the ceiling, so the ceiling is substituted before clamping.
    """

    levels: list[list[int]] = []
    storage: list[list[int]] = []
    total = 0
    for r in range(surface.height):
        lrow: list[int] = []
        srow: list[int] = []
        for c in range(surface.width):
            d = dist[r * surface.width + c]
            level = surface.ceiling if d == _INF else min(surface.ceiling, d)
            amount = level - surface.heights[r][c]
            lrow.append(level)
            srow.append(amount)
            total += amount
        levels.append(lrow)
        storage.append(srow)
    return WaterResult(
        opened=tuple(opened),
        levels=tuple(tuple(row) for row in levels),
        storage=tuple(tuple(row) for row in storage),
        total_storage=total,
    )


def _violations(
    total: int,
    storage: Sequence[Sequence[int]],
    total_max: int,
    cell_max: Sequence[tuple[Cell, int]],
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    if total > total_max:
        violations.append({"constraint": "total", "limit": total_max, "actual": total})
    for (r, c), limit in cell_max:
        actual = storage[r][c]
        if actual > limit:
            violations.append(
                {"constraint": "cell", "cell": [r, c], "limit": limit, "actual": actual}
            )
    return violations


def enumerate_subsets(outlets: Sequence[Outlet], opens: int) -> Iterable[tuple[Outlet, ...]]:
    """Yield outlet subsets of size 0..opens in deterministic order.

    Ordering is by size then by outlet id so a repeated request with the same
    input produces an identical product.  ``opens`` is only an upper bound;
    opening fewer outlets is always allowed.
    """

    ordered = sorted(outlets, key=lambda outlet: outlet.id)
    for size in range(0, min(opens, len(ordered)) + 1):
        yield from itertools.combinations(ordered, size)


def solve(
    surface: Slice,
    outlets: Sequence[Outlet],
    opens: int,
    total_max: int,
    cell_max: Sequence[tuple[Cell, int]] = (),
    strategy: str = "reuse",
    cache: BottleneckCache | None = None,
    max_subsets: int = 4096,
    stop_on_first: bool = True,
) -> SolveResult:
    """Search outlet sets (size <= opens) for one satisfying every water limit.

    ``stop_on_first`` returns as soon as a feasible set is found.  With
    ``stop_on_first=False`` the search continues and, if some set is feasible,
    returns the feasible set with the smallest total storage (ties broken by
    deterministic order) while retaining every attempt as evidence.
    """

    if strategy not in ("reuse", "recompute"):
        raise ValueError("strategy must be 'reuse' or 'recompute'")
    if opens < 0:
        raise ValueError("opens must be non-negative")
    subsets = list(enumerate_subsets(outlets, opens))
    total_subsets = len(subsets)
    stats = SolverStats(strategy=strategy)
    stats.bottleneck_invalidations = cache.invalidations if cache else 0
    if total_subsets > max_subsets:
        return SolveResult(
            ok=False,
            reason="search_space_too_large",
            opened=(),
            water=None,
            attempts=[],
            stats=stats,
            exhaustive=False,
            candidate_subsets=total_subsets,
        )
    if cache is None:
        cache = BottleneckCache()
    attempts: list[SubsetAttempt] = []
    best: tuple[int, int, tuple[Outlet, ...], WaterResult] | None = None
    for order, subset in enumerate(subsets):
        stats.subsets_evaluated += 1
        if strategy == "recompute":
            dist, updates = _run_minimax(surface, subset)
            stats.dijkstra_runs += 1
            stats.cell_updates += updates
        else:
            combined: list[int] | None = None
            for outlet in subset:
                values = cache.get(surface, outlet)
                if combined is None:
                    combined = list(values)
                else:
                    for i, v in enumerate(values):
                        if v < combined[i]:
                            combined[i] = v
            if combined is None:
                combined = [_INF] * (surface.height * surface.width)
            dist = combined
        water = _materialize(surface, dist, subset)
        storage_cells = {
            cell: water.storage[cell[0]][cell[1]] for cell, _ in cell_max
        }
        violations = _violations(
            water.total_storage, water.storage, total_max, tuple(cell_max)
        )
        attempts.append(
            SubsetAttempt(
                opened=tuple(outlet.id for outlet in subset),
                feasible=not violations,
                total_storage=water.total_storage,
                storage_at_cells=storage_cells,
                violations=violations,
            )
        )
        if not violations:
            candidate = (water.total_storage, order, subset, water)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            if stop_on_first:
                break
    stats.bottleneck_cache_hits = cache.hits
    stats.bottleneck_cache_misses = cache.misses
    stats.bottleneck_invalidations = cache.invalidations
    if best is not None:
        _, _, subset, water = best
        return SolveResult(
            ok=True,
            reason=None,
            opened=tuple(subset),
            water=water,
            attempts=attempts,
            stats=stats,
            exhaustive=not stop_on_first,
            candidate_subsets=total_subsets,
        )
    # Every candidate set was checked and none satisfied the limits.
    return SolveResult(
        ok=False,
        reason="no_feasible_outlet_set",
        opened=(),
        water=None,
        attempts=attempts,
        stats=stats,
        exhaustive=True,
        candidate_subsets=total_subsets,
    )
