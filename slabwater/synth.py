"""Deterministic 12x12 / 8-outlet stress material generator.

Used by the tests and for manual explorers.  The generator is seeded so runs are
reproducible; it intentionally creates ridges that make the "cheapest sill"
shortcut wrong, forcing the solver to respect the max-height path rule.
"""

from __future__ import annotations

import random
from typing import Any


def make_material(seed: int = 772, size: int = 12, ceiling: int = 9) -> dict[str, Any]:
    rng = random.Random(seed)
    h = w = size
    heights = [[rng.randint(0, 6) for _ in range(w)] for _ in range(h)]
    # A blocking ridge down the middle plus a high corner so paths matter.
    ridge_col = size // 2
    for r in range(h):
        heights[r][ridge_col] = 7
    heights[0][ridge_col] = 6  # leave one lower crossing
    heights[h - 1][w - 1] = 8

    # Two tiles: left half columns [0, ridge_col], right half [ridge_col, end).
    # They overlap on ridge_col and must agree there (they share heights above).
    left = {
        "tile_id": "LEFT",
        "revision": 1,
        "row": 0,
        "col": 0,
        "values": [row[: ridge_col + 1] for row in heights],
    }
    right = {
        "tile_id": "RIGHT",
        "revision": 1,
        "row": 0,
        "col": ridge_col,
        "values": [row[ridge_col:] for row in heights],
    }
    boundary_cells = []
    for c in range(w):
        boundary_cells.append((0, c))
        boundary_cells.append((h - 1, c))
    for r in range(h):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, w - 1))
    boundary_cells = sorted(set(boundary_cells))
    rng.shuffle(boundary_cells)
    outlets = []
    chosen = []
    for cand in boundary_cells:
        if len(chosen) >= 8:
            break
        if cand not in chosen:
            chosen.append(cand)
    for i, (r, c) in enumerate(chosen):
        outlets.append(
            {
                "id": f"O{i}",
                "cell": [r, c],
                "sill": rng.randint(0, 7),
            }
        )

    lowering = {
        "tile_id": "RIGHT",
        "revision": 2,
        "row": 0,
        "col": ridge_col,
        "values": [row[ridge_col:] for row in heights],
    }
    # Lower the ridge in the second revision; the shared overlap column (local
    # col 0) must stay identical to the untouched LEFT tile or the batch would
    # be rejected as an overlap mismatch.
    lowered = []
    for row in lowering["values"]:
        new_row = list(row)
        for c in range(1, len(new_row)):
            new_row[c] = max(0, new_row[c] - 3)
        lowered.append(new_row)
    lowering["values"] = lowered

    requests = [
        {
            "id": "S1",
            "source_revision": 1,
            "opens": 3,
            "total_max": 10_000,
            "cell_max": [],
        },
        {
            "id": "S2",
            "source_revision": 2,
            "opens": 8,
            "total_max": 10_000,
            "cell_max": [],
        },
        {
            "id": "S3",
            "source_revision": 2,
            "opens": 1,
            "total_max": 0,
            "cell_max": [],
        },
    ]
    return {
        "height": h,
        "width": w,
        "ceiling": ceiling,
        "initial_tiles": [left, right],
        "correction": lowering,
        "outlets": outlets,
        "requests": requests,
    }
