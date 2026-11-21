"""End-to-end tests for the elevation catalog and storage-plan publisher.

Run with::

    python -m unittest discover -s tests -v

Materials live in ``materials/772``; the values asserted here are taken from
that README so the tests double as a regression of the stated example.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from slabwater.catalog import Catalog
from slabwater.model import Outlet
from slabwater.publisher import Publisher
from slabwater.solver import BottleneckCache, _materialize, _run_minimax, solve
from slabwater.synth import make_material

ROOT = Path(__file__).resolve().parent.parent
MATERIAL = ROOT / "materials" / "772" / "input.json"


def load_material():
    return json.loads(MATERIAL.read_text(encoding="utf-8"))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.material = load_material()
        self.catalog = Catalog(
            self.dir / "catalog.db",
            self.material["height"],
            self.material["width"],
            self.material["ceiling"],
        )
        self.publisher = Publisher(self.dir / "plans.db", self.catalog)

    def tearDown(self) -> None:
        self.publisher.close()
        self.catalog.close()
        self.tmp.cleanup()

    def outlets(self):
        return [Outlet.from_dict(o) for o in self.material["outlets"]]

    def explicit_water(self, surface, ids):
        opened = [o for o in self.outlets() if o.id in ids]
        dist, _ = _run_minimax(surface, opened)
        return _materialize(surface, dist, opened)


class CatalogTests(Base):
    def test_initial_slice_matches_readme(self):
        result = self.catalog.apply_batch(self.material["initial_tiles"])
        self.assertTrue(result.ok)
        self.assertEqual(result.version, 1)
        self.assertEqual(result.overlap_cells, 1)
        surface = self.catalog.get_slice(1)
        self.assertEqual([list(r) for r in surface.heights], [[0, 4, 1, 4, 0]])
        # Overlap cell (0,2) carries provenance from both tiles.
        refs = surface.sources[(0, 2)]
        self.assertEqual({r.tile_id for r in refs}, {"T1", "T2"})
        self.assertEqual({r.height for r in refs}, {1})

    def test_per_cell_row_col_source(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        source = self.catalog.source_at(1, 0, 3)
        self.assertEqual(len(source), 1)
        self.assertEqual(source[0].tile_id, "T2")
        self.assertEqual((source[0].local_row, source[0].local_col), (0, 1))

    def test_invalid_correction_rejected_and_version_unchanged(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        self.catalog.apply_batch([self.material["correction"]])
        before = self.catalog.list_versions()
        bad = self.catalog.apply_batch([self.material["invalid_correction"]])
        self.assertFalse(bad.ok)
        self.assertIsNone(bad.version)
        codes = {e.code for e in bad.errors}
        self.assertIn("overlap_mismatch", codes)
        self.assertEqual(self.catalog.list_versions(), before)
        # Old product still readable and unchanged.
        self.assertEqual(
            [list(r) for r in self.catalog.get_slice(2).heights], [[0, 4, 1, 2, 0]]
        )

    def test_coverage_gap_rejected(self):
        gap = {
            "tile_id": "T1",
            "revision": 1,
            "row": 0,
            "col": 0,
            "values": [[0, 4]],
        }
        result = self.catalog.apply_batch([gap])
        self.assertFalse(result.ok)
        self.assertIn("coverage_gap", {e.code for e in result.errors})

    def test_out_of_bounds_rejected(self):
        bad = {
            "tile_id": "X",
            "revision": 1,
            "row": 0,
            "col": 3,
            "values": [[1, 2, 3]],
        }
        result = self.catalog.apply_batch([bad])
        self.assertFalse(result.ok)
        self.assertIn("out_of_bounds", {e.code for e in result.errors})

    def test_batch_replaces_multiple_and_is_atomic(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        # Overlap cell (0,2) must keep height 1 to match the untouched T2.
        replacement = {
            "tile_id": "T1",
            "revision": 2,
            "row": 0,
            "col": 0,
            "values": [[0, 3, 1]],
        }
        ok = self.catalog.apply_batch([replacement])
        self.assertTrue(ok.ok)
        self.assertEqual(ok.version, 2)
        surface = self.catalog.get_slice(2)
        self.assertEqual([list(r) for r in surface.heights], [[0, 3, 1, 4, 0]])
        # A later failed batch leaves version 2 intact.
        fail = self.catalog.apply_batch(
            [
                {
                    "tile_id": "T2",
                    "revision": 9,
                    "row": 0,
                    "col": 2,
                    "values": [[9, 9, 9]],
                }
            ]
        )
        self.assertFalse(fail.ok)
        self.assertEqual(self.catalog.get_slice(2).elevation_hash, surface.elevation_hash)

    def test_immutable_revision(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        clash = {
            "tile_id": "T1",
            "revision": 1,
            "row": 0,
            "col": 0,
            "values": [[7, 7, 7]],
        }
        result = self.catalog.apply_batch([clash])
        self.assertFalse(result.ok)
        self.assertIn("immutable_revision", {e.code for e in result.errors})

    def test_reopen_catalog_from_disk(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        path = self.dir / "catalog.db"
        self.catalog.close()
        self.catalog = Catalog.open(path)
        self.assertEqual(self.catalog.list_versions(), [1])
        self.assertEqual([list(r) for r in self.catalog.get_slice(1).heights], [[0, 4, 1, 4, 0]])


class WaterTests(Base):
    def test_initial_two_outlets(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        surface = self.catalog.get_slice(1)
        water = self.explicit_water(surface, ["L", "R"])
        self.assertEqual([list(r) for r in water.levels], [[1, 4, 4, 4, 3]])
        self.assertEqual([list(r) for r in water.storage], [[1, 0, 3, 0, 3]])
        self.assertEqual(water.total_storage, 7)

    def test_lowest_sill_alone_is_not_enough(self):
        # With only R, the high middle cell blocks more drainage than the low
        # sill at L would suggest; the lowest-sill shortcut would be wrong.
        self.catalog.apply_batch(self.material["initial_tiles"])
        surface = self.catalog.get_slice(1)
        water = self.explicit_water(surface, ["R"])
        self.assertEqual([list(r) for r in water.levels], [[4, 4, 4, 4, 3]])
        self.assertEqual(water.storage[0][2], 3)

    def test_corrected_two_outlets(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        self.catalog.apply_batch([self.material["correction"]])
        surface = self.catalog.get_slice(2)
        water = self.explicit_water(surface, ["L", "R"])
        self.assertEqual([list(r) for r in water.levels], [[1, 4, 3, 3, 3]])
        self.assertEqual([list(r) for r in water.storage], [[1, 0, 2, 1, 3]])
        self.assertEqual(water.total_storage, 7)

    def test_reuse_equals_recompute(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        self.catalog.apply_batch([self.material["correction"]])
        surface = self.catalog.get_slice(2)
        outs = self.outlets()
        # Compare like for like: both search every subset and pick the same
        # minimum-storage feasible set, so only the evaluation strategy differs.
        for opens in (0, 1, 2):
            reuse = solve(surface, outs, opens, 999, strategy="reuse", stop_on_first=False)
            recompute = solve(
                surface, outs, opens, 999, strategy="recompute", stop_on_first=False
            )
            self.assertEqual(reuse.water.levels, recompute.water.levels)
            self.assertEqual(reuse.water.storage, recompute.water.storage)
            self.assertEqual([o.id for o in reuse.opened], [o.id for o in recompute.opened])

    def test_cache_invalidated_by_revision(self):
        self.catalog.apply_batch(self.material["initial_tiles"])
        v1 = self.catalog.get_slice(1)
        self.catalog.apply_batch([self.material["correction"]])
        v2 = self.catalog.get_slice(2)
        cache = BottleneckCache()
        cache.get(v1, self.outlets()[1])
        first_hash = cache.current_hash
        cache.get(v2, self.outlets()[1])
        self.assertNotEqual(cache.current_hash, first_hash)
        self.assertEqual(cache.invalidations, 1)


class PublisherTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.catalog.apply_batch(self.material["initial_tiles"])
        self.catalog.apply_batch([self.material["correction"]])

    def publish(self, request):
        return self.publisher.publish(request, default_outlets=self.material["outlets"])

    def test_a1_rejected_cell_limit(self):
        result = self.publish(self.material["requests"][0])
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "rejected")
        stored = self.publisher.outcome_for_request("A1")
        attempts = stored["payload"]["evidence"]["attempts"]
        self.assertTrue(attempts)
        # No plan row was written for the rejection.
        self.assertNotIn("A1", {p["request_id"] for p in self.publisher.list_plans()})
        # Both outlets open still has middle-cell storage 3 > 2.
        both = next(a for a in attempts if a["opened"] == ["L", "R"])
        mid = next(
            c["storage"] for c in both["storage_at_cells"] if c["cell"] == [0, 2]
        )
        self.assertEqual(mid, 3)

    def test_a2_published(self):
        result = self.publish(self.material["requests"][1])
        self.assertTrue(result.ok)
        self.assertEqual(result.artifact["opened"], ["L", "R"])
        self.assertEqual(result.artifact["storage"][0][2], 2)
        self.assertEqual(result.artifact["total_storage"], 7)
        self.assertTrue(result.artifact["elevation_source"]["elevation_hash"])

    def test_a3_rejected_total(self):
        result = self.publish(self.material["requests"][2])
        self.assertFalse(result.ok)
        stored = self.publisher.outcome_for_request("A3")
        attempts = stored["payload"]["evidence"]["attempts"]
        self.assertTrue(stored["payload"]["evidence"]["exhaustive"])
        # Every size-0 or size-1 set is present.
        self.assertEqual(len(attempts), 3)
        right = next(a for a in attempts if a["opened"] == ["R"])
        self.assertEqual(right["storage_at_cells"][0]["storage"], 2)
        self.assertTrue(right["violations"])

    def test_replay_same_input_returns_same_artifact(self):
        first = self.publish(self.material["requests"][1])
        second = self.publish(self.material["requests"][1])
        self.assertTrue(second.replayed)
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(first.artifact, second.artifact)

    def test_idempotency_conflict(self):
        self.publish(self.material["requests"][1])
        changed = dict(self.material["requests"][1])
        changed["total_max"] = 6
        from slabwater.publisher import PublishError

        with self.assertRaises(PublishError) as ctx:
            self.publish(changed)
        self.assertEqual(ctx.exception.code, "idempotency_conflict")

    def test_rejection_replay_is_stable(self):
        first = self.publish(self.material["requests"][0])
        second = self.publish(self.material["requests"][0])
        self.assertTrue(second.replayed)
        self.assertEqual(first.status, second.status)

    def test_new_version_does_not_rewrite_old_plan(self):
        plan = self.publish(self.material["requests"][1]).artifact
        lower = dict(self.material["correction"])
        lower["revision"] = 5
        lower["values"] = [[1, 0, 0]]
        self.catalog.apply_batch([lower])
        new_version = self.catalog.latest_version()
        self.assertGreater(new_version, 2)
        self.assertEqual(self.publisher.get_plan(plan["plan_id"]), plan)
        self.assertEqual(
            self.publisher.plan_for_request("A2")["elevation_hash"],
            plan["elevation_hash"],
        )

    def test_verify_and_reopen(self):
        plan = self.publish(self.material["requests"][1]).artifact
        checks = self.publisher.verify(plan["plan_id"])
        self.assertTrue(checks.ok, checks.messages)
        reopened = self.publisher.reopen(plan["plan_id"])
        self.assertEqual(reopened.artifact, plan)

    def test_verify_detects_tampering(self):
        plan = self.publish(self.material["requests"][1]).artifact
        tampered = json.loads(json.dumps(plan))
        tampered["storage"][0][2] = 0
        # Overwrite the stored artifact directly to simulate corruption.
        self.publisher._conn.execute(
            "UPDATE plans SET artifact=? WHERE plan_id=?",
            (json.dumps(tampered), plan["plan_id"]),
        )
        result = self.publisher.verify(plan["plan_id"])
        self.assertFalse(result.ok)
        self.assertFalse(result.checks["storage_match"])

    def test_unknown_source_revision_errors_without_plan(self):
        request = dict(self.material["requests"][1])
        request["id"] = "MISSING"
        request["source_revision"] = 99
        result = self.publish(request)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "error")
        self.assertNotIn("MISSING", {p["request_id"] for p in self.publisher.list_plans()})

    def test_outlet_out_of_bounds(self):
        # An outlet cell outside the surface must be rejected.  (Every cell of
        # this 1x5 surface is on the boundary, so use an off-surface cell.)
        request = dict(self.material["requests"][1])
        request["id"] = "OFFSURFACE"
        request["outlets"] = [{"id": "M", "cell": [0, 9], "sill": 0}]
        result = self.publish(request)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "error")


class TwelveByTwelveTests(unittest.TestCase):
    def test_large_material_runs_and_covers(self):
        material = make_material(seed=772)
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Catalog(
                Path(tmp) / "catalog.db",
                material["height"],
                material["width"],
                material["ceiling"],
            )
            publisher = Publisher(Path(tmp) / "plans.db", catalog)
            try:
                first = catalog.apply_batch(material["initial_tiles"])
                self.assertTrue(first.ok)
                self.assertEqual(first.overlap_cells, material["height"])
                second = catalog.apply_batch([material["correction"]])
                self.assertTrue(second.ok)
                surface = catalog.get_slice(second.version)
                self.assertEqual(len(surface.heights), 12)
                self.assertEqual(len(surface.heights[0]), 12)
                # Reuse and recompute must agree at 8 outlets.
                outs = [Outlet.from_dict(o) for o in material["outlets"]]
                cache = BottleneckCache()
                reuse = solve(surface, outs, 8, 20_000, strategy="reuse", cache=cache,
                              stop_on_first=False)
                recompute = solve(surface, outs, 8, 20_000, strategy="recompute",
                                  stop_on_first=False)
                self.assertEqual(reuse.water.levels, recompute.water.levels)
                self.assertGreater(cache.hits, 0)
                for request in material["requests"]:
                    publisher.publish(request, default_outlets=material["outlets"])
                # S3 asks for total_max 0 with one outlet: always rejected, but
                # evidence must be exhaustive.
                stored = publisher.outcome_for_request("S3")
                self.assertEqual(stored["status"], "rejected")
                self.assertTrue(stored["payload"]["evidence"]["exhaustive"])
            finally:
                publisher.close()
                catalog.close()


if __name__ == "__main__":
    unittest.main()
