"""Plan publisher and verifier built on a frozen elevation catalog.

The publisher never mutates the catalog.  A plan binds to one catalog version
(and therefore one ``elevation_hash``); later catalog versions leave old plans
byte-for-byte intact.

Idempotency: a request id is tied to the exact canonical input it was first
seen with.  Replaying the same id with the same input returns the stored outcome
(a published artifact for success, the stored evidence for failure).  Reusing an
id with a different input is an error rather than a silent overwrite.  Failed
requests never create a plan row.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import Catalog
from .model import (
    Cell,
    Outlet,
    Slice,
    is_int,
    sha256_hex,
)
from .solver import BottleneckCache, SolveResult, _materialize, _run_minimax, solve

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    request_id  TEXT PRIMARY KEY,
    input_hash  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('published','rejected','error')),
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS plans (
    plan_id       TEXT PRIMARY KEY,
    request_id    TEXT NOT NULL UNIQUE REFERENCES outcomes(request_id),
    input_hash    TEXT NOT NULL,
    catalog_version INTEGER NOT NULL,
    elevation_hash  TEXT NOT NULL,
    artifact      TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class PublishError(Exception):
    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "message": self.message, "details": self.details}


@dataclass
class PublishResult:
    ok: bool
    request_id: str
    status: str
    artifact: dict[str, Any] | None
    plan_id: str | None
    replayed: bool
    solve: SolveResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request_id": self.request_id,
            "status": self.status,
            "plan_id": self.plan_id,
            "replayed": self.replayed,
            "artifact": self.artifact,
            "solve": self.solve.evidence() if self.solve else None,
        }


@dataclass
class VerifyResult:
    ok: bool
    plan_id: str | None
    checks: dict[str, bool]
    messages: list[str]
    artifact: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "plan_id": self.plan_id,
            "checks": self.checks,
            "messages": self.messages,
            "artifact": self.artifact,
        }


def _parse_outlets(raw_outlets: Sequence[Mapping[str, Any]]) -> list[Outlet]:
    outlets: list[Outlet] = []
    seen: set[str] = set()
    for raw in raw_outlets:
        outlet = Outlet.from_dict(raw)
        if outlet.id in seen:
            raise PublishError("duplicate_outlet", f"outlet id {outlet.id} is repeated")
        seen.add(outlet.id)
        outlets.append(outlet)
    return sorted(outlets, key=lambda outlet: outlet.id)


def _hide_cells(mapping: Mapping[str, Any]) -> list[list[int]]:
    return sorted((list(cell) for cell in mapping), key=lambda c: (c[0], c[1]))


class Publisher:
    def __init__(self, path: str | Path, catalog: Catalog):
        self.path = str(path)
        self.catalog = catalog
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Publisher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading an elevation surface -------------------------------------
    def read_elevation(self, version: int | None = None, include_cells: bool = True) -> dict[str, Any]:
        surface = self.catalog.get_slice(version)
        return surface.elevation_source(include_cells=include_cells)

    # -- request normalization --------------------------------------------
    def _normalize_request(
        self,
        request: Mapping[str, Any],
        default_outlets: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise PublishError("bad_request", "request must be an object")
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise PublishError("bad_request", "request id must be a non-empty string")
        source_revision = request.get("source_revision")
        if not is_int(source_revision):
            raise PublishError("bad_request", "source_revision must be an integer version")
        opens = request.get("opens")
        if not is_int(opens) or opens < 0:
            raise PublishError("bad_request", "opens must be a non-negative integer")
        total_max = request.get("total_max")
        if not is_int(total_max) or total_max < 0:
            raise PublishError("bad_request", "total_max must be a non-negative integer")
        raw_cell_max = request.get("cell_max", [])
        if not isinstance(raw_cell_max, list):
            raise PublishError("bad_request", "cell_max must be a list")
        cell_max: list[tuple[Cell, int]] = []
        for item in raw_cell_max:
            if not isinstance(item, Mapping):
                raise PublishError("bad_request", "cell_max entries must be objects")
            cell = item.get("cell")
            limit = item.get("max")
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise PublishError("bad_request", "cell_max cell must be [row, col]")
            r, c = cell
            if not is_int(r) or not is_int(c) or not is_int(limit):
                raise PublishError("bad_request", "cell_max cell and max must be integers")
            cell_max.append(((r, c), limit))
        cell_max.sort(key=lambda pair: (pair[0][0], pair[0][1]))
        raw_outlets = request.get("outlets", default_outlets)
        if raw_outlets is None:
            raise PublishError("bad_request", "no candidate outlets supplied")
        outlets = _parse_outlets(raw_outlets)
        return {
            "id": request_id,
            "source_revision": source_revision,
            "opens": opens,
            "total_max": total_max,
            "cell_max": cell_max,
            "outlets": outlets,
        }

    def _input_hash(self, norm: Mapping[str, Any], surface: Slice | None) -> str:
        return sha256_hex(
            {
                "request_id": norm["id"],
                "source_revision": norm["source_revision"],
                "elevation_hash": surface.elevation_hash if surface is not None else None,
                "opens": norm["opens"],
                "total_max": norm["total_max"],
                "cell_max": [
                    {"cell": [cell[0], cell[1]], "max": limit}
                    for cell, limit in norm["cell_max"]
                ],
                "outlets": [outlet.to_dict() for outlet in norm["outlets"]],
            }
        )

    def _validate_geometry(self, surface: Slice, norm: Mapping[str, Any]) -> None:
        for (r, c), _ in norm["cell_max"]:
            if not (0 <= r < surface.height and 0 <= c < surface.width):
                raise PublishError(
                    "cell_out_of_bounds", f"limit cell ({r},{c}) is outside the surface"
                )
        for outlet in norm["outlets"]:
            r, c = outlet.cell
            if not (0 <= r < surface.height and 0 <= c < surface.width):
                raise PublishError(
                    "outlet_out_of_bounds",
                    f"outlet {outlet.id} cell ({r},{c}) is outside the surface",
                )
            if not surface.is_boundary(r, c):
                raise PublishError(
                    "outlet_not_boundary",
                    f"outlet {outlet.id} cell ({r},{c}) is not on the boundary",
                )
            if outlet.sill < 0:
                raise PublishError(
                    "outlet_bad_sill", f"outlet {outlet.id} sill must be non-negative"
                )

    # -- publishing --------------------------------------------------------
    def publish(
        self,
        request: Mapping[str, Any],
        default_outlets: Sequence[Mapping[str, Any]] | None = None,
        strategy: str = "reuse",
        cache: BottleneckCache | None = None,
        stop_on_first: bool = True,
    ) -> PublishResult:
        norm = self._normalize_request(request, default_outlets)
        request_id = norm["id"]
        try:
            surface = self.catalog.get_slice(norm["source_revision"])
        except KeyError as exc:
            # Record an idempotent rejection keyed by the request itself; no
            # elevation hash exists because the version is unknown.
            input_hash = self._input_hash(norm, None)
            replay = self._replay(request_id, input_hash)
            if replay is not None:
                return replay
            outcome = {
                "ok": False,
                "code": "unknown_source_revision",
                "message": str(exc),
                "details": {"source_revision": norm["source_revision"]},
            }
            self._record_outcome(request_id, input_hash, "error", outcome)
            return PublishResult(False, request_id, "error", None, None, False, None)
        input_hash = self._input_hash(norm, surface)
        replay = self._replay(request_id, input_hash)
        if replay is not None:
            return replay
        try:
            self._validate_geometry(surface, norm)
        except PublishError as exc:
            self._record_outcome(request_id, input_hash, "error", exc.to_dict())
            return PublishResult(False, request_id, "error", None, None, False, None)
        result = solve(
            surface=surface,
            outlets=norm["outlets"],
            opens=norm["opens"],
            total_max=norm["total_max"],
            cell_max=norm["cell_max"],
            strategy=strategy,
            cache=cache,
            stop_on_first=stop_on_first,
        )
        if not result.ok:
            payload = {
                "ok": False,
                "code": "rejected",
                "message": "no outlet set satisfies every storage limit",
                "input": self._public_request(norm),
                "elevation": {
                    "version": surface.version,
                    "elevation_hash": surface.elevation_hash,
                },
                "evidence": result.evidence(),
            }
            self._record_outcome(request_id, input_hash, "rejected", payload)
            return PublishResult(False, request_id, "rejected", None, None, False, result)
        assert result.water is not None
        plan_id = "P-" + sha256_hex(
            {
                "request_id": request_id,
                "input_hash": input_hash,
                "opened": [outlet.id for outlet in result.opened],
                "elevation_hash": surface.elevation_hash,
            }
        )[:16]
        artifact = {
            "plan_id": plan_id,
            "request_id": request_id,
            "request": self._public_request(norm),
            "catalog_version": surface.version,
            "elevation_hash": surface.elevation_hash,
            "elevation_source": surface.elevation_source(include_cells=False),
            "opened": [outlet.id for outlet in result.opened],
            "opened_outlets": [outlet.to_dict() for outlet in result.opened],
            "levels": [list(row) for row in result.water.levels],
            "storage": [list(row) for row in result.water.storage],
            "total_storage": result.water.total_storage,
            "limits": {
                "total_max": norm["total_max"],
                "cell_max": [
                    {"cell": [cell[0], cell[1]], "max": limit}
                    for cell, limit in norm["cell_max"]
                ],
            },
            "solver": result.stats.to_dict(),
            "artifact_hash": sha256_hex(
                {
                    "opened": [outlet.id for outlet in result.opened],
                    "levels": [list(row) for row in result.water.levels],
                    "storage": [list(row) for row in result.water.storage],
                    "total_storage": result.water.total_storage,
                    "elevation_hash": surface.elevation_hash,
                }
            ),
        }
        self._record_plan(plan_id, request_id, input_hash, surface, artifact)
        return PublishResult(True, request_id, "published", artifact, plan_id, False, result)

    def _public_request(self, norm: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": norm["id"],
            "source_revision": norm["source_revision"],
            "opens": norm["opens"],
            "total_max": norm["total_max"],
            "cell_max": [
                {"cell": [cell[0], cell[1]], "max": limit}
                for cell, limit in norm["cell_max"]
            ],
            "outlets": [outlet.to_dict() for outlet in norm["outlets"]],
        }

    # -- storage -----------------------------------------------------------
    def _replay(self, request_id: str, input_hash: str) -> PublishResult | None:
        row = self._conn.execute(
            "SELECT input_hash, status, payload FROM outcomes WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        stored_hash, status, payload_json = row
        if stored_hash != input_hash:
            raise PublishError(
                "idempotency_conflict",
                f"request id {request_id} was already used with different input",
                {"request_id": request_id},
            )
        payload = json.loads(payload_json)
        if status == "published":
            plan = self._conn.execute(
                "SELECT plan_id, artifact FROM plans WHERE request_id=?", (request_id,)
            ).fetchone()
            if plan is None:
                raise PublishError(
                    "inconsistent_store", f"published outcome {request_id} has no plan row"
                )
            return PublishResult(
                True, request_id, "published", json.loads(plan[1]), plan[0], True, None
            )
        return PublishResult(
            False, request_id, status, payload, None, True, None
        )

    def _record_outcome(
        self, request_id: str, input_hash: str, status: str, payload: Mapping[str, Any]
    ) -> None:
        self._conn.execute(
            "INSERT INTO outcomes(request_id,input_hash,status,payload) VALUES (?,?,?,?)",
            (request_id, input_hash, status, json.dumps(payload, sort_keys=True)),
        )

    def _record_plan(
        self, plan_id: str, request_id: str, input_hash: str, surface: Slice, artifact: Mapping[str, Any]
    ) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "INSERT INTO outcomes(request_id,input_hash,status,payload) VALUES (?,?,?,?)",
                (
                    request_id,
                    input_hash,
                    "published",
                    json.dumps({"ok": True, "plan_id": plan_id}, sort_keys=True),
                ),
            )
            self._conn.execute(
                "INSERT INTO plans(plan_id,request_id,input_hash,catalog_version,"
                "elevation_hash,artifact) VALUES (?,?,?,?,?,?)",
                (
                    plan_id,
                    request_id,
                    input_hash,
                    surface.version,
                    surface.elevation_hash,
                    json.dumps(artifact, sort_keys=True),
                ),
            )
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # -- retrieval / reopen ------------------------------------------------
    def get_plan(self, plan_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT artifact FROM plans WHERE plan_id=?", (plan_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such plan: {plan_id}")
        return json.loads(row[0])

    def plan_for_request(self, request_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT artifact FROM plans WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no published plan for request: {request_id}")
        return json.loads(row[0])

    def outcome_for_request(self, request_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT status, payload FROM outcomes WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no outcome for request: {request_id}")
        return {"status": row[0], "payload": json.loads(row[1])}

    def list_plans(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT plan_id, request_id, catalog_version, elevation_hash, created_at "
            "FROM plans ORDER BY plan_id"
        ).fetchall()
        return [
            {
                "plan_id": r[0],
                "request_id": r[1],
                "catalog_version": r[2],
                "elevation_hash": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def reopen(self, plan_id: str) -> PublishResult:
        """Re-open a published plan exactly as stored (no recomputation)."""

        artifact = self.get_plan(plan_id)
        return PublishResult(
            True, artifact["request_id"], "published", artifact, plan_id, True, None
        )

    # -- verification ------------------------------------------------------
    def verify(self, plan_id: str) -> VerifyResult:
        try:
            artifact = self.get_plan(plan_id)
        except KeyError:
            return VerifyResult(False, plan_id, {"exists": False}, ["plan not found"], None)
        checks: dict[str, bool] = {"exists": True}
        messages: list[str] = []
        version = artifact["catalog_version"]
        try:
            surface = self.catalog.get_slice(version)
        except KeyError:
            return VerifyResult(
                False, plan_id, {**checks, "surface_available": False},
                ["catalog version missing"], artifact,
            )
        checks["surface_available"] = True
        checks["elevation_hash_matches"] = surface.elevation_hash == artifact["elevation_hash"]
        if not checks["elevation_hash_matches"]:
            messages.append("frozen elevation hash does not match the plan")
        request = artifact["request"]
        outlets = [Outlet.from_dict(raw) for raw in request["outlets"]]
        by_id = {outlet.id: outlet for outlet in outlets}
        opened_ids = artifact["opened"]
        checks["opened_within_cap"] = len(opened_ids) <= request["opens"]
        checks["opened_known"] = all(oid in by_id for oid in opened_ids)
        checks["opened_unique"] = len(set(opened_ids)) == len(opened_ids)
        if not (checks["opened_within_cap"] and checks["opened_known"] and checks["opened_unique"]):
            messages.append("opened outlet list is invalid for this request")
            return VerifyResult(False, plan_id, checks, messages, artifact)
        opened = [by_id[oid] for oid in opened_ids]
        cell_max = [
            ((item["cell"][0], item["cell"][1]), item["max"])
            for item in request["cell_max"]
        ]
        # Recompute the *specific* published set rather than searching, so the
        # stored numbers are checked against the frozen elevations directly.
        dist, _ = _run_minimax(surface, opened)
        water = _materialize(surface, dist, opened)
        checks["levels_match"] = [list(r) for r in water.levels] == artifact["levels"]
        checks["storage_match"] = [list(r) for r in water.storage] == artifact["storage"]
        checks["total_match"] = water.total_storage == artifact["total_storage"]
        checks["total_within_limit"] = water.total_storage <= request["total_max"]
        cell_ok = True
        for (r, c), limit in cell_max:
            if water.storage[r][c] > limit:
                cell_ok = False
                messages.append(
                    f"cell ({r},{c}) storage {water.storage[r][c]} exceeds limit {limit}"
                )
        checks["cells_within_limit"] = cell_ok
        # Independently search the frozen surface; a published plan should imply
        # at least one feasible set for the same limits.
        independent = solve(
            surface=surface,
            outlets=outlets,
            opens=request["opens"],
            total_max=request["total_max"],
            cell_max=cell_max,
            strategy="recompute",
            stop_on_first=True,
        )
        checks["search_confirmable"] = independent.ok
        if not independent.ok:
            messages.append("independent search found no feasible set for this request")
        if not all(checks.values()):
            if not checks["levels_match"] or not checks["storage_match"] or not checks["total_match"]:
                messages.append("stored water numbers do not reproduce from frozen elevations")
        return VerifyResult(all(checks.values()), plan_id, checks, messages, artifact)
