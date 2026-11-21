"""Command-line driver for the synthetic catalog/publisher.

Examples::

    python -m slabwater.cli run materials/772/input.json
    python -m slabwater.cli verify <db_dir> <plan_id>
    python -m slabwater.cli reopen <db_dir> <plan_id>

The ``run`` subcommand builds the catalog from the material's tile batches,
processes every request and prints a JSON report.  It is a demonstration
harness, not a service.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .publisher import Publisher


def run_material(path: str | Path, db_dir: str | Path | None = None) -> dict[str, Any]:
    material = json.loads(Path(path).read_text(encoding="utf-8"))
    db_dir = Path(db_dir) if db_dir else Path(tempfile.mkdtemp(prefix="slabwater-"))
    db_dir.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(db_dir / "catalog.db", material["height"], material["width"], material["ceiling"])
    publisher = Publisher(db_dir / "plans.db", catalog)
    report: dict[str, Any] = {"db_dir": str(db_dir), "batches": [], "requests": []}
    try:
        initial = catalog.apply_batch(material["initial_tiles"])
        report["batches"].append(
            {"name": "initial", "batch": initial.to_dict(), "version": initial.version}
        )
        if "correction" in material:
            correction = catalog.apply_batch([material["correction"]])
            report["batches"].append(
                {"name": "correction", "batch": correction.to_dict(), "version": correction.version}
            )
        if "invalid_correction" in material:
            invalid = catalog.apply_batch([material["invalid_correction"]])
            report["batches"].append(
                {
                    "name": "invalid_correction",
                    "batch": invalid.to_dict(),
                    "version": invalid.version,
                }
            )
        report["versions"] = catalog.list_versions()
        for request in material["requests"]:
            result = publisher.publish(request, default_outlets=material.get("outlets"))
            entry: dict[str, Any] = {
                "request": request,
                "status": result.status,
                "replayed": result.replayed,
                "plan_id": result.plan_id,
                "artifact": result.artifact,
            }
            if result.status == "rejected":
                stored = publisher.outcome_for_request(request["id"])
                entry["evidence"] = stored["payload"]["evidence"]
            if result.status == "error":
                entry["error"] = publisher.outcome_for_request(request["id"])["payload"]
            report["requests"].append(entry)
        # Reject-on-invalid surface check: the invalid correction must not have
        # produced a version; list_versions above is the evidence.
        report["plans"] = publisher.list_plans()
    finally:
        publisher.close()
        catalog.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slabwater", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a material input file end to end")
    run_parser.add_argument("input", help="path to input.json")
    run_parser.add_argument("--db-dir", default=None, help="directory for catalog.db/plans.db")

    verify_parser = sub.add_parser("verify", help="verify a stored plan")
    verify_parser.add_argument("db_dir")
    verify_parser.add_argument("plan_id")

    reopen_parser = sub.add_parser("reopen", help="re-open a stored plan artifact")
    reopen_parser.add_argument("db_dir")
    reopen_parser.add_argument("plan_id")

    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_material(args.input, args.db_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    db_dir = Path(args.db_dir)
    catalog = Catalog.open(db_dir / "catalog.db")
    publisher = Publisher(db_dir / "plans.db", catalog)
    try:
        if args.command == "verify":
            result = publisher.verify(args.plan_id).to_dict()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 1
        reopened = publisher.reopen(args.plan_id)
        print(json.dumps(reopened.to_dict(), indent=2, ensure_ascii=False))
        return 0
    finally:
        publisher.close()
        catalog.close()


if __name__ == "__main__":
    sys.exit(main())
