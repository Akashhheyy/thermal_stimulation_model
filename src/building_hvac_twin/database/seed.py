"""Idempotent database initialization from EXISTING project data.

Seeds the ``designs`` collection from the shelter ML dataset CSV and the
``weather_scenarios`` collection from the dataset metadata JSON.  Both use
upserts keyed by their natural ids, so running the seed twice never creates
duplicates.  Source files are only read, never modified; no weather, design
or target values are invented.

Usage (from the repository root)::

    python -m building_hvac_twin.database.seed

Environment:
    MONGODB_URI          required (no default credentials)
    MONGODB_DATABASE     optional, default building_energy_hvac_twin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .collections import design_document, scenario_document
from .connection import MongoSettings, connect, settings_from_env
from .repositories import RepositoryBundle, build_repositories

__all__ = [
    "seed_designs",
    "seed_scenarios",
    "seed_database",
    "main",
]

DEFAULT_DATASET_PATH = Path("data") / "shelter_ml_dataset.csv"
DEFAULT_METADATA_PATH = Path("data") / "shelter_ml_dataset_metadata.json"


def _resolve_path(env_var: str, default: Path) -> Path:
    """Resolve a data path against the CWD first, then the repository root."""
    import os

    configured = os.environ.get(env_var)
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[3]
    for base in (Path.cwd(), repo_root):
        candidate = base / default
        if candidate.exists():
            return candidate
    return Path.cwd() / default


def _design_parameter_columns() -> tuple[str, ...]:
    """Existing design parameter columns from the shelter dataset module."""
    from ..shelter.ml_dataset import DESIGN_PARAMETER_COLUMNS

    return DESIGN_PARAMETER_COLUMNS


def seed_designs(design_repository, dataset_path: Path | str) -> int:
    """Upsert every existing design from the dataset CSV (read-only)."""
    import pandas as pd

    frame = pd.read_csv(dataset_path)
    # Design parameters are constant across the scenario rows of a design;
    # keep the first row per design_id.
    first_rows = frame.drop_duplicates(subset="design_id", keep="first")
    first_rows = first_rows.sort_values("design_id")
    documents = [
        design_document(row.to_dict(), _design_parameter_columns())
        for _, row in first_rows.iterrows()
    ]
    design_repository.ensure_indexes()
    return design_repository.upsert_many(documents)


def seed_scenarios(scenario_repository, metadata_path: Path | str) -> int:
    """Upsert every existing NASA POWER scenario from the metadata (read-only)."""
    metadata_path = Path(metadata_path)
    metadata: dict[str, Any] = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    nasa_power = metadata.get("nasa_power", {})
    used = metadata.get("weather_scenarios", {}).get("used", [])
    documents = [
        scenario_document(scenario, nasa_power) for scenario in used
    ]
    scenario_repository.ensure_indexes()
    return scenario_repository.upsert_many(documents)


def seed_database(
    bundle: RepositoryBundle,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> dict[str, int]:
    """Seed designs and scenarios; idempotent by natural key upserts."""
    designs = seed_designs(bundle.designs, dataset_path)
    scenarios = seed_scenarios(bundle.weather_scenarios, metadata_path)
    return {
        "designs": bundle.designs.count(),
        "weather_scenarios": bundle.weather_scenarios.count(),
        "designs_upserted": designs,
        "scenarios_upserted": scenarios,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="building_hvac_twin.database.seed",
        description=(
            "Seed MongoDB with the existing shelter designs and NASA POWER "
            "weather scenarios. Idempotent: safe to run repeatedly."
        ),
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="MongoDB URI; falls back to the MONGODB_URI environment variable",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Database name; falls back to MONGODB_DATABASE",
    )
    parser.add_argument("--dataset", default=None, help="Path to the ML dataset CSV")
    parser.add_argument("--metadata", default=None, help="Path to the dataset metadata JSON")
    args = parser.parse_args(argv)

    settings: MongoSettings = settings_from_env(args.uri, args.database)
    if not settings.configured:
        print(
            "error: MONGODB_URI is not set. Provide --uri or set the "
            "MONGODB_URI environment variable (see .env.example).",
            file=sys.stderr,
        )
        return 2

    dataset_path = (
        Path(args.dataset)
        if args.dataset
        else _resolve_path("BHVAC_DATASET_PATH", DEFAULT_DATASET_PATH)
    )
    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else _resolve_path("BHVAC_METADATA_PATH", DEFAULT_METADATA_PATH)
    )

    client, database = connect(settings)
    try:
        bundle = build_repositories(
            client, database, settings.database_name
        )
        summary = seed_database(bundle, dataset_path, metadata_path)
    except Exception as exc:  # connection or read failure: report clearly
        print(f"error: seeding failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(
        f"seeded database {settings.database_name!r}: "
        f"{summary['designs']} designs, "
        f"{summary['weather_scenarios']} weather scenarios "
        f"(upserted {summary['designs_upserted']} design rows, "
        f"{summary['scenarios_upserted']} scenario rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
