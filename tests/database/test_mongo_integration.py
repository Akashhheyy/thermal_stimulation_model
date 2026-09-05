"""Optional live MongoDB integration test.

Skipped unless the environment variable ``MONGODB_TEST_URI`` is set, for
example::

    set MONGODB_TEST_URI=mongodb://localhost:27017
    python -m pytest tests/database/test_mongo_integration.py -v

The test seeds a throwaway database (name suffix ``_test``), verifies the
catalogs, and drops it afterwards.  It never touches the user's real
application database and no credentials are read from this repository.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from building_hvac_twin.database import connect, build_repositories
from building_hvac_twin.database.connection import MongoSettings
from building_hvac_twin.database.seed import seed_database

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "shelter_ml_dataset.csv"
METADATA_PATH = REPO_ROOT / "data" / "shelter_ml_dataset_metadata.json"

pytestmark = pytest.mark.skipif(
    not os.environ.get("MONGODB_TEST_URI"),
    reason="set MONGODB_TEST_URI to run the live MongoDB integration test",
)


def test_live_seed_and_catalog_roundtrip():
    settings = MongoSettings(
        uri=os.environ["MONGODB_TEST_URI"],
        database_name="building_energy_hvac_twin_test",
    )
    client, database = connect(settings)
    try:
        bundle = build_repositories(
            client, database, settings.database_name
        )
        summary = seed_database(bundle, DATASET_PATH, METADATA_PATH)
        assert summary["designs"] == 300
        assert summary["weather_scenarios"] == 10

        designs = bundle.designs.list(limit=5)
        assert [d["design_id"] for d in designs] == [
            "D0000", "D0001", "D0002", "D0003", "D0004",
        ]
        scenarios = bundle.weather_scenarios.list()
        assert "S01_winter" in [s["scenario_id"] for s in scenarios]
        assert bundle.ping() is True
    finally:
        client.drop_database(settings.database_name)
        client.close()
