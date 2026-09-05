"""MongoDB connection helpers.

Configuration comes exclusively from environment variables so no credentials
are ever hard-coded:

- ``MONGODB_URI``: for example ``mongodb://localhost:27017`` (or with
  credentials ``mongodb://user:password@localhost:27017``).  When unset the
  application runs without a database and database-backed endpoints answer
  with a clear error instead of fake data.
- ``MONGODB_DATABASE``: database name, default ``building_energy_hvac_twin``.

MongoDB is application/persistence storage only.  The ML dataset CSV, the
trained model artifacts and the NASA POWER raw weather cache stay in their
existing project locations and are never copied into MongoDB by this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_DATABASE_NAME",
    "MongoSettings",
    "settings_from_env",
    "connect",
    "read_env_file",
]

DEFAULT_DATABASE_NAME = "building_energy_hvac_twin"


@dataclass(frozen=True)
class MongoSettings:
    """Connection settings resolved from the environment."""

    uri: str | None
    database_name: str

    @property
    def configured(self) -> bool:
        return bool(self.uri)


def settings_from_env(
    uri: str | None = None,
    database_name: str | None = None,
) -> MongoSettings:
    """Build settings from explicit values or the environment.

    Explicit arguments win over environment variables; the database name has
    a documented default while the URI never does (no defaults credentials).
    """
    return MongoSettings(
        uri=uri if uri is not None else os.environ.get("MONGODB_URI"),
        database_name=(
            database_name
            if database_name is not None
            else os.environ.get("MONGODB_DATABASE", DEFAULT_DATABASE_NAME)
        ),
    )


def connect(settings: MongoSettings):
    """Create a ``pymongo`` client and database handle.

    The client is created lazily by the driver; a failed connection is
    surfaced on first operation or by the health ping, never hidden.  Callers
    must close the returned client on shutdown (``client.close()``).
    """
    if not settings.configured:
        raise ValueError("MONGODB_URI is not configured")
    import pymongo

    client = pymongo.MongoClient(settings.uri, serverSelectionTimeoutMS=5000)
    return client, client[settings.database_name]


def read_env_file(path: Path | str = ".env") -> dict[str, str]:
    """Read KEY=VALUE pairs from a dotenv file if it exists.

    This avoids a python-dotenv dependency: only simple uncommented
    assignments are recognised, quotes are stripped.  The FastAPI entry point
    can alternatively be started with ``uvicorn --env-file .env``.
    """
    path = Path(path)
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values
