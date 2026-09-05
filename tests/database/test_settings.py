"""Focused tests for database configuration precedence.

Required order: explicit arguments > process environment > ``.env`` file >
defaults.  ``MONGODB_DATABASE`` falls back to ``DEFAULT_DATABASE_NAME``;
``MONGODB_URI`` never has a hardcoded default.

The repository-root ``.env`` (which may hold the developer's real values) is
isolated in every test via ``REPO_DOTENV_PATH``, so these tests neither
depend on nor expose real credentials.  All values used here are fakes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from building_hvac_twin.database import connection
from building_hvac_twin.database.connection import (
    DEFAULT_DATABASE_NAME,
    read_env_file,
    settings_from_env,
)

FAKE_DOTENV_URI = "mongodb://fake-from-dotenv:27017"
FAKE_ENV_URI = "mongodb://fake-from-env:27017"
FAKE_ARG_URI = "mongodb://fake-from-argument:27017"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove the real variables so tests are deterministic."""
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)


@pytest.fixture
def isolated_dotenv(monkeypatch, tmp_path) -> Path:
    """Point the .env lookup at tmp_path; the real repo .env is ignored.

    The repository-root candidate is redirected to a nonexistent path and
    the CWD is moved into ``tmp_path``, where tests may write ``.env``.
    """
    monkeypatch.setattr(connection, "REPO_DOTENV_PATH", tmp_path / "no" / "repo" / ".env")
    monkeypatch.chdir(tmp_path)
    return tmp_path / ".env"


def write_dotenv(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_settings_loaded_from_dotenv_when_env_absent(isolated_dotenv):
    write_dotenv(
        isolated_dotenv,
        [
            "# comment line",
            f"MONGODB_URI={FAKE_DOTENV_URI}",
            "MONGODB_DATABASE=dotenv_db",
        ],
    )
    settings = settings_from_env()
    assert settings.configured is True
    assert settings.uri == FAKE_DOTENV_URI
    assert settings.database_name == "dotenv_db"


def test_environment_variables_override_dotenv(isolated_dotenv, monkeypatch):
    write_dotenv(
        isolated_dotenv,
        [
            f"MONGODB_URI={FAKE_DOTENV_URI}",
            "MONGODB_DATABASE=dotenv_db",
        ],
    )
    monkeypatch.setenv("MONGODB_URI", FAKE_ENV_URI)
    settings = settings_from_env()
    assert settings.uri == FAKE_ENV_URI
    # Database name still comes from .env; only the URI was overridden.
    assert settings.database_name == "dotenv_db"


def test_explicit_arguments_override_environment_and_dotenv(
    isolated_dotenv, monkeypatch
):
    write_dotenv(
        isolated_dotenv,
        [
            f"MONGODB_URI={FAKE_DOTENV_URI}",
            "MONGODB_DATABASE=dotenv_db",
        ],
    )
    monkeypatch.setenv("MONGODB_URI", FAKE_ENV_URI)
    monkeypatch.setenv("MONGODB_DATABASE", "env_db")
    settings = settings_from_env(
        uri=FAKE_ARG_URI,
        database_name="explicit_db",
    )
    assert settings.uri == FAKE_ARG_URI
    assert settings.database_name == "explicit_db"
    assert settings.configured is True


def test_missing_dotenv_and_missing_uri_is_not_configured(isolated_dotenv):
    # No environment variables (fixture), no .env file at all.
    settings = settings_from_env()
    assert settings.configured is False
    assert settings.uri is None
    # The database name still falls back to the documented default.
    assert settings.database_name == DEFAULT_DATABASE_NAME


def test_database_env_variable_never_provides_a_uri(isolated_dotenv, monkeypatch):
    # Even with MONGODB_DATABASE set, a missing URI means "not configured".
    monkeypatch.setenv("MONGODB_DATABASE", "some_db")
    settings = settings_from_env()
    assert settings.configured is False
    assert settings.uri is None
    assert settings.database_name == "some_db"


def test_repository_root_dotenv_used_when_cwd_has_none(
    monkeypatch, tmp_path
):
    repo_dotenv = tmp_path / "repo" / ".env"
    write_dotenv(
        repo_dotenv,
        [f"MONGODB_URI={FAKE_DOTENV_URI}", "MONGODB_DATABASE=repo_db"],
    )
    monkeypatch.setattr(connection, "REPO_DOTENV_PATH", repo_dotenv)
    monkeypatch.chdir(tmp_path)  # tmp_path/.env does not exist
    settings = settings_from_env()
    assert settings.configured is True
    assert settings.uri == FAKE_DOTENV_URI
    assert settings.database_name == "repo_db"


def test_cwd_dotenv_wins_over_repository_root_dotenv(monkeypatch, tmp_path):
    repo_dotenv = tmp_path / "repo" / ".env"
    write_dotenv(
        repo_dotenv,
        [f"MONGODB_URI={FAKE_DOTENV_URI}", "MONGODB_DATABASE=repo_db"],
    )
    monkeypatch.setattr(connection, "REPO_DOTENV_PATH", repo_dotenv)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    write_dotenv(
        cwd / ".env",
        [f"MONGODB_URI={FAKE_ENV_URI}", "MONGODB_DATABASE=cwd_db"],
    )
    settings = settings_from_env()
    assert settings.uri == FAKE_ENV_URI
    assert settings.database_name == "cwd_db"


def test_read_env_file_parses_simple_assignments(tmp_path):
    dotenv = tmp_path / ".env"
    write_dotenv(
        dotenv,
        [
            "# a comment",
            "",
            "PLAIN=value",
            "QUOTED='single quoted'",
            "DQUOTED=\"double quoted\"",
            "SPACED = spaced value ",
            "NO_EQUALS_IGNORED",
        ],
    )
    values = read_env_file(dotenv)
    assert values == {
        "PLAIN": "value",
        "QUOTED": "single quoted",
        "DQUOTED": "double quoted",
        "SPACED": "spaced value",
    }


def test_read_env_file_missing_file_returns_empty(tmp_path):
    assert read_env_file(tmp_path / "missing.env") == {}
