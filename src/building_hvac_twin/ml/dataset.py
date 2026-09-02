"""Load and validate the shelter ML dataset with clear failure modes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, ML_TARGETS

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_METADATA_PATH",
    "REQUIRED_DATASET_COLUMNS",
    "MLBundle",
    "load_dataset",
]

DEFAULT_DATASET_PATH = Path("data") / "shelter_ml_dataset.csv"
DEFAULT_METADATA_PATH = Path("data") / "shelter_ml_dataset_metadata.json"

# Columns the ML pipeline requires from the dataset.  performance_score is
# deliberately excluded and must not be present as a usable target.
REQUIRED_DATASET_COLUMNS = (
    "design_id",
    "weather_scenario_id",
    "window_u_value_w_m2k",
    "door_area_m2",
    "window_solar_heat_gain_coefficient",
    "thermal_mass_heat_capacity_j_k",
    "net_wall_area_m2",
    "mean_outdoor_temperature_c",
    "minimum_outdoor_temperature_c",
    "maximum_outdoor_temperature_c",
    "daily_solar_sum_wh_m2",
    *ML_TARGETS,
)

FORBIDDEN_TARGETS = ("performance_score", "auxiliary_heating_kwh", "auxiliary_cooling_kwh")

NASA_PROVENANCE_PREFIX = "NASA POWER data are satellite"


@dataclass
class MLBundle:
    """The validated dataset frame together with its metadata document."""

    frame: pd.DataFrame
    metadata: dict

    @property
    def provenance_statement(self) -> str:
        return str(self.metadata["nasa_power"]["provenance_statement"])


def _require(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def validate_dataset(
    frame: pd.DataFrame,
    metadata: dict,
    require_live_retrieval: bool = True,
) -> list[str]:
    """Return a list of problems; an empty list means the dataset is usable.

    ``require_live_retrieval`` is True in production load_dataset() and may be
    set False only by offline tests that build a metadata document from a mock
    NASA transport.
    """
    problems: list[str] = []
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in frame.columns]
    _require(not missing, f"missing required columns: {missing}", problems)
    if missing:
        return problems
    if frame.empty:
        problems.append("dataset frame is empty")
        return problems

    for column in REQUIRED_DATASET_COLUMNS:
        if frame[column].isna().any():
            problems.append(f"required column {column!r} contains NaN")
    used_columns = list(REQUIRED_DATASET_COLUMNS)
    numeric = frame[used_columns].select_dtypes(include=[np.number])
    for column in numeric.columns:
        if not np.isfinite(numeric[column].to_numpy(dtype=float)).all():
            problems.append(f"numeric column {column!r} contains non-finite values")

    duplicates = int(frame.duplicated(subset=["design_id", "weather_scenario_id"]).sum())
    _require(duplicates == 0, f"{duplicates} duplicate design x scenario rows", problems)

    expected_rows = metadata.get("dataset", {}).get("row_count")
    _require(
        expected_rows is None or int(expected_rows) == len(frame),
        f"metadata row_count {expected_rows} does not match dataset rows {len(frame)}",
        problems,
    )

    provenance = str(
        metadata.get("nasa_power", {}).get("provenance_statement", "")
    )
    _require(
        provenance.startswith(NASA_PROVENANCE_PREFIX),
        "dataset metadata does not carry the NASA POWER satellite/reanalysis "
        "provenance statement",
        problems,
    )
    retrieval_mode = str(metadata.get("nasa_power", {}).get("retrieval_mode", ""))
    if require_live_retrieval:
        _require(
            retrieval_mode == "live",
            f"dataset weather was not retrieved live from NASA POWER (mode: {retrieval_mode!r})",
            problems,
        )

    return problems


def load_dataset(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> MLBundle:
    """Load the dataset and metadata, validating both; raise on any problem."""
    dataset_path = Path(dataset_path)
    metadata_path = Path(metadata_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"shelter ML dataset not found at {dataset_path}; generate it first "
            f"with scripts/generate_ml_dataset.py"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset metadata not found at {metadata_path}")

    frame = pd.read_csv(dataset_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    problems = validate_dataset(frame, metadata)
    if problems:
        raise ValueError(
            "shelter ML dataset is malformed: " + "; ".join(problems)
        )

    present_forbidden = [c for c in FORBIDDEN_TARGETS if c in frame.columns]
    if present_forbidden:
        raise ValueError(
            f"dataset unexpectedly contains forbidden ML target columns: "
            f"{present_forbidden}"
        )

    valid_materials = set(frame["wall_material"].unique())
    if not valid_materials:
        raise ValueError("dataset contains no wall materials")
    return MLBundle(frame=frame, metadata=metadata)
