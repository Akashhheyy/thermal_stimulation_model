"""Trained-model loading and prediction for the recommendation layer.

The loaded artifacts are the SAME fitted pipelines produced by
``scripts/train_ml_models.py``. Each pipeline carries its own preprocessing,
so prediction applies the exact training-time feature ordering and encodings.

This module never retrains, fabricates features, or invents weather values.

Feature construction reuses ``ml.features.build_feature_row`` and the design
space definitions from ``shelter.ml_dataset`` so the recommendation layer
does not define a second shelter parameter system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

from ..ml import ML_TARGETS as ML_TARGETS_KNOWN
from ..ml.predict import predict_features as _predict_features
from ..ml.train import MODEL_FILE_PATTERN
from ..shelter.geometry import build_geometry
from ..shelter.ml_dataset import WALL_MATERIALS
from ..shelter.models import ShelterConfig
from .schemas import (
    DISPLAY_BOUNDS,
    NASA_PROVENANCE_STATEMENT,
    ModelArtifactInfo,
    PredictionOutcome,
)

__all__ = [
    "DEFAULT_MODELS_DIR",
    "DEFAULT_METRICS_REPORT",
    "WEATHER_FEATURE_KEYS",
    "VALID_WALL_MATERIALS",
    "VALID_WINDOW_ORIENTATIONS",
    "VALID_MASS_MATERIALS",
    "PredictorBundle",
    "load_predictors",
    "select_primary_models",
    "design_features_from_config",
    "weather_features_from_scenario",
    "predict_design",
    "predict_candidates",
]


# ---------------------------------------------------------------------------
# Paths and supported feature values
# ---------------------------------------------------------------------------

DEFAULT_MODELS_DIR = Path("models") / "regression"
DEFAULT_METRICS_REPORT = Path("reports") / "ml_metrics.json"
DEFAULT_TRAINING_METADATA = Path("reports") / "training_metadata.json"
DEFAULT_DATASET_PATH = Path("data") / "shelter_ml_dataset.csv"
DEFAULT_METADATA_PATH = Path("data") / "shelter_ml_dataset_metadata.json"

WEATHER_FEATURE_KEYS = (
    "mean_outdoor_temperature_c",
    "minimum_outdoor_temperature_c",
    "maximum_outdoor_temperature_c",
    "daily_solar_sum_wh_m2",
)

VALID_WALL_MATERIALS = set(WALL_MATERIALS)
VALID_WINDOW_ORIENTATIONS = {"north", "east", "south", "west"}
VALID_MASS_MATERIALS = {"stone", "water", "none"}

DEFAULT_MODEL_NAMES = (
    "linear_regression",
    "random_forest",
    "gradient_boosting",
)


# ---------------------------------------------------------------------------
# Predictor bundle
# ---------------------------------------------------------------------------

@dataclass
class PredictorBundle:
    """Loaded model artifacts plus per-target primary model selection."""

    models: dict[tuple[str, str], object]
    models_dir: Path
    targets: tuple[str, ...]
    model_names: tuple[str, ...]
    primary_models: dict[str, str]
    artifact_info: ModelArtifactInfo

    def predict(
        self,
        features: pd.DataFrame,
        target: str | None = None,
    ) -> pd.DataFrame:
        """Reuse the existing prediction function on loaded pipelines."""
        return _predict_features(self.models, features, target=target)


# ---------------------------------------------------------------------------
# Metadata and model selection
# ---------------------------------------------------------------------------

def _load_training_metadata(path: Path | str) -> dict[str, Any]:
    """Load training metadata if available."""
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def select_primary_models(
    metrics_report: Path | str = DEFAULT_METRICS_REPORT,
    loaded_models: Sequence[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Choose the best available model for each target by test RMSE.

    If the metrics report does not contain a model's RMSE, that model is
    ranked after models with known RMSE values.

    If no metric is available for any model of a target, a deterministic
    fallback order is used.
    """
    fallback_order = {
        name: index
        for index, name in enumerate(
            (
                "gradient_boosting",
                "random_forest",
                "linear_regression",
            )
        )
    }

    metrics_path = Path(metrics_report)

    if metrics_path.exists():
        try:
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            report = {}
    else:
        report = {}

    available = tuple(loaded_models or ())
    selection: dict[str, str] = {}

    for target in ML_TARGETS_KNOWN:
        choices = [
            model_name
            for model_target, model_name in available
            if model_target == target
        ]

        if not choices:
            continue

        per_model = report.get(target, {})

        def sort_key(model_name: str) -> tuple[bool, float, int]:
            model_metrics = per_model.get(model_name, {})
            metrics = model_metrics.get("metrics", {})

            try:
                rmse = float(metrics["rmse"])
                return (False, rmse, fallback_order.get(model_name, 999))
            except (KeyError, TypeError, ValueError):
                return (True, float("inf"), fallback_order.get(model_name, 999))

        selection[target] = min(choices, key=sort_key)

    return selection


def load_predictors(
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    targets: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    primary_only: bool = False,
    metrics_report: Path | str = DEFAULT_METRICS_REPORT,
) -> PredictorBundle:
    """Load trained model artifacts.

    Parameters
    ----------
    models_dir:
        Directory containing ``*.joblib`` model artifacts.

    targets:
        Optional subset of supported ML targets.

    model_names:
        Optional subset of model types.

    primary_only:
        If True, load only the selected primary model for each target.

    metrics_report:
        Training metrics used to select the primary model.
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(
            f"trained models directory not found: {models_dir}"
        )

    used_targets = (
        tuple(targets)
        if targets is not None
        else tuple(ML_TARGETS_KNOWN)
    )

    unknown_targets = [
        target
        for target in used_targets
        if target not in ML_TARGETS_KNOWN
    ]

    if unknown_targets:
        raise ValueError(
            f"unsupported prediction targets: {unknown_targets}; "
            f"supported: {list(ML_TARGETS_KNOWN)}"
        )

    used_model_names = (
        tuple(model_names)
        if model_names is not None
        else DEFAULT_MODEL_NAMES
    )

    loaded: dict[tuple[str, str], object] = {}

    for target in used_targets:
        for model_name in used_model_names:
            path = models_dir / MODEL_FILE_PATTERN.format(
                target=target,
                model=model_name,
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"trained artifact not found: {path}. "
                    "Train models with scripts/train_ml_models.py "
                    "before predicting."
                )

            loaded[(target, model_name)] = joblib.load(path)

    if not loaded:
        raise ValueError(
            f"no trained model artifacts loaded from {models_dir}"
        )

    selection = select_primary_models(
        metrics_report=metrics_report,
        loaded_models=tuple(loaded),
    )

    if primary_only:
        primary_loaded: dict[tuple[str, str], object] = {}

        for target in used_targets:
            chosen_model = selection.get(target)

            if chosen_model is None:
                continue

            key = (target, chosen_model)

            if key not in loaded:
                continue

            primary_loaded[key] = loaded[key]

        if not primary_loaded:
            raise ValueError(
                "no primary trained models could be selected"
            )

        loaded = primary_loaded

    loaded_targets = tuple(
        sorted({target for target, _ in loaded})
    )

    loaded_model_names = tuple(
        sorted({model for _, model in loaded})
    )

    artifact_info = ModelArtifactInfo(
        models_dir=str(models_dir),
        file_count=len(loaded),
        targets=loaded_targets,
        model_names=loaded_model_names,
        training_metadata=_load_training_metadata(
            DEFAULT_TRAINING_METADATA
        ),
        primary_models=selection,
    )

    return PredictorBundle(
        models=loaded,
        models_dir=models_dir,
        targets=loaded_targets,
        model_names=loaded_model_names,
        primary_models=selection,
        artifact_info=artifact_info,
    )


# ---------------------------------------------------------------------------
# Shelter design → ML feature mapping
# ---------------------------------------------------------------------------

def _insulation_thickness(assembly: Any) -> float:
    """Sum insulation-layer thicknesses in an envelope assembly."""
    return sum(
        layer.thickness_m
        for layer in assembly.layers
        if layer.material_name == "insulation"
    )


def _structural_material(assembly: Any) -> str:
    """Return the structural material from an envelope assembly."""
    structural = [
        layer
        for layer in assembly.layers
        if layer.material_name != "insulation"
    ]

    if not structural:
        raise ValueError(
            f"envelope assembly {assembly.name!r} has no "
            "structural (non-insulation) layer"
        )

    return structural[-1].material_name


def design_features_from_config(
    config: ShelterConfig,
) -> dict[str, Any]:
    """Map a validated ShelterConfig onto the ML feature dictionary."""
    built = build_geometry(
        config.geometry,
        config.openings,
    )

    wall_material = _structural_material(
        config.wall_assembly
    )

    if wall_material not in VALID_WALL_MATERIALS:
        raise ValueError(
            f"wall_material {wall_material!r} is not in the "
            f"supported design space {sorted(VALID_WALL_MATERIALS)}"
        )

    window_orientation = (
        str(config.openings.window_wall_orientation)
        .strip()
        .lower()
    )

    if window_orientation not in VALID_WINDOW_ORIENTATIONS:
        raise ValueError(
            f"window_wall_orientation {window_orientation!r} "
            f"must be one of {sorted(VALID_WINDOW_ORIENTATIONS)}"
        )

    mass_material = (
        str(config.thermal_mass.material_name)
        .strip()
        .lower()
        if config.thermal_mass is not None
        else "none"
    )

    if mass_material not in VALID_MASS_MATERIALS:
        raise ValueError(
            f"thermal_mass_material {mass_material!r} "
            f"must be one of {sorted(VALID_MASS_MATERIALS)}"
        )

    mass_heat_capacity_j_k = (
        float(config.thermal_mass.heat_capacity_j_k)
        if config.thermal_mass is not None
        else 0.0
    )

    return {
        "wall_material": wall_material,
        "window_wall_orientation": window_orientation,
        "thermal_mass_material": mass_material,
        "length_m": float(config.geometry.length_m),
        "width_m": float(config.geometry.width_m),
        "height_m": float(config.geometry.height_m),
        "floor_area_m2": float(built.floor_area_m2),
        "volume_m3": float(built.volume_m3),
        "wall_insulation_thickness_m": _insulation_thickness(
            config.wall_assembly
        ),
        "roof_insulation_thickness_m": _insulation_thickness(
            config.roof_assembly
        ),
        "floor_insulation_thickness_m": _insulation_thickness(
            config.floor_assembly
        ),
        "window_area_m2": float(
            config.openings.window_area_m2
        ),
        "window_u_value_w_m2k": float(
            config.openings.window_u_value_w_m2k
        ),
        "door_area_m2": float(
            config.openings.door_area_m2
        ),
        "window_solar_heat_gain_coefficient": float(
            config.openings.window_solar_heat_gain_coefficient
        ),
        "thermal_mass_heat_capacity_j_k": mass_heat_capacity_j_k,
        "net_wall_area_m2": float(
            built.net_wall_area_m2
        ),
        "orientation_deg": float(
            config.geometry.orientation_deg
        ),
    }


# ---------------------------------------------------------------------------
# Prediction output formatting
# ---------------------------------------------------------------------------

def _apply_display_bounds(
    value: float,
    target: str,
) -> tuple[float, bool]:
    """Apply presentation bounds and report whether raw value is valid."""
    lo, hi = DISPLAY_BOUNDS.get(target, (None, None))

    in_bounds = True
    bounded = value

    if lo is not None and value < lo:
        bounded = lo
        in_bounds = False
    elif hi is not None and value > hi:
        bounded = hi
        in_bounds = False

    return bounded, in_bounds


def _build_outcome(
    *,
    design_id: str | None,
    weather_scenario_id: str | None,
    input_mode: str,
    prediction: pd.DataFrame,
    bundle: PredictorBundle,
) -> PredictionOutcome:
    """Translate model predictions into a PredictionOutcome."""
    raw: dict[str, dict[str, float]] = {}
    primary: dict[str, dict[str, Any]] = {}
    out_of_bounds: dict[str, list[str]] = {}

    model_names = sorted(
        {
            model
            for _, model in bundle.models
        }
    )

    for row in prediction.itertuples():
        target = str(row.target)

        per_model: dict[str, float] = {}

        for model in model_names:
            value = getattr(row, model, None)

            if value is None:
                continue

            per_model[model] = float(value)

        raw[target] = per_model

        chosen_model = bundle.primary_models.get(target)

        chosen_value = (
            per_model.get(chosen_model)
            if chosen_model is not None
            else None
        )

        if chosen_value is None and per_model:
            chosen_model, chosen_value = next(
                iter(per_model.items())
            )

        if chosen_value is None:
            chosen_value = float("nan")

        display_value, in_bounds = _apply_display_bounds(
            chosen_value,
            target,
        )

        primary[target] = {
            "value": chosen_value,
            "model": chosen_model,
            "in_bounds": in_bounds,
            "display_value": display_value,
        }

        flagged = [
            model
            for model, value in per_model.items()
            if not _apply_display_bounds(
                value,
                target,
            )[1]
        ]

        if flagged:
            out_of_bounds[target] = flagged

    return PredictionOutcome(
        design_id=design_id,
        weather_scenario_id=weather_scenario_id,
        input_mode=input_mode,
        raw_predictions=raw,
        primary_predictions=primary,
        out_of_bounds=out_of_bounds,
        provenance=NASA_PROVENANCE_STATEMENT,
        artifact_info=bundle.artifact_info,
    )


# ---------------------------------------------------------------------------
# Single-design prediction
# ---------------------------------------------------------------------------

def predict_design(
    bundle: PredictorBundle,
    *,
    config: ShelterConfig | None = None,
    design_id: str | None = None,
    scenario_id: str | None = None,
    config_features: dict[str, Any] | None = None,
    weather_features: dict[str, Any] | None = None,
    target: str | None = None,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> PredictionOutcome:
    """Predict physical targets for one design and weather scenario.

    Supported input modes:

    - ``config``: validated ShelterConfig + scenario metadata
    - ``design_id``: existing dataset row + scenario
    - ``config_features`` + ``weather_features``: explicit features

    Exactly one mode must be supplied.
    """
    from ..ml.features import build_feature_row
    from ..ml.predict import load_dataset_row_features

    explicit_features = (
        config_features is not None
        or weather_features is not None
    )

    modes = sum(
        (
            config is not None,
            design_id is not None,
            explicit_features,
        )
    )

    if modes != 1:
        raise ValueError(
            "exactly one input mode must be given: "
            "config, design_id, or "
            "config_features+weather_features"
        )

    if config is not None:
        if scenario_id is None:
            raise ValueError(
                "scenario_id is required when predicting "
                "from a ShelterConfig"
            )

        cfg_features = design_features_from_config(config)

        wx_features = weather_features_from_scenario(
            scenario_id,
            metadata_path=metadata_path,
        )

        features = build_feature_row(
            cfg_features,
            wx_features,
        )

        outcome_id = config.name
        outcome_scenario = scenario_id
        mode = "config"

    elif design_id is not None:
        if scenario_id is None:
            raise ValueError(
                "scenario_id is required when predicting "
                "a dataset design_id"
            )

        features = load_dataset_row_features(
            design_id,
            scenario_id,
            dataset_path,
        )

        outcome_id = design_id
        outcome_scenario = scenario_id
        mode = "dataset_row"

    else:
        if config_features is None or weather_features is None:
            raise ValueError(
                "both config_features and weather_features "
                "are required"
            )

        features = build_feature_row(
            config_features,
            weather_features,
        )

        outcome_id = None
        outcome_scenario = None
        mode = "explicit_features"

    prediction = bundle.predict(
        features,
        target=target,
    )

    return _build_outcome(
        design_id=outcome_id,
        weather_scenario_id=outcome_scenario,
        input_mode=mode,
        prediction=prediction,
        bundle=bundle,
    )


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_candidates(
    bundle: PredictorBundle,
    configs: Sequence[ShelterConfig],
    scenario_id: str,
    *,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> list[PredictionOutcome]:
    """Predict every candidate design under one weather scenario."""
    return [
        predict_design(
            bundle,
            config=config,
            scenario_id=scenario_id,
            metadata_path=metadata_path,
        )
        for config in configs
    ]


# ---------------------------------------------------------------------------
# Weather scenario feature extraction
# ---------------------------------------------------------------------------

def weather_features_from_scenario(
    scenario_id: str,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Read training-time weather features for a scenario from metadata.

    The metadata contains the exact scenario summaries used during training,
    avoiding any new weather fetch or fabricated weather values.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"dataset metadata not found: {metadata_path}"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:
        raise ValueError(
            f"unable to read dataset metadata: {metadata_path}"
        ) from exc

    scenarios = metadata.get(
        "weather_scenarios",
        {},
    ).get(
        "used",
        [],
    )

    for scenario in scenarios:
        if scenario.get("scenario_id") == scenario_id:
            missing = [
                key
                for key in WEATHER_FEATURE_KEYS
                if key not in scenario
            ]

            if missing:
                raise ValueError(
                    f"weather scenario {scenario_id!r} "
                    f"is missing required features: {missing}"
                )

            return {
                key: float(scenario[key])
                for key in WEATHER_FEATURE_KEYS
            }

    available = [
        scenario.get("scenario_id")
        for scenario in scenarios
    ]

    raise ValueError(
        f"weather scenario {scenario_id!r} not found in metadata; "
        f"available: {available}"
    )