"""Trained-model loading and prediction for the recommendation layer.

The loaded artifacts are the SAME fitted pipelines produced by
``scripts/train_ml_models.py`` (each pipeline carries its own preprocessing),
so prediction applies the exact training-time feature ordering and encodings.
This module never retrains, never fabricates features, and never invents a
weather value.

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
from ..shelter.ml_dataset import (
    DEFAULT_SEED,
    WALL_MATERIALS,
    build_shelter_config,
)
from ..shelter.models import ShelterConfig
from .schemas import (
    DISPLAY_BOUNDS,
    NASA_PROVENANCE_STATEMENT,
    SURROGATE_MODEL_DISCLAIMER,
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

@dataclass
class PredictorBundle:
    """Loaded model artifacts plus the per-target primary model selection."""

    models: dict[tuple[str, str], object]
    models_dir: Path
    targets: tuple[str, ...]
    model_names: tuple[str, ...]
    primary_models: dict[str, str]
    artifact_info: ModelArtifactInfo

    def predict(self, features: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Reuse the existing prediction function on the loaded pipelines."""
        return _predict_features(self.models, features, target=target)


def _load_training_metadata(path: Path | str) -> dict[str, Any]:
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
    """Choose the best model per target by test RMSE, with fallbacks.

    The metrics report is the artifact produced by the existing training run.
    When it is missing (for example in offline tests reading a minimal model
    set), each target falls back to the first loaded model in a documented
    order so prediction still works without inventing anything.
    """
    fallback_order = ("gradient_boosting", "random_forest", "linear_regression")
    report = json.loads(Path(metrics_report).read_text(encoding="utf-8")) if Path(metrics_report).exists() else {}
    available = set(loaded_models or ())
    selection: dict[str, str] = {}
    for target in ML_TARGETS_KNOWN:
        if available and not any(model_target == target for model_target, _ in available):
            continue
        choices = [name for model_target, name in (loaded_models or ()) if model_target == target]
        if not choices:
            continue
        per_model = report.get(target, {})
        ranked = sorted(
            choices,
            key=lambda name: (
                float(per_model[name]["metrics"]["rmse"]) if name in per_model else float("inf"),
                name,
            ),
        )
        chosen = ranked[0]
        for candidate in fallback_order:
            if candidate in choices:
                chosen = candidate
                break
        selection[target] = chosen
    return selection


def load_predictors(
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    targets: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    primary_only: bool = False,
    metrics_report: Path | str = DEFAULT_METRICS_REPORT,
) -> PredictorBundle:
    """Load trained artifacts; ``primary_only`` loads one model per target.

    ``targets`` and ``model_names`` restrict the loaded set (used by offline
    tests and fast CLI runs); the default loads every trained artifact for
    every supported physical target.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"trained models directory not found: {models_dir}")
def _insulation_thickness(assembly: "EnvelopeAssembly") -> float:
    """Sum the thickness of insulation layers in an envelope assembly.

    Insulation layers are identified by name; the structural layer is the
    remaining one.  This mirrors how the dataset generator builds assemblies.
    """
    return sum(layer.thickness_m for layer in assembly.layers if layer.material_name == "insulation")


def _structural_material(assembly: "EnvelopeAssembly") -> str:
    """Return the structural (non-insulation) material name of an assembly.

    Every assembly in the design space has exactly one structural layer.
    """
    structural = [layer for layer in assembly.layers if layer.material_name != "insulation"]
    if not structural:
        raise ValueError(
            f"envelope assembly {assembly.name!r} has no structural (non-insulation) layer"
        )
    return structural[-1].material_name


def design_features_from_config(config: ShelterConfig) -> dict[str, Any]:
    """Map a validated ``ShelterConfig`` onto the ML feature dict.

    The mapping is the inverse of the dataset generator's feature extraction:
    geometry is derived with ``build_geometry`` (so ``floor_area_m2``,
    ``volume_m3`` and ``net_wall_area_m2`` match the training rows), and the
    envelope assemblies are decomposed into the structural material and
    insulation thickness the models were trained on.

    Categorical values are validated against the documented design space so
    an unsupported material or orientation raises instead of being silently
    encoded as an unknown category.
    """
    built = build_geometry(config.geometry, config.openings)

    wall_material = _structural_material(config.wall_assembly)
    if wall_material not in VALID_WALL_MATERIALS:
        raise ValueError(
            f"wall_material {wall_material!r} is not in the supported design space "
            f"{sorted(VALID_WALL_MATERIALS)}"
        )
    window_orientation = str(config.openings.window_wall_orientation).strip().lower()
    if window_orientation not in VALID_WINDOW_ORIENTATIONS:
        raise ValueError(
            f"window_wall_orientation {window_orientation!r} must be one of "
            f"{sorted(VALID_WINDOW_ORIENTATIONS)}"
        )
    mass_material = (
        str(config.thermal_mass.material_name).strip().lower()
        if config.thermal_mass is not None
        else "none"
    )
    if mass_material not in VALID_MASS_MATERIALS:
        raise ValueError(
            f"thermal_mass_material {mass_material!r} must be one of "
            f"{sorted(VALID_MASS_MATERIALS)}"
        )

    mass_heat_capacity_j_k = (
        float(config.thermal_mass.heat_capacity_j_k) if config.thermal_mass is not None else 0.0
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
        "wall_insulation_thickness_m": _insulation_thickness(config.wall_assembly),
        "roof_insulation_thickness_m": _insulation_thickness(config.roof_assembly),
        "floor_insulation_thickness_m": _insulation_thickness(config.floor_assembly),
        "window_area_m2": float(config.openings.window_area_m2),
        "window_u_value_w_m2k": float(config.openings.window_u_value_w_m2k),
        "door_area_m2": float(config.openings.door_area_m2),
        "window_solar_heat_gain_coefficient": float(
            config.openings.window_solar_heat_gain_coefficient
        ),
        "thermal_mass_heat_capacity_j_k": mass_heat_capacity_j_k,
        "net_wall_area_m2": float(built.net_wall_area_m2),
        "orientation_deg": float(config.geometry.orientation_deg),
    }
def _apply_display_bounds(value: float, target: str) -> tuple[float, bool]:
    """Return a bounded presentation value and whether the raw value is in bounds."""
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
    """Translate a target x model prediction frame into a ``PredictionOutcome``."""
    raw: dict[str, dict[str, float]] = {}
    primary: dict[str, dict[str, Any]] = {}
    out_of_bounds: dict[str, list[str]] = {}
    model_names = sorted({model for (_, model) in bundle.models})
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
        chosen_value = per_model.get(chosen_model) if chosen_model else None
        if chosen_value is None and per_model:
            chosen_model, chosen_value = next(iter(per_model.items()))
        if chosen_value is None:
            chosen_value = float("nan")
        display_value, in_bounds = _apply_display_bounds(chosen_value, target)
        primary[target] = {
            "value": chosen_value,
            "model": chosen_model,
            "in_bounds": in_bounds,
            "display_value": display_value,
        }
        flagged = [model for model, value in per_model.items() if not _apply_display_bounds(value, target)[1]]
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
    """Predict physical targets for one design under one weather scenario.

    Three input modes are supported and documented on the outcome:

    - ``config``: a validated ``ShelterConfig`` is mapped onto features and the
      scenario's weather is read from dataset metadata.
    - ``design_id``: an existing dataset row is reused verbatim (reproducible).
    - ``config_features`` + ``weather_features``: explicit feature dicts.

    Exactly one mode must be given.
    """
    from ..ml.features import build_feature_row
    from ..ml.predict import load_dataset_row_features

    modes = sum(bool(x) for x in (config, design_id, (config_features or weather_features)))
    if modes != 1:
        raise ValueError(
            "exactly one input mode must be given: config, design_id, or "
            "config_features+weather_features"
        )

    if config is not None:
        if scenario_id is None:
            raise ValueError("scenario_id is required when predicting from a ShelterConfig")
        cfg_features = design_features_from_config(config)
        wx_features = weather_features_from_scenario(scenario_id, metadata_path=metadata_path)
        features = build_feature_row(cfg_features, wx_features)
        outcome_id = config.name
        outcome_scenario = scenario_id
        mode = "config"
    elif design_id is not None:
        if scenario_id is None:
            raise ValueError("scenario_id is required when predicting a dataset design_id")
        features = load_dataset_row_features(design_id, scenario_id, dataset_path)
        outcome_id = design_id
        outcome_scenario = scenario_id
        mode = "dataset_row"
    else:
        if not (config_features and weather_features):
            raise ValueError("both config_features and weather_features are required")
        features = build_feature_row(config_features, weather_features)
        outcome_id = None
        outcome_scenario = None
        mode = "explicit_features"

    prediction = bundle.predict(features, target=target)
    return _build_outcome(
        design_id=outcome_id,
        weather_scenario_id=outcome_scenario,
        input_mode=mode,
        prediction=prediction,
        bundle=bundle,
    )


def predict_candidates(
    bundle: PredictorBundle,
    configs: Sequence[ShelterConfig],
    scenario_id: str,
    *,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> list[PredictionOutcome]:
    """Predict every candidate design under one weather scenario.

    Each config is mapped independently so the design space mapping is the
    same one used for a single prediction.
    """
    outcomes: list[PredictionOutcome] = []
    for config in configs:
        outcomes.append(
            predict_design(
                bundle,
                config=config,
                scenario_id=scenario_id,
                metadata_path=metadata_path,
            )
        )
    return outcomes


def weather_features_from_scenario(
    scenario_id: str,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Read the documented weather features for one scenario from metadata.

    The dataset metadata records the exact scenario summaries (mean/min/max
    temperature and daily solar sum) that were used at training time, so the
    recommendation layer feeds the models the same weather representation
    without re-fetching or inventing any value.
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    scenarios = metadata.get("weather_scenarios", {}).get("used", [])
    for scenario in scenarios:
        if scenario.get("scenario_id") == scenario_id:
            return {key: float(scenario[key]) for key in WEATHER_FEATURE_KEYS}
    available = [scenario.get("scenario_id") for scenario in scenarios]
    raise ValueError(
        f"weather scenario {scenario_id!r} not found in metadata; available: {available}"
    )

    used_targets = tuple(targets) if targets is not None else tuple(ML_TARGETS_KNOWN)
    unknown_targets = [target for target in used_targets if target not in ML_TARGETS_KNOWN]
    if unknown_targets:
        raise ValueError(
            f"unsupported prediction targets: {unknown_targets}; supported: {list(ML_TARGETS_KNOWN)}"
        )
    used_model_names = tuple(model_names) if model_names is not None else (
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    )

    loaded: dict[tuple[str, str], object] = {}
    for target in used_targets:
        for model in used_model_names:
            path = models_dir / MODEL_FILE_PATTERN.format(target=target, model=model)
            if not path.exists():
                raise FileNotFoundError(
                    f"trained artifact not found: {path}.  Train models with "
                    f"scripts/train_ml_models.py before predicting."
                )
            loaded[(target, model)] = joblib.load(path)

    selection = select_primary_models(metrics_report, loaded_models=list(loaded))

    if primary_only:
        primary_only_loaded: dict[tuple[str, str], object] = {}
        for target in used_targets:
            chosen = selection.get(target)
            if chosen is None:
                continue
            path = models_dir / MODEL_FILE_PATTERN.format(target=target, model=chosen)
            primary_only_loaded[(target, chosen)] = loaded[(target, chosen)]
        loaded = primary_only_loaded

    info = ModelArtifactInfo(
        models_dir=str(models_dir),
        file_count=len(loaded),
        targets=used_targets,
        model_names=used_model_names,
        training_metadata=_load_training_metadata(DEFAULT_TRAINING_METADATA),
        primary_models={target: model for (target, model) in loaded},
    )
    return PredictorBundle(
        models=loaded,
        models_dir=models_dir,
        targets=used_targets,
        model_names=used_model_names,
        primary_models=selection,
        artifact_info=info,
    )
VALID_WALL_MATERIALS = set(WALL_MATERIALS)
VALID_WINDOW_ORIENTATIONS = {"north", "east", "south", "west"}
VALID_MASS_MATERIALS = {"stone", "water", "none"}