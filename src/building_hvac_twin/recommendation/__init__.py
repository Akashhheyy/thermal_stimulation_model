"""ML prediction and design recommendation layer.

An application layer on top of the existing trained surrogate models.  It
predicts physical targets for shelter designs, ranks candidate designs by a
transparent weighted decision score, and cross-checks ML predictions against
the physics engine.  It never retrains, never invents targets, and never
presents model outputs as measurements.
"""
from .predictor import (
    DEFAULT_METRICS_REPORT,
    DEFAULT_MODELS_DIR,
    VALID_MASS_MATERIALS,
    VALID_WALL_MATERIALS,
    VALID_WINDOW_ORIENTATIONS,
    WEATHER_FEATURE_KEYS,
    PredictorBundle,
    design_features_from_config,
    load_predictors,
    predict_candidates,
    predict_design,
    select_primary_models,
    weather_features_from_scenario,
)
from .ranking import DEFAULT_OBJECTIVES, rank_designs, score_components
from .schemas import (
    DISPLAY_BOUNDS,
    NASA_PROVENANCE_STATEMENT,
    SURROGATE_MODEL_DISCLAIMER,
    PHYSICAL_TARGETS,
    FastObjective,
    ModelArtifactInfo,
    PredictionOutcome,
    RankedRecommendation,
    RecommendationObjective,
)
from .validation import (
    SCENARIO_WEATHER_CACHE_DIR,
    compare_prediction_with_physics,
    load_scenario_weather,
)

__all__ = [
    "NASA_PROVENANCE_STATEMENT",
    "SURROGATE_MODEL_DISCLAIMER",
    "DISPLAY_BOUNDS",
    "PHYSICAL_TARGETS",
    "FastObjective",
    "RecommendationObjective",
    "RankedRecommendation",
    "ModelArtifactInfo",
    "PredictionOutcome",
    "DEFAULT_OBJECTIVES",
    "DEFAULT_METRICS_REPORT",
    "DEFAULT_MODELS_DIR",
    "WEATHER_FEATURE_KEYS",
    "VALID_WALL_MATERIALS",
    "VALID_WINDOW_ORIENTATIONS",
    "VALID_MASS_MATERIALS",
    "SCENARIO_WEATHER_CACHE_DIR",
    "PredictorBundle",
    "load_predictors",
    "select_primary_models",
    "design_features_from_config",
    "weather_features_from_scenario",
    "predict_design",
    "predict_candidates",
    "rank_designs",
    "score_components",
    "load_scenario_weather",
    "compare_prediction_with_physics",
]