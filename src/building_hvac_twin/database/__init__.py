"""MongoDB persistence layer for application data only.

Collections and their purpose:

- ``designs``            the existing 300 shelter designs from the ML dataset
- ``weather_scenarios``  the existing 10 NASA POWER scenario metadata records
- ``predictions``        application results of POST /predict
- ``recommendations``    application results of POST /recommend
- ``comparisons``        application results of POST /compare

The ML dataset CSV, trained model artifacts and NASA POWER raw weather files
stay in their existing project locations; MongoDB never stores them and never
runs ML or physics.
"""
from .collections import (
    COLLECTION_NAMES,
    COMPARISONS_COLLECTION,
    DESIGNS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RECOMMENDATIONS_COLLECTION,
    WEATHER_SCENARIOS_COLLECTION,
    comparison_document,
    design_document,
    prediction_document,
    recommendation_document,
    scenario_document,
)
from .connection import MongoSettings, connect, read_env_file, settings_from_env
from .repositories import (
    ComparisonRepository,
    DesignRepository,
    PersistenceResult,
    PredictionRepository,
    RecommendationRepository,
    RepositoryBundle,
    ScenarioRepository,
    build_repositories,
)

__all__ = [
    "DESIGNS_COLLECTION",
    "WEATHER_SCENARIOS_COLLECTION",
    "PREDICTIONS_COLLECTION",
    "RECOMMENDATIONS_COLLECTION",
    "COMPARISONS_COLLECTION",
    "COLLECTION_NAMES",
    "design_document",
    "scenario_document",
    "prediction_document",
    "recommendation_document",
    "comparison_document",
    "MongoSettings",
    "settings_from_env",
    "connect",
    "read_env_file",
    "DesignRepository",
    "ScenarioRepository",
    "PredictionRepository",
    "RecommendationRepository",
    "ComparisonRepository",
    "RepositoryBundle",
    "PersistenceResult",
    "build_repositories",
]
