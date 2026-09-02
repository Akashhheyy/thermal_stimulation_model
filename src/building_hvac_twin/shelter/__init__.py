"""Passive shelter thermal design and prediction model.

This package is a shelter-specific thermal modelling layer built beside the
existing Vicena building-energy reference (which remains available as a
baseline).  It focuses on area-specific passive shelter thermal design:

- climate (outdoor weather)
- geometry
- orientation
- materials
- insulation
- openings
- solar gain
- thermal mass
- indoor temperature
- thermal comfort
- minimum auxiliary energy

The first version is a research/prototype model, not a certified engineering
design tool.  No machine-learning, frontend, API, or application database is
part of this package yet.
"""
from .models import (
    ComfortRange,
    DataCategory,
    EnvelopeAssembly,
    InternalHeatSources,
    Layer,
    Location,
    Material,
    Openings,
    ShelterConfig,
    ShelterGeometry,
    SimulationResult,
    ThermalMass,
    WeatherRecord,
)
from .materials import REFERENCE_MATERIALS, MaterialLibrary, material_library
from .geometry import (
    CARDINAL_DIRECTIONS,
    RectangularGeometry,
    build_geometry,
    cardinal_direction,
    derived_areas,
)
from .validation import validate_shelter_config
from .thermal_mass import (
    ThermalMassState,
    effective_capacitance_j_k,
    heat_absorbed_j,
    heat_flow_w,
    heat_released_j,
    stored_energy_j,
    temperature_change_c,
)
from .simulation import OUTPUT_COLUMNS as SIMULATION_OUTPUT_COLUMNS, simulate_shelter, weather_frame
from .comfort import ComfortSummary, comfort_metrics, comfort_summary
from .comparison import (
    ComparisonReport,
    PerformanceWeights,
    compare_designs,
    design_metrics,
    performance_scores,
    score_components,
    simulate_designs,
)
from .weather import (
    DEFAULT_NASA_PARAMETERS,
    NasaWeatherError,
    PARAMETER_MAP,
    clear_weather_cache,
    fetch_nasa_power_hourly,
    get_nasa_weather_data,
    parse_nasa_power_hourly,
    weather_cache_info,
)
from .ml_dataset import (
    DATASET_COLUMNS as ML_DATASET_COLUMNS,
    DEFAULT_DESIGN_COUNT,
    DEFAULT_SCENARIOS,
    DEFAULT_SEED,
    GENERATOR_VERSION,
    TARGET_COLUMNS as ML_TARGET_COLUMNS,
    DatasetResult,
    WeatherScenario,
    build_dataset_row,
    build_shelter_config,
    fetch_or_load_weather,
    generate_designs,
    generate_ml_dataset,
    get_weather_scenarios,
    simulate_design_scenario,
    validate_dataset as validate_ml_dataset,
    write_dataset as write_ml_dataset,
    write_metadata as write_ml_dataset_metadata,
)

__all__ = [
    "ComfortRange",
    "DataCategory",
    "EnvelopeAssembly",
    "InternalHeatSources",
    "Layer",
    "Location",
    "Material",
    "MaterialLibrary",
    "Openings",
    "REFERENCE_MATERIALS",
    "ShelterConfig",
    "ShelterGeometry",
    "SimulationResult",
    "ThermalMass",
    "WeatherRecord",
    "CARDINAL_DIRECTIONS",
    "RectangularGeometry",
    "build_geometry",
    "cardinal_direction",
    "derived_areas",
    "validate_shelter_config",
    "ThermalMassState",
    "effective_capacitance_j_k",
    "heat_absorbed_j",
    "heat_flow_w",
    "heat_released_j",
    "stored_energy_j",
    "temperature_change_c",
    "SIMULATION_OUTPUT_COLUMNS",
    "simulate_shelter",
    "weather_frame",
    "ComfortSummary",
    "comfort_metrics",
    "comfort_summary",
    "ComparisonReport",
    "PerformanceWeights",
    "compare_designs",
    "design_metrics",
    "performance_scores",
    "score_components",
    "simulate_designs",
    "NasaWeatherError",
    "fetch_nasa_power_hourly",
    "get_nasa_weather_data",
    "parse_nasa_power_hourly",
    "clear_weather_cache",
    "weather_cache_info",
    "DEFAULT_NASA_PARAMETERS",
    "PARAMETER_MAP",
    "ML_DATASET_COLUMNS",
    "ML_TARGET_COLUMNS",
    "DEFAULT_DESIGN_COUNT",
    "DEFAULT_SCENARIOS",
    "DEFAULT_SEED",
    "GENERATOR_VERSION",
    "DatasetResult",
    "WeatherScenario",
    "build_dataset_row",
    "build_shelter_config",
    "fetch_or_load_weather",
    "generate_designs",
    "generate_ml_dataset",
    "get_weather_scenarios",
    "simulate_design_scenario",
    "validate_ml_dataset",
    "write_ml_dataset",
    "write_ml_dataset_metadata",
]

