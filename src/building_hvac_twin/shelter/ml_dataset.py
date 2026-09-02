"""Reproducible ML dataset generation for the passive shelter thermal model.

This module combines existing components without duplicating any of them:

1. the existing NASA POWER integration (``shelter.weather``) for real hourly
   weather, used through its public functions ``fetch_nasa_power_hourly`` and
   ``parse_nasa_power_hourly`` (no second NASA client exists anywhere);
2. the existing configurable shelter design space (``ShelterConfig`` plus the
   reference material library) sampled deterministically with a fixed seed;
3. the existing thermal engine (``simulate_shelter``) and metric layer
   (``design_metrics``).

Provenance rules honoured here:

- NASA POWER values are satellite/reanalysis-derived estimates and are not
  ground measurements; every row and the metadata carry that label.
- No synthetic weather enters this dataset.  Offline tests use an injected
  fake transport and are clearly separated from live retrieval.
- The thermal engine is passive-only, so no auxiliary heating or cooling
  energy is produced or invented here.
- ``performance_score`` from ``comparison.py`` is a relative within-batch
  ranking score and is intentionally excluded from the dataset; the ML targets
  are the raw physical metrics only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .comparison import design_metrics
from .geometry import CARDINAL_DIRECTIONS, build_geometry
from .materials import REFERENCE_MATERIALS
from .models import (
    ComfortRange,
    DataCategory,
    EnvelopeAssembly,
    InternalHeatSources,
    Layer,
    Openings,
    ShelterConfig,
    ShelterGeometry,
    ThermalMass,
)
from .simulation import simulate_shelter
from .validation import validate_shelter_config
from .weather import (
    DEFAULT_NASA_PARAMETERS,
    NASA_BASE_URL,
    NasaWeatherError,
    fetch_nasa_power_hourly,
    parse_nasa_power_hourly,
)

__all__ = [
    "GENERATOR_VERSION",
    "DEFAULT_DESIGN_COUNT",
    "DEFAULT_SEED",
    "DEFAULT_LATITUDE",
    "DEFAULT_LONGITUDE",
    "DEFAULT_LOCATION_NAME",
    "WeatherScenario",
    "DEFAULT_SCENARIOS",
    "DATASET_COLUMNS",
    "DESIGN_PARAMETER_COLUMNS",
    "WEATHER_COLUMNS",
    "TARGET_COLUMNS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "DatasetResult",
    "generate_designs",
    "build_shelter_config",
    "fetch_or_load_weather",
    "get_weather_scenarios",
    "simulate_design_scenario",
    "build_dataset_row",
    "generate_ml_dataset",
    "write_dataset",
    "write_metadata",
    "validate_dataset",
]

GENERATOR_VERSION = "1.0.0"
DEFAULT_DESIGN_COUNT = 300
DEFAULT_SEED = 42
DEFAULT_LATITUDE = 34.1645
DEFAULT_LONGITUDE = 77.5789
DEFAULT_LOCATION_NAME = "Leh, Ladakh, India"

# A 24-hour scenario must yield at least this many usable hourly records after
# NASA missing-value handling; otherwise the scenario date is rejected and the
# documented fallback date is tried instead.  Nothing is ever invented to fill
# the gap.
MIN_HOURLY_RECORDS = 18

NASA_PROVENANCE_STATEMENT = (
    "NASA POWER data are satellite/reanalysis-derived estimates and are not "
    "ground measurements."
)
LIVE_NASA_STATEMENT = (
    "Final dataset weather values were retrieved from live NASA POWER requests."
)

@dataclass(frozen=True)
class WeatherScenario:
    """One deterministic 24-hour weather scenario request.

    ``weather_date`` is always tried first.  ``fallback_date`` (one week
    later) is used only when the requested date returns unusable NASA data;
    any replacement is recorded in the metadata and never hidden.
    """

    scenario_id: str
    name: str
    season: str
    weather_date: str
    fallback_date: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# Ten documented, fixed 2024 dates at Leh covering the seasonal cycle.  These
# are never chosen randomly at runtime.
DEFAULT_SCENARIOS: tuple[WeatherScenario, ...] = (
    WeatherScenario("S01_winter", "winter", "winter", "2024-01-15", "2024-01-22"),
    WeatherScenario("S02_late_winter", "late winter", "late winter / early spring", "2024-02-20", "2024-02-27"),
    WeatherScenario("S03_spring", "spring", "spring", "2024-04-05", "2024-04-12"),
    WeatherScenario("S04_late_spring", "late spring", "late spring", "2024-05-10", "2024-05-17"),
    WeatherScenario("S05_early_summer", "early summer", "early summer", "2024-06-05", "2024-06-12"),
    WeatherScenario("S06_peak_summer", "peak summer", "peak summer", "2024-06-25", "2024-07-02"),
    WeatherScenario("S07_monsoon_period", "monsoon period", "monsoon-period / summer", "2024-07-25", "2024-08-01"),
    WeatherScenario("S08_late_summer", "late summer", "late summer", "2024-08-20", "2024-08-27"),
    WeatherScenario("S09_autumn", "autumn", "autumn", "2024-10-05", "2024-10-12"),
    WeatherScenario("S10_early_winter", "early winter", "early winter", "2024-11-20", "2024-11-27"),
)


def get_weather_scenarios() -> list[WeatherScenario]:
    """Return the documented default scenario list (deterministic order)."""
    return list(DEFAULT_SCENARIOS)


# Column order is fixed so the CSV is stable across runs.
DESIGN_PARAMETER_COLUMNS = (
    "design_id",
    "wall_material",
    "wall_structural_thickness_m",
    "wall_insulation_thickness_m",
    "roof_insulation_thickness_m",
    "floor_insulation_thickness_m",
    "length_m",
    "width_m",
    "height_m",
    "orientation_deg",
    "window_area_m2",
    "window_wall_orientation",
    "door_area_m2",
    "door_wall_orientation",
    "window_solar_heat_gain_coefficient",
    "window_u_value_w_m2k",
    "door_u_value_w_m2k",
    "has_thermal_mass",
    "thermal_mass_material",
    "thermal_mass_kg",
    "thermal_mass_specific_heat_j_kgk",
    "thermal_mass_heat_capacity_j_k",
    "occupant_count",
    "sensible_heat_per_person_w",
    "equipment_heat_w",
    "lighting_heat_w",
    "floor_area_m2",
    "volume_m3",
    "net_wall_area_m2",
)

WEATHER_COLUMNS = (
    "weather_scenario_id",
    "weather_name",
    "weather_season",
    "weather_date",
    "weather_effective_date",
    "latitude",
    "longitude",
    "weather_source",
    "weather_data_category",
    "weather_provenance",
    "weather_retrieval_status",
    "weather_record_count",
    "skipped_missing_required_records",
    "mean_outdoor_temperature_c",
    "minimum_outdoor_temperature_c",
    "maximum_outdoor_temperature_c",
    "daily_solar_sum_wh_m2",
    "mean_wind_speed_m_s",
    "mean_relative_humidity_percent",
)

TARGET_COLUMNS = (
    "percent_time_comfortable",
    "percent_time_below_comfort",
    "percent_time_above_comfort",
    "minimum_indoor_temperature_c",
    "maximum_indoor_temperature_c",
    "mean_indoor_temperature_c",
    "indoor_temperature_range_c",
    "degree_hours_below_comfort",
    "degree_hours_above_comfort",
    "total_heat_loss_kwh",
    "total_solar_gain_kwh",
    "thermal_mass_absorbed_kwh",
    "thermal_mass_released_kwh",
    "thermal_mass_net_kwh",
)

DATASET_COLUMNS = DESIGN_PARAMETER_COLUMNS + WEATHER_COLUMNS + TARGET_COLUMNS

# Wind and humidity are carried by NASA but unused by the current thermal
# engine; they may be NaN when NASA reports them missing, so they are the only
# optional columns.  Every other column must be finite.
OPTIONAL_COLUMNS = ("mean_wind_speed_m_s", "mean_relative_humidity_percent")
REQUIRED_COLUMNS = tuple(
    column for column in DATASET_COLUMNS if column not in OPTIONAL_COLUMNS
)

# Design-space constants.  Every structural thickness below is taken from the
# existing repository examples (0.35 m walls, 0.15 m concrete roof/floor);
# nothing here invents a new physical value.
WALL_MATERIALS = ("brick_masonry", "earth_adobe", "concrete", "timber", "stone")
WALL_STRUCTURAL_THICKNESS_M = 0.35
ROOF_STRUCTURAL_MATERIAL = "concrete"
ROOF_STRUCTURAL_THICKNESS_M = 0.15
FLOOR_STRUCTURAL_MATERIAL = "concrete"
FLOOR_STRUCTURAL_THICKNESS_M = 0.15
INSULATION_MATERIAL = "insulation"
INSULATION_THICKNESS_RANGE = (0.05, 0.20)
NO_INSULATION_PROBABILITY = 0.2
LENGTH_RANGE = (4.0, 6.0)
WIDTH_M = 4.0
HEIGHT_M = 3.0
SHELTER_ORIENTATIONS = (0.0, 90.0, 180.0, 270.0)
WINDOW_AREA_RANGE = (0.5, 4.0)
SHGC_RANGE = (0.4, 0.8)
DOOR_AREA_M2 = 2.0
MASS_RANGE = (500.0, 3000.0)
MASS_MATERIALS = ("stone", "water")
NO_MASS_PROBABILITY = 0.25
MASS_INITIAL_TEMPERATURE_C = 5.0
# Operational assumptions are held FIXED at the existing repository reference
# values; they are not design variables and are not varied.
INTERNAL_HEAT_ASSUMPTIONS = {
    "occupant_count": 2,
    "sensible_heat_per_person_w": 75.0,
    "equipment_heat_w": 50.0,
    "lighting_heat_w": 20.0,
}
INITIAL_INDOOR_TEMPERATURE_C = 10.0

Transport = Callable[[str, float], bytes]

WEATHER_SOURCE_LABEL = "NASA POWER hourly point API"
WEATHER_CATEGORY_LABEL = "nasa_power_satellite_reanalysis"


def _cache_file_name(
    latitude: float, longitude: float, nasa_date: str, community: str
) -> str:
    params = "-".join(DEFAULT_NASA_PARAMETERS)
    return (
        f"nasa_power_hourly_lat{latitude:.4f}_lon{longitude:.4f}"
        f"_{nasa_date}_{community}_{params}.json"
    )


def _payload_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _read_cache(path: Path) -> tuple[dict, str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    return body["payload"], str(body["transport_kind"])


def _write_cache(path: Path, payload: dict, transport_kind: str) -> None:
    body = {
        "request_source": WEATHER_SOURCE_LABEL,
        "endpoint": NASA_BASE_URL,
        "transport_kind": transport_kind,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": _payload_sha256(payload),
        "payload": payload,
    }
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def _validate_shape_date(value: str) -> str:
    """Normalise a YYYY-MM-DD string into the YYYYMMDD NASA date form."""
    stamp = datetime.strptime(str(value).strip(), "%Y-%m-%d")
    return stamp.strftime("%Y%m%d")


def summarize_weather(frame: pd.DataFrame) -> dict[str, Any]:
    """Scenario-level weather features from one hourly NASA frame.

    Wind and humidity means may be NaN when NASA reported them missing; they
    are the only optional dataset columns and the thermal engine does not use
    them.
    """
    wind = frame["wind_speed_m_s"] if "wind_speed_m_s" in frame.columns else pd.Series(dtype=float)
    humidity = (
        frame["relative_humidity_percent"]
        if "relative_humidity_percent" in frame.columns
        else pd.Series(dtype=float)
    )
    return {
        "weather_record_count": int(len(frame)),
        "skipped_missing_required_records": int(
            frame.attrs.get("skipped_missing_required_records", 0)
        ),
        "mean_outdoor_temperature_c": float(frame["outdoor_temperature_c"].mean()),
        "minimum_outdoor_temperature_c": float(frame["outdoor_temperature_c"].min()),
        "maximum_outdoor_temperature_c": float(frame["outdoor_temperature_c"].max()),
        "daily_solar_sum_wh_m2": float(frame["solar_radiation_w_m2"].sum()),
        "mean_wind_speed_m_s": (
            float(wind.mean()) if len(wind) and wind.notna().any() else float("nan")
        ),
        "mean_relative_humidity_percent": (
            float(humidity.mean()) if len(humidity) and humidity.notna().any() else float("nan")
        ),
    }


def fetch_or_load_weather(
    scenario: WeatherScenario,
    cache_dir: Path | str,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    transport: Transport | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (weather frame, provenance info) for one scenario.

    The requested date is tried first and the documented fallback date second.
    Raw NASA payloads are cached on disk under ``cache_dir`` so repeated runs
    never re-download identical requests.  ``transport`` is injectable for
    offline tests; ``None`` performs a live NASA POWER request through the
    existing client.  No synthetic weather is ever substituted.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    transport_kind = "live_nasa_power" if transport is None else "injected_transport"
    attempts: list[dict[str, str]] = []
    for label, requested in (
        ("requested", scenario.weather_date),
        ("fallback", scenario.fallback_date),
    ):
        try:
            nasa_date = _validate_shape_date(requested)
            cache_path = cache_dir / _cache_file_name(latitude, longitude, nasa_date, "RE")
            if cache_path.exists():
                payload, cached_kind = _read_cache(cache_path)
                status = f"disk_cache_from_{cached_kind}"
            else:
                payload = fetch_nasa_power_hourly(
                    latitude,
                    longitude,
                    requested,
                    requested,
                    parameters=DEFAULT_NASA_PARAMETERS,
                    community="RE",
                    timeout_seconds=timeout_seconds,
                    transport=transport,
                )
                _write_cache(cache_path, payload, transport_kind)
                status = transport_kind
            frame = parse_nasa_power_hourly(payload, DEFAULT_NASA_PARAMETERS)
            if len(frame) < MIN_HOURLY_RECORDS:
                raise NasaWeatherError(
                    f"only {len(frame)} usable hourly records for {requested}; "
                    f"need at least {MIN_HOURLY_RECORDS}"
                )
            info: dict[str, Any] = {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "season": scenario.season,
                "requested_date": scenario.weather_date,
                "effective_date": requested,
                "date_was_replaced": label == "fallback",
                "retrieval_status": status,
                "cache_file": cache_path.name,
                "payload_sha256": _payload_sha256(payload),
                "transport_kind": transport_kind,
            }
            info.update(summarize_weather(frame))
            return frame.reset_index(drop=True), info
        except (NasaWeatherError, ValueError) as error:
            attempts.append({"date": requested, "kind": label, "error": str(error)})
    raise NasaWeatherError(
        f"scenario {scenario.scenario_id} failed for the requested date "
        f"{scenario.weather_date} and the fallback date {scenario.fallback_date}: "
        + " | ".join(entry["error"] for entry in attempts)
    )


def _draw_design(rng: np.random.Generator) -> dict[str, Any]:
    """Draw one candidate design from the documented design space."""
    def insulation_thickness() -> float:
        if rng.random() < NO_INSULATION_PROBABILITY:
            return 0.0
        return round(float(rng.uniform(*INSULATION_THICKNESS_RANGE)), 4)

    has_mass = rng.random() >= NO_MASS_PROBABILITY
    mass_material = MASS_MATERIALS[int(rng.integers(len(MASS_MATERIALS)))] if has_mass else "none"
    mass_kg = round(float(rng.uniform(*MASS_RANGE)), 1) if has_mass else 0.0
    mass_cp = (
        float(REFERENCE_MATERIALS.get(mass_material).specific_heat_j_kgk)
        if has_mass
        else 0.0
    )
    return {
        "wall_material": WALL_MATERIALS[int(rng.integers(len(WALL_MATERIALS)))],
        "wall_structural_thickness_m": WALL_STRUCTURAL_THICKNESS_M,
        "wall_insulation_thickness_m": insulation_thickness(),
        "roof_insulation_thickness_m": insulation_thickness(),
        "floor_insulation_thickness_m": insulation_thickness(),
        "length_m": round(float(rng.uniform(*LENGTH_RANGE)), 3),
        "width_m": WIDTH_M,
        "height_m": HEIGHT_M,
        "orientation_deg": float(
            SHELTER_ORIENTATIONS[int(rng.integers(len(SHELTER_ORIENTATIONS)))]
        ),
        "window_area_m2": round(float(rng.uniform(*WINDOW_AREA_RANGE)), 3),
        "window_wall_orientation": CARDINAL_DIRECTIONS[
            int(rng.integers(len(CARDINAL_DIRECTIONS)))
        ],
        "door_area_m2": DOOR_AREA_M2,
        "door_wall_orientation": CARDINAL_DIRECTIONS[
            int(rng.integers(len(CARDINAL_DIRECTIONS)))
        ],
        "window_solar_heat_gain_coefficient": round(float(rng.uniform(*SHGC_RANGE)), 3),
        "window_u_value_w_m2k": 5.0,
        "door_u_value_w_m2k": 2.0,
        "occupant_count": INTERNAL_HEAT_ASSUMPTIONS["occupant_count"],
        "sensible_heat_per_person_w": INTERNAL_HEAT_ASSUMPTIONS["sensible_heat_per_person_w"],
        "equipment_heat_w": INTERNAL_HEAT_ASSUMPTIONS["equipment_heat_w"],
        "lighting_heat_w": INTERNAL_HEAT_ASSUMPTIONS["lighting_heat_w"],
        "has_thermal_mass": 1 if has_mass else 0,
        "thermal_mass_material": mass_material,
        "thermal_mass_kg": mass_kg,
        "thermal_mass_specific_heat_j_kgk": mass_cp,
        "thermal_mass_heat_capacity_j_k": round(mass_kg * mass_cp, 3),
    }


def _design_key(design: dict[str, Any]) -> tuple:
    return tuple(
        design[column]
        for column in DESIGN_PARAMETER_COLUMNS
        if column in design and column != "design_id"
    )


def generate_designs(count: int, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Deterministically sample ``count`` unique design parameter records.

    The same seed always yields the same designs in the same order.  Designs
    are sorted by a canonical parameter key and then assigned stable ids, so
    ids are reproducible too.  Duplicate parameter sets are rejected, never
    silently kept.
    """
    if count <= 0:
        raise ValueError("count must be a positive number of designs")
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    designs: list[dict[str, Any]] = []
    max_attempts = max(2000, count * 60)
    attempts = 0
    while len(designs) < count:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"could only generate {len(designs)} unique designs out of {count} requested"
            )
        candidate = _draw_design(rng)
        key = _design_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        designs.append(candidate)
    designs.sort(key=_design_key)
    for index, design in enumerate(designs):
        design["design_id"] = f"D{index:04d}"
    return designs

def build_shelter_config(design: dict[str, Any], name: str | None = None) -> ShelterConfig:
    """Build a validated :class:`ShelterConfig` from one design record.

    Only the existing reference material library is used; material properties
    are never invented.  The config must pass ``validate_shelter_config`` or
    this function raises.
    """
    wall_material = REFERENCE_MATERIALS.get(design["wall_material"])
    insulation = REFERENCE_MATERIALS.get(INSULATION_MATERIAL)
    slab_material = REFERENCE_MATERIALS.get(ROOF_STRUCTURAL_MATERIAL)

    def insulation_layer(thickness: float) -> Layer:
        return Layer(
            thickness,
            insulation.thermal_conductivity_w_mk,
            insulation.density_kg_m3,
            insulation.specific_heat_j_kgk,
            insulation.name,
        )

    wall_layers = [
        Layer(
            float(design["wall_structural_thickness_m"]),
            wall_material.thermal_conductivity_w_mk,
            wall_material.density_kg_m3,
            wall_material.specific_heat_j_kgk,
            wall_material.name,
        )
    ]
    if float(design["wall_insulation_thickness_m"]) > 0.0:
        wall_layers.append(insulation_layer(float(design["wall_insulation_thickness_m"])))

    roof_layers = []
    if float(design["roof_insulation_thickness_m"]) > 0.0:
        roof_layers.append(insulation_layer(float(design["roof_insulation_thickness_m"])))
    roof_layers.append(
        Layer(
            ROOF_STRUCTURAL_THICKNESS_M,
            slab_material.thermal_conductivity_w_mk,
            slab_material.density_kg_m3,
            slab_material.specific_heat_j_kgk,
            slab_material.name,
        )
    )

    floor_layers = []
    if float(design["floor_insulation_thickness_m"]) > 0.0:
        floor_layers.append(insulation_layer(float(design["floor_insulation_thickness_m"])))
    floor_layers.append(
        Layer(
            FLOOR_STRUCTURAL_THICKNESS_M,
            slab_material.thermal_conductivity_w_mk,
            slab_material.density_kg_m3,
            slab_material.specific_heat_j_kgk,
            slab_material.name,
        )
    )

    thermal_mass = None
    if design["has_thermal_mass"]:
        thermal_mass = ThermalMass(
            mass_kg=float(design["thermal_mass_kg"]),
            specific_heat_j_kgk=float(design["thermal_mass_specific_heat_j_kgk"]),
            initial_temperature_c=MASS_INITIAL_TEMPERATURE_C,
            material_name=str(design["thermal_mass_material"]),
        )

    config = ShelterConfig(
        name=name or str(design["design_id"]),
        geometry=ShelterGeometry(
            length_m=float(design["length_m"]),
            width_m=float(design["width_m"]),
            height_m=float(design["height_m"]),
            orientation_deg=float(design["orientation_deg"]),
        ),
        wall_assembly=EnvelopeAssembly("wall", wall_layers),
        roof_assembly=EnvelopeAssembly("roof", roof_layers),
        floor_assembly=EnvelopeAssembly("floor", floor_layers),
        openings=Openings(
            window_area_m2=float(design["window_area_m2"]),
            door_area_m2=float(design["door_area_m2"]),
            window_solar_heat_gain_coefficient=float(
                design["window_solar_heat_gain_coefficient"]
            ),
            window_wall_orientation=str(design["window_wall_orientation"]),
            door_wall_orientation=str(design["door_wall_orientation"]),
        ),
        thermal_mass=thermal_mass,
        internal_heat_sources=InternalHeatSources(**INTERNAL_HEAT_ASSUMPTIONS),
        initial_indoor_temperature_c=INITIAL_INDOOR_TEMPERATURE_C,
    )
    problems = validate_shelter_config(config)
    if problems:
        raise ValueError(
            f"design {design.get('design_id', name)!r} failed validation: {problems}"
        )
    return config

def simulate_design_scenario(
    config: ShelterConfig,
    weather: pd.DataFrame,
    comfort_range: ComfortRange | None = None,
) -> dict[str, float]:
    """Run the existing engine for one design on one weather scenario.

    Returns the raw physical metrics from ``design_metrics``.  The thermal
    engine is never bypassed or reimplemented here.
    """
    result = simulate_shelter(
        config, weather, weather_category=DataCategory.MEASURED
    )
    metrics = design_metrics(result, comfort_range)
    metrics.pop("design", None)
    return metrics


def build_dataset_row(
    design: dict[str, Any],
    scenario_info: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Assemble one dataset row: design parameters + weather + metrics."""
    row: dict[str, Any] = {}
    for column in DESIGN_PARAMETER_COLUMNS:
        if column in design:
            row[column] = design[column]
        else:
            raise ValueError(f"design record is missing required column {column!r}")
    for column in WEATHER_COLUMNS:
        if column in scenario_info:
            row[column] = scenario_info[column]
        else:
            raise ValueError(f"scenario info is missing required column {column!r}")
    for column in TARGET_COLUMNS:
        if column in metrics:
            row[column] = metrics[column]
        else:
            raise ValueError(f"simulation metrics are missing required column {column!r}")
    return row


@dataclass
class DatasetResult:
    """Everything produced by one dataset generation run."""

    frame: pd.DataFrame
    metadata: dict[str, Any]
    quality_report: dict[str, Any]
    designs: list[dict[str, Any]]
    scenario_infos: list[dict[str, Any]]
    failed_scenarios: list[dict[str, Any]]

def _row_scenario_info(
    info: dict[str, Any], latitude: float, longitude: float
) -> dict[str, Any]:
    """Map fetch provenance info onto the dataset weather columns."""
    return {
        "weather_scenario_id": info["scenario_id"],
        "weather_name": info["name"],
        "weather_season": info["season"],
        "weather_date": info["requested_date"],
        "weather_effective_date": info["effective_date"],
        "latitude": latitude,
        "longitude": longitude,
        "weather_source": WEATHER_SOURCE_LABEL,
        "weather_data_category": WEATHER_CATEGORY_LABEL,
        "weather_provenance": NASA_PROVENANCE_STATEMENT,
        "weather_retrieval_status": info["retrieval_status"],
        "weather_record_count": info["weather_record_count"],
        "skipped_missing_required_records": info["skipped_missing_required_records"],
        "mean_outdoor_temperature_c": info["mean_outdoor_temperature_c"],
        "minimum_outdoor_temperature_c": info["minimum_outdoor_temperature_c"],
        "maximum_outdoor_temperature_c": info["maximum_outdoor_temperature_c"],
        "daily_solar_sum_wh_m2": info["daily_solar_sum_wh_m2"],
        "mean_wind_speed_m_s": info["mean_wind_speed_m_s"],
        "mean_relative_humidity_percent": info["mean_relative_humidity_percent"],
    }


def generate_ml_dataset(
    design_count: int = DEFAULT_DESIGN_COUNT,
    seed: int = DEFAULT_SEED,
    scenarios: Sequence[WeatherScenario] | None = None,
    cache_dir: Path | str = Path("data") / "nasa_weather_raw",
    transport: Transport | None = None,
    timeout_seconds: float = 60.0,
    comfort_range: ComfortRange | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> DatasetResult:
    """Generate the full design-by-weather dataset.

    Pipeline per row: design record -> ShelterConfig (validated) -> NASA
    weather DataFrame -> simulate_shelter -> design_metrics -> dataset row.
    The same weather frame is reused for every design within a scenario.
    """
    used_scenarios = list(scenarios) if scenarios is not None else get_weather_scenarios()
    if not used_scenarios:
        raise ValueError("at least one weather scenario is required")
    comfort = comfort_range if comfort_range is not None else ComfortRange()

    designs = generate_designs(design_count, seed)
    configs = [build_shelter_config(design) for design in designs]
    for design, built in zip(designs, configs):
        geometry = build_geometry(built.geometry, built.openings)
        design["floor_area_m2"] = geometry.floor_area_m2
        design["volume_m3"] = geometry.volume_m3
        design["net_wall_area_m2"] = geometry.net_wall_area_m2

    scenario_infos: list[dict[str, Any]] = []
    failed_scenarios: list[dict[str, Any]] = []
    weather_frames: dict[str, pd.DataFrame] = {}
    for scenario in used_scenarios:
        try:
            frame, info = fetch_or_load_weather(
                scenario,
                cache_dir,
                latitude=latitude,
                longitude=longitude,
                transport=transport,
                timeout_seconds=timeout_seconds,
            )
        except NasaWeatherError as error:
            failed_scenarios.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "requested_date": scenario.weather_date,
                    "fallback_date": scenario.fallback_date,
                    "error": str(error),
                }
            )
            continue
        scenario_infos.append(info)
        weather_frames[scenario.scenario_id] = frame
    if not scenario_infos:
        raise NasaWeatherError(
            "no weather scenario could be retrieved from NASA POWER; the dataset "
            "was not generated and no synthetic substitute was used"
        )

    rows: list[dict[str, Any]] = []
    for info in scenario_infos:
        row_info = _row_scenario_info(info, latitude, longitude)
        frame = weather_frames[info["scenario_id"]]
        for design, config in zip(designs, configs):
            metrics = simulate_design_scenario(config, frame, comfort)
            rows.append(build_dataset_row(design, row_info, metrics))

    frame = pd.DataFrame(rows, columns=list(DATASET_COLUMNS))
    problems = validate_dataset(frame, len(designs), len(scenario_infos))
    if problems:
        raise ValueError("generated dataset failed validation: " + "; ".join(problems))

    metadata = build_metadata(
        designs=designs,
        scenario_infos=scenario_infos,
        failed_scenarios=failed_scenarios,
        frame=frame,
        seed=seed,
        cache_dir=cache_dir,
        latitude=latitude,
        longitude=longitude,
        live_retrieval=transport is None,
    )
    quality = quality_report(
        frame,
        designs,
        scenario_infos,
        requested_scenarios=len(used_scenarios),
        live_retrieval=transport is None,
    )
    return DatasetResult(
        frame=frame,
        metadata=metadata,
        quality_report=quality,
        designs=designs,
        scenario_infos=scenario_infos,
        failed_scenarios=failed_scenarios,
    )

def validate_dataset(
    frame: pd.DataFrame,
    expected_designs: int,
    expected_scenarios: int,
) -> list[str]:
    """Structural and provenance checks; an empty list means the data pass."""
    problems: list[str] = []
    if frame.empty:
        return ["dataset frame is empty"]
    missing_columns = [c for c in DATASET_COLUMNS if c not in frame.columns]
    if missing_columns:
        problems.append(f"missing columns: {missing_columns}")
        return problems
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any():
            problems.append(f"required column {column!r} contains NaN")
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=list(OPTIONAL_COLUMNS), errors="ignore"
    )
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            problems.append(f"numeric column {column!r} contains non-finite values")
    duplicates = frame.duplicated(subset=["design_id", "weather_scenario_id"]).sum()
    if duplicates:
        problems.append(f"{duplicates} duplicate design x scenario rows")
    unique_designs = frame["design_id"].nunique()
    if unique_designs != expected_designs:
        problems.append(
            f"expected {expected_designs} unique designs, found {unique_designs}"
        )
    unique_scenarios = frame["weather_scenario_id"].nunique()
    if unique_scenarios != expected_scenarios:
        problems.append(
            f"expected {expected_scenarios} unique weather scenarios, found {unique_scenarios}"
        )
    expected_rows = expected_designs * expected_scenarios
    if len(frame) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(frame)}")
    bad_materials = set(frame["wall_material"].unique()) - set(WALL_MATERIALS)
    if bad_materials:
        problems.append(f"unknown wall materials: {sorted(bad_materials)}")
    bad_mass = set(frame["thermal_mass_material"].unique()) - set(MASS_MATERIALS) - {"none"}
    if bad_mass:
        problems.append(f"unknown thermal mass materials: {sorted(bad_mass)}")
    sources = frame["weather_source"].unique()
    if any(not str(source).startswith("NASA POWER") for source in sources):
        problems.append(f"unexpected weather source labels: {sorted(sources)}")
    if (frame["window_area_m2"] < 0).any() or (frame["door_area_m2"] < 0).any():
        problems.append("negative opening areas present")
    if not frame["weather_provenance"].str.startswith("NASA POWER data are").all():
        problems.append("weather provenance statement missing from rows")
    return problems


def quality_report(
    frame: pd.DataFrame,
    designs: list[dict[str, Any]],
    scenario_infos: list[dict[str, Any]],
    requested_scenarios: int,
    live_retrieval: bool,
) -> dict[str, Any]:
    """The data-quality report required after every generation run."""
    missing_counts = {
        column: int(frame[column].isna().sum())
        for column in frame.columns
        if frame[column].isna().sum() > 0
    }
    target_statistics = {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
            "mean": float(frame[column].mean()),
        }
        for column in TARGET_COLUMNS
    }
    return {
        "nasa_scenarios_requested": int(requested_scenarios),
        "nasa_scenarios_retrieved": len(scenario_infos),
        "nasa_hourly_records_total": int(
            sum(info["weather_record_count"] for info in scenario_infos)
        ),
        "designs_generated": len(designs),
        "designs_valid": len(designs),
        "final_row_count": int(len(frame)),
        "columns": list(frame.columns),
        "missing_value_counts": missing_counts,
        "duplicate_row_count": int(
            frame.duplicated(subset=["design_id", "weather_scenario_id"]).sum()
        ),
        "target_statistics": target_statistics,
        "weather_scenario_summary": [
            {
                "scenario_id": info["scenario_id"],
                "requested_date": info["requested_date"],
                "effective_date": info["effective_date"],
                "date_was_replaced": info["date_was_replaced"],
                "retrieval_status": info["retrieval_status"],
                "records": info["weather_record_count"],
                "mean_outdoor_temperature_c": info["mean_outdoor_temperature_c"],
                "daily_solar_sum_wh_m2": info["daily_solar_sum_wh_m2"],
            }
            for info in scenario_infos
        ],
        "provenance": NASA_PROVENANCE_STATEMENT,
        "weather_from_live_nasa_power": bool(live_retrieval),
    }

def build_metadata(
    designs: list[dict[str, Any]],
    scenario_infos: list[dict[str, Any]],
    failed_scenarios: list[dict[str, Any]],
    frame: pd.DataFrame,
    seed: int,
    cache_dir: Path | str,
    latitude: float,
    longitude: float,
    live_retrieval: bool,
) -> dict[str, Any]:
    """Assemble the full provenance metadata document."""

    def numeric_range(values: Sequence[float]) -> dict[str, float]:
        finite = [float(v) for v in values if np.isfinite(v)]
        return {"min": min(finite), "max": max(finite)}

    metadata: dict[str, Any] = {
        "dataset_name": "shelter_ml_dataset",
        "generator_module": "building_hvac_twin.shelter.ml_dataset",
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "design_generation_method": (
            "seeded numpy Generator sampling over the documented design space "
            "with duplicate rejection, canonical parameter sort, and stable "
            "design ids"
        ),
        "nasa_power": {
            "source": "NASA POWER (Prediction Of Worldwide Energy Resources)",
            "endpoint": NASA_BASE_URL,
            "parameters": list(DEFAULT_NASA_PARAMETERS),
            "parameter_mapping": {
                "T2M": "outdoor_temperature_c",
                "ALLSKY_SFC_SW_DWN": "solar_radiation_w_m2",
                "WS10M": "wind_speed_m_s",
                "RH2M": "relative_humidity_percent",
            },
            "latitude": latitude,
            "longitude": longitude,
            "location_name": DEFAULT_LOCATION_NAME,
            "time_standard": "UTC",
            "provenance_statement": NASA_PROVENANCE_STATEMENT,
            "live_retrieval_statement": LIVE_NASA_STATEMENT
            if live_retrieval
            else (
                "Offline/test run: weather came from an injected mock transport, "
                "not from live NASA POWER requests."
            ),
            "retrieval_mode": "live" if live_retrieval else "mocked (offline test)",
            "raw_cache_directory": str(cache_dir),
            "missing_value_policy": (
                "NASA marks missing points with -999. Rows with missing required "
                "values (T2M, ALLSKY_SFC_SW_DWN) are dropped and counted in "
                "skipped_missing_required_records; missing optional values "
                "(WS10M, RH2M) become NaN in the optional columns only. No "
                "values are invented and no synthetic weather is substituted."
            ),
        },
    }
    metadata["weather_scenarios"] = {
        "count_used": len(scenario_infos),
        "count_failed": len(failed_scenarios),
        "used": scenario_infos,
        "failed": failed_scenarios,
        "dates_are_fixed": True,
        "note": (
            "Scenario dates are documented constants; a documented fallback "
            "date (seven days later) is used only when a requested date "
            "returns unusable NASA data, and any replacement is recorded "
            "here and in the weather_effective_date column."
        ),
    }
    metadata["designs"] = {
        "count": len(designs),
        "ranges": {
            "length_m": numeric_range([d["length_m"] for d in designs]),
            "wall_insulation_thickness_m": numeric_range(
                [d["wall_insulation_thickness_m"] for d in designs]
            ),
            "roof_insulation_thickness_m": numeric_range(
                [d["roof_insulation_thickness_m"] for d in designs]
            ),
            "floor_insulation_thickness_m": numeric_range(
                [d["floor_insulation_thickness_m"] for d in designs]
            ),
            "window_area_m2": numeric_range([d["window_area_m2"] for d in designs]),
            "window_solar_heat_gain_coefficient": numeric_range(
                [d["window_solar_heat_gain_coefficient"] for d in designs]
            ),
            "thermal_mass_kg": numeric_range(
                [d["thermal_mass_kg"] for d in designs if d["has_thermal_mass"]]
            ),
        },
        "categorical_values": {
            "wall_material": sorted({d["wall_material"] for d in designs}),
            "orientation_deg": sorted({d["orientation_deg"] for d in designs}),
            "window_wall_orientation": sorted(
                {d["window_wall_orientation"] for d in designs}
            ),
            "door_wall_orientation": sorted({d["door_wall_orientation"] for d in designs}),
            "thermal_mass_material": sorted({d["thermal_mass_material"] for d in designs}),
        },
        "fixed_values": {
            "width_m": WIDTH_M,
            "height_m": HEIGHT_M,
            "wall_structural_thickness_m": WALL_STRUCTURAL_THICKNESS_M,
            "roof_structural_thickness_m": ROOF_STRUCTURAL_THICKNESS_M,
            "floor_structural_thickness_m": FLOOR_STRUCTURAL_THICKNESS_M,
            "door_area_m2": DOOR_AREA_M2,
            "thermal_mass_initial_temperature_c": MASS_INITIAL_TEMPERATURE_C,
            "initial_indoor_temperature_c": INITIAL_INDOOR_TEMPERATURE_C,
            "comfort_range_c": {"min": 18.0, "max": 24.0},
            **INTERNAL_HEAT_ASSUMPTIONS,
            "air_capacity_per_volume_j_m3k": 1200.0,
            "surface_film_resistances_m2k_w": {"inner": 0.13, "outer": 0.04},
        },
    }
    metadata["dataset"] = {
        "row_count": int(len(frame)),
        "design_count": int(frame["design_id"].nunique()),
        "weather_scenario_count": int(frame["weather_scenario_id"].nunique()),
        "columns": list(DATASET_COLUMNS),
        "feature_columns": list(DESIGN_PARAMETER_COLUMNS)
        + [
            "mean_outdoor_temperature_c",
            "minimum_outdoor_temperature_c",
            "maximum_outdoor_temperature_c",
            "daily_solar_sum_wh_m2",
        ],
        "target_columns": list(TARGET_COLUMNS),
        "excluded_columns": {
            "performance_score": (
                "relative within-batch comparison score from comparison.py; "
                "intentionally excluded and never used as an ML target"
            ),
            "auxiliary_heating_or_cooling_energy": (
                "the thermal engine is passive-only and computes no HVAC "
                "energy, so none is fabricated here"
            ),
            "number_of_windows": "label only, no effect on engine physics",
            "number_of_doors": "label only, no effect on engine physics",
            "ground_temperature_c": (
                "left unset; the engine uses outdoor air for the floor path"
            ),
        },
        "optional_columns": list(OPTIONAL_COLUMNS),
        "validation_policy": (
            "every design passed validate_shelter_config before simulation; the "
            "assembled dataset passes validate_dataset (required columns, no NaN "
            "in required fields, finite numerics, no duplicate design-scenario "
            "rows, provenance labels present)"
        ),
        "row_order": "scenario-major, then canonical design_id order",
    }
    return metadata


def write_dataset(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write the dataset CSV with a fixed column order and no index column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=list(DATASET_COLUMNS))
    return path


def write_metadata(metadata: dict[str, Any], path: Path | str) -> Path:
    """Write the metadata JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


    return metadata










