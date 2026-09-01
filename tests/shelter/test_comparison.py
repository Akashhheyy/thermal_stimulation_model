"""Tests for same-weather design comparison and the transparent score."""
from dataclasses import replace

import pandas as pd
import pytest

from building_hvac_twin.shelter import (
    ComfortRange,
    EnvelopeAssembly,
    Layer,
    Openings,
    PerformanceWeights,
    ShelterConfig,
    ThermalMass,
    compare_designs,
    design_metrics,
    performance_scores,
    simulate_designs,
    simulate_shelter,
)

BAND = ComfortRange(minimum_comfort_temperature_c=18.0, maximum_comfort_temperature_c=24.0)


def variant(config: ShelterConfig, name: str, **changes) -> ShelterConfig:
    return replace(config, name=name, **changes)


def test_identical_designs_produce_identical_results(reference_config, synthetic_weather):
    twin_a = variant(reference_config, "twin-a")
    twin_b = variant(reference_config, "twin-b")
    report = compare_designs([twin_a, twin_b], synthetic_weather, comfort_range=BAND)
    row_a = report.table.loc[report.table["design"] == "twin-a"].iloc[0]
    row_b = report.table.loc[report.table["design"] == "twin-b"].iloc[0]
    for column in report.table.columns:
        if column == "design":
            continue
        assert row_a[column] == pytest.approx(row_b[column]), column
    assert report.ranking == ["twin-a", "twin-b"]  # deterministic tie-break


def test_identical_weather_is_used_for_every_design(reference_config, synthetic_weather):
    designs = [
        variant(reference_config, "small", geometry=replace(reference_config.geometry, length_m=4.0)),
        variant(reference_config, "windowed", openings=replace(reference_config.openings, window_area_m2=4.0)),
        variant(reference_config, "heavy", thermal_mass=ThermalMass(2000.0, 900.0, 5.0)),
    ]
    results = simulate_designs(designs, synthetic_weather)
    expected_outdoor = synthetic_weather["outdoor_temperature_c"].to_numpy()
    expected_solar = synthetic_weather["solar_radiation_w_m2"].to_numpy()
    for result in results:
        assert result.records["outdoor_temperature_c"].to_numpy() == pytest.approx(expected_outdoor)
        assert result.records["solar_radiation_w_m2"].to_numpy() == pytest.approx(expected_solar)


def test_better_insulation_reduces_heat_loss(reference_config, synthetic_weather):
    insulated = variant(
        reference_config,
        "insulated",
        wall_assembly=EnvelopeAssembly(
            "wall",
            [
                *reference_config.wall_assembly.layers,
                Layer(0.1, 0.035, 30.0, 1400.0, "insulation"),
            ],
        ),
    )
    report = compare_designs([reference_config, insulated], synthetic_weather, comfort_range=BAND)
    table = report.table.set_index("design")
    assert table.loc["insulated", "total_heat_loss_kwh"] < table.loc[
        "reference-demo-shelter", "total_heat_loss_kwh"
    ]
    assert table.loc["insulated", "mean_indoor_temperature_c"] > table.loc[
        "reference-demo-shelter", "mean_indoor_temperature_c"
    ]


def test_increased_thermal_mass_reduces_temperature_swings(reference_config, synthetic_weather):
    no_mass = variant(reference_config, "no-mass", thermal_mass=None)
    heavy = variant(
        reference_config,
        "heavy",
        thermal_mass=ThermalMass(mass_kg=3000.0, specific_heat_j_kgk=900.0, initial_temperature_c=5.0),
    )
    report = compare_designs([no_mass, reference_config, heavy], synthetic_weather, comfort_range=BAND)
    table = report.table.set_index("design")
    assert table.loc["heavy", "indoor_temperature_range_c"] < table.loc[
        "reference-demo-shelter", "indoor_temperature_range_c"
    ]
    assert table.loc["reference-demo-shelter", "indoor_temperature_range_c"] < table.loc[
        "no-mass", "indoor_temperature_range_c"
    ]
    assert table.loc["heavy", "thermal_mass_absorbed_kwh"] > table.loc[
        "reference-demo-shelter", "thermal_mass_absorbed_kwh"
    ] > table.loc["no-mass", "thermal_mass_absorbed_kwh"]
    assert table.loc["no-mass", "thermal_mass_absorbed_kwh"] == pytest.approx(0.0)


def test_increased_window_area_increases_solar_contribution(reference_config, synthetic_weather):
    glazed = variant(
        reference_config,
        "glazed",
        openings=replace(
            reference_config.openings,
            window_area_m2=4.0,
            window_wall_orientation="south",
        ),
    )
    report = compare_designs([reference_config, glazed], synthetic_weather, comfort_range=BAND)
    table = report.table.set_index("design")
    assert table.loc["glazed", "total_solar_gain_kwh"] > table.loc[
        "reference-demo-shelter", "total_solar_gain_kwh"
    ]


def test_ranking_is_deterministic_across_repeated_runs(reference_config, synthetic_weather):
    designs = [
        reference_config,
        variant(reference_config, "insulated",
                wall_assembly=EnvelopeAssembly(
                    "wall",
                    [Layer(0.5, 0.035, 30.0, 1400.0, "insulation")])),
        variant(reference_config, "glazed",
                openings=replace(reference_config.openings, window_area_m2=4.0)),
    ]
    first = compare_designs(designs, synthetic_weather, comfort_range=BAND)
    second = compare_designs(list(designs), synthetic_weather, comfort_range=BAND)
    assert first.ranking == second.ranking
    assert first.table["performance_score"].tolist() == pytest.approx(
        second.table["performance_score"].tolist()
    )


def test_configurable_weights_change_scores_and_ranking():
    # Hand-built metric rows make the scoring contract explicit:
    # design A wins on comfort, design B wins on thermal stability.
    rows = [
        {
            "design": "A-comfortable",
            "percent_time_comfortable": 80.0,
            "total_heat_loss_kwh": 10.0,
            "total_solar_gain_kwh": 5.0,
            "indoor_temperature_range_c": 10.0,
        },
        {
            "design": "B-stable",
            "percent_time_comfortable": 40.0,
            "total_heat_loss_kwh": 10.0,
            "total_solar_gain_kwh": 5.0,
            "indoor_temperature_range_c": 2.0,
        },
    ]
    comfort_only = PerformanceWeights(comfort=1.0, heat_retention=0.0,
                                      solar_utilization=0.0, thermal_stability=0.0)
    stability_only = PerformanceWeights(comfort=0.0, heat_retention=0.0,
                                        solar_utilization=0.0, thermal_stability=1.0)
    scores_comfort = performance_scores(rows, comfort_only)
    scores_stable = performance_scores(rows, stability_only)
    assert scores_comfort[0] > scores_comfort[1]
    assert scores_stable[1] > scores_stable[0]


def test_pipeline_scores_respond_to_weight_change(reference_config, synthetic_weather):
    insulated = variant(
        reference_config,
        "insulated",
        wall_assembly=EnvelopeAssembly(
            "wall",
            [Layer(0.5, 0.035, 30.0, 1400.0, "insulation")],
        ),
    )
    designs = [reference_config, insulated]
    solar_heavy = compare_designs(
        designs,
        synthetic_weather,
        comfort_range=BAND,
        weights=PerformanceWeights(comfort=0.0, heat_retention=0.0,
                                   solar_utilization=1.0, thermal_stability=0.0),
    )
    retention_heavy = compare_designs(
        designs,
        synthetic_weather,
        comfort_range=BAND,
        weights=PerformanceWeights(comfort=0.0, heat_retention=1.0,
                                   solar_utilization=0.0, thermal_stability=0.0),
    )
    solar_scores = solar_heavy.table.set_index("design")["performance_score"]
    retention_scores = retention_heavy.table.set_index("design")["performance_score"]
    # The glazed-equivalent baseline captures more sun only via windows; the
    # insulated design must win retention, and scores must differ per weights.
    assert retention_scores["insulated"] > retention_scores["reference-demo-shelter"]
    assert solar_scores.tolist() != retention_scores.tolist()


def test_design_metrics_contain_required_measures(reference_config, synthetic_weather):
    result = simulate_shelter(reference_config, synthetic_weather)
    metrics = design_metrics(result, BAND)
    for column in (
        "percent_time_comfortable",
        "minimum_indoor_temperature_c",
        "maximum_indoor_temperature_c",
        "mean_indoor_temperature_c",
        "indoor_temperature_range_c",
        "degree_hours_below_comfort",
        "degree_hours_above_comfort",
        "total_heat_loss_kwh",
        "total_solar_gain_kwh",
        "thermal_mass_net_kwh",
    ):
        assert column in metrics
    # Energy columns are positive in a winter scenario and mass terms exist.
    assert metrics["total_heat_loss_kwh"] > 0.0
    assert metrics["total_solar_gain_kwh"] > 0.0


def test_degree_hours_track_comfort_percentage(reference_config, synthetic_weather):
    report = compare_designs([reference_config], synthetic_weather, comfort_range=BAND)
    row = report.table.iloc[0]
    assert row["percent_time_comfortable"] == pytest.approx(
        100.0
        - row["percent_time_below_comfort"]
        - row["percent_time_above_comfort"]
    )
    # A design below comfort the whole day must have positive below degree-hours
    # and zero above degree-hours (winter scenario never overshoots).
    if row["percent_time_above_comfort"] == pytest.approx(0.0):
        assert row["degree_hours_above_comfort"] == pytest.approx(0.0)


def test_invalid_weights_and_inputs_are_rejected(reference_config, synthetic_weather):
    with pytest.raises(ValueError):
        PerformanceWeights(comfort=-1.0)
    with pytest.raises(ValueError):
        PerformanceWeights(comfort=0.0, heat_retention=0.0,
                           solar_utilization=0.0, thermal_stability=0.0)
    with pytest.raises(ValueError):
        compare_designs([], synthetic_weather)
    with pytest.raises(ValueError):
        performance_scores([])

