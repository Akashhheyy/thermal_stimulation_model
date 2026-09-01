"""Physical sanity tests for thermal mass behaviour (Q = m * cp * dT)."""
import pytest

from building_hvac_twin.shelter import (
    ThermalMass,
    ThermalMassState,
    effective_capacitance_j_k,
    heat_absorbed_j,
    heat_flow_w,
    heat_released_j,
    stored_energy_j,
    temperature_change_c,
)

WATER_CP = 4186.0


def test_stored_energy_follows_m_cp_delta_t():
    # 2 kg water warmed 10 C above the reference stores exactly m*cp*dT.
    assert stored_energy_j(2.0, WATER_CP, 30.0, 20.0) == pytest.approx(2.0 * WATER_CP * 10.0)
    # Doubling the temperature difference doubles the stored energy.
    double = stored_energy_j(2.0, WATER_CP, 40.0, 20.0)
    assert double == pytest.approx(2.0 * stored_energy_j(2.0, WATER_CP, 30.0, 20.0))


def test_absorb_raises_and_release_lowers_temperature():
    state = ThermalMassState(mass_kg=2.0, specific_heat_j_kgk=WATER_CP, temperature_c=20.0)
    energy = 2.0 * WATER_CP * 10.0  # enough for exactly +10 C
    assert state.absorb(energy) == pytest.approx(30.0)
    assert state.release(energy) == pytest.approx(20.0)


def test_heat_absorbed_and_released_are_directional_and_nonnegative():
    absorbed = heat_absorbed_j(10.0, 900.0, 5.0, 15.0)
    released = heat_released_j(10.0, 900.0, 15.0, 5.0)
    assert absorbed == pytest.approx(10.0 * 900.0 * 10.0)
    assert released == pytest.approx(10.0 * 900.0 * 10.0)
    # A mass that cooled cannot have absorbed heat, and vice versa.
    assert heat_absorbed_j(10.0, 900.0, 15.0, 5.0) == 0.0
    assert heat_released_j(10.0, 900.0, 5.0, 15.0) == 0.0


def test_higher_mass_reduces_temperature_change_for_same_heat():
    heat_j = 50_000.0
    light = temperature_change_c(heat_j, 100.0, 900.0)
    heavy = temperature_change_c(heat_j, 400.0, 900.0)
    assert heavy == pytest.approx(light / 4.0)
    assert heavy < light


def test_storage_rate_sign_matches_temperature_direction():
    warming = heat_flow_w(100.0, 900.0, +2.0, 3600.0)
    cooling = heat_flow_w(100.0, 900.0, -2.0, 3600.0)
    assert warming > 0.0, "absorbing heat must be a positive storage rate"
    assert cooling < 0.0, "releasing heat must be a negative storage rate"
    assert warming == pytest.approx(-cooling)
    # watts = J/K * K / s
    assert warming == pytest.approx(100.0 * 900.0 * 2.0 / 3600.0)


def test_state_tracks_mass_temperature_dynamically():
    mass = ThermalMass(mass_kg=500.0, specific_heat_j_kgk=900.0, initial_temperature_c=5.0)
    state = ThermalMassState.from_thermal_mass(mass)
    assert state.temperature_c == pytest.approx(5.0)
    assert state.heat_capacity_j_k == pytest.approx(effective_capacitance_j_k(500.0, 900.0))
    # Absorbing the same energy twice compounds, so the mass is a dynamic store.
    step = 500.0 * 900.0 * 1.0
    state.absorb(step)
    first = state.temperature_c
    state.absorb(step)
    assert state.temperature_c == pytest.approx(first + 1.0)
    assert state.storage_rate_w(5.0, 7.0, 3600.0) == pytest.approx(500.0 * 900.0 * 2.0 / 3600.0)


def test_invalid_thermal_mass_inputs_are_rejected():
    with pytest.raises(ValueError):
        stored_energy_j(0.0, WATER_CP, 20.0)
    with pytest.raises(ValueError):
        temperature_change_c(1000.0, 10.0, -1.0)
    with pytest.raises(ValueError):
        heat_flow_w(10.0, 900.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        ThermalMassState(mass_kg=-5.0, specific_heat_j_kgk=900.0, temperature_c=10.0)
    with pytest.raises(ValueError):
        ThermalMass(mass_kg=100.0, specific_heat_j_kgk=0.0, initial_temperature_c=5.0)
