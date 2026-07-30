# Reference Runs

All results are synthetic Level 0 execution and scientific sanity evidence, not measured building validation.

## Environment and routing

- Command: `python scripts/run_reference_cases.py`
- Simulator: `building_hvac_twin` 1R1C explicit-Euler model, package 0.1.0
- Python: 3.12.10
- Units: degC, kW, kWh per interval, kW/K, kWh/K, hours
- Random seed: 42
- Runtime: 0.020 seconds for the compact reference cases
- Remote compute: not submitted. The registered CFD workflows cover cavity flow and NACA0012, neither is scientifically applicable to this lumped single-zone thermal model.

## R1, smallest meaningful baseline, PASS

A deterministic 24-hour hourly synthetic case completed with **32.17657 kWh** electricity, **69.52971 kWh thermal heating**, **19.17411 degC** mean indoor temperature, and **5.70744 occupied comfort degree-hours**.

## R2, physical parameter response, PASS

Envelope conductance increased from 0.420 to 0.525 kW/K. Electricity increased from **32.17657 to 36.72164 kWh**, a **14.12540 percent** increase.

## R3, analytical convergence, PASS

For free 1R1C decay against `Tin(t)=Tout+(Tin0-Tout) exp(-UA t/C)`, maximum absolute error decreased from **0.384020 degC** at a 1-hour step to **0.092940 degC** at a 0.25-hour step, an error-reduction factor of **4.1319**.

## R4, intentional invalid input, PASS

Removing `energy_kwh` produced the deterministic rejection: `Missing columns: energy_kwh`.

## R5, full workflow and packaged quickstart

Commands: `hvac-twin validate-data datasets/example/building_timeseries.csv`, `python examples/quickstart.py`, and `hvac-twin run datasets/example/building_timeseries.csv --output outputs/reference_run`.

## Artifacts and SHA-256

- `results/reference-runs/baseline_predictions.csv`: `b5e7584d4e40477efea86fd75884c7cbaa4c6b7168f35f86762d7b8c608927c8`
- `results/reference-runs/case_metrics.csv`: `8177d3e855befa5a16be9425eb2ce5ce8736d4429958864f726add6e5f4973f8`
- `results/reference-runs/final_generate_console.txt`: `7e33881ed02383fdeb5fdcda0ba2674ba205ebdacfa74f321572a1af44649ec7`
- `results/reference-runs/final_onepager_console.txt`: `d797f8d88bd2af375082e282572a1604027bac2d58bd06c6862812fb8aaec023`
- `results/reference-runs/final_quickstart_console.txt`: `bcdf8bddd94393618d141644caf8e06468d95181f93d04dab273fb6a14f6b4fc`
- `results/reference-runs/final_validate_console.txt`: `655823ac7f996b259837b1279f4757264aac1ab82808d9b562c06a9f640f36af`
- `results/reference-runs/final_workflow_console.txt`: `2cdc3523d412c64a91155afb866458d4fd96f324751b1ef996bcb2288973fb75`
- `results/reference-runs/reference_run_summary.json`: `70d5868937080e4f80a79f2ae97657265a6deceaddb6fc0f03e33aba8fd6b61b`

## Limitations

Single well-mixed zone, explicit first-order integration, idealized thermostat loads, constant COP, simplified gains, no humidity or ventilation physics, local Jacobian uncertainty, and a temperature-band comfort proxy. See `docs/limitations.md`.
