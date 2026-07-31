# Measured validation lane

## Scope

Release 0.2.0 adds a measured-data Level 1 result for one-week-ahead hourly HVAC electricity prediction. This claim applies only to the forecasting lane, not to the complete thermal twin.

The transparent ridge model uses the HVAC reading from 168 hours earlier, outdoor temperature, humidity, hour, weekday, and month. It is not a control policy or causal savings estimator.

## Fixed chronological protocol

- Training: 2015-01-08 through 2016-06-30.
- Interval calibration: 2016-07-01 through 2016-12-31.
- Untouched test: all of 2017.
- Comparator: weekly persistence.
- Uncertainty: symmetric empirical intervals from the calibration residual 95th percentile. Temporal dependence means coverage is empirical, not a guarantee for deployment.

## Executed result

Across 20 measured office buildings:

- median model CVRMSE: 76.634 percent;
- median persistence CVRMSE: 89.185 percent;
- model RMSE lower than persistence for 18 of 20 buildings;
- median empirical interval coverage: 0.952.

Errors remain large, and two buildings did not beat persistence. This supports a measured forecasting benchmark, not a production controller.

## Reproduction

Extract the EnergyDetective archive under `datasets/external/energy_detective_2020_v2`, then run `PYTHONPATH=src python scripts/run_measured_validation.py datasets/external/energy_detective_2020_v2 --output outputs/measured_energy_detective`.

## Remaining path

Indoor-temperature and comfort validation require aligned zone temperature, HVAC actuation or thermal delivery, occupancy, weather, and geometry. Causal savings need a field experiment or defensible counterfactual. Multiple sites and seasons are required for robust-design claims.
