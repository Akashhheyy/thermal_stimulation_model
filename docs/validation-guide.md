# Validation Guide

1. Validate schema, monotonic timestamps, units, setpoint ordering, and physical ranges.
2. Plot all signals and document missingness, resets, clipping, sensor changes, and operating modes.
3. Select a chronological calibration period and a later held-out validation period before fitting.
4. Calibrate within defensible parameter bounds and report bound hits and parameter uncertainty.
5. Report MAE, RMSE, bias, CVRMSE, NMBE, residual quantiles, and autocorrelation separately for calibration and validation.
6. Stratify by season, occupied state, equipment mode, and outdoor temperature when measured data are available.
7. Perform sensitivity and uncertainty analysis. Do not hide infeasible optimization constraints.
8. State the validated range. Higher maturity requires multiple seasons and conditions.
