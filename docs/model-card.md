# Model Card

## Purpose
A transparent vendor-neutral R&D baseline for single-zone building heat balance, HVAC energy, occupied comfort, calibration, validation, and setpoint studies.

## Model
The 1R1C balance is `C dTin/dt = UA(Tout - Tin) + Qinternal + Qsolar + Qheat - Qcool`. Ideal bounded heating and cooling loads are converted to electric power using COP plus fan and base power.

## Inputs and outputs
Inputs are documented in the data contract. Outputs include indoor temperature, heating and cooling load, equipment power, interval energy, cost, and occupied comfort violation.

## Calibration and uncertainty
Bounded nonlinear least squares jointly fits temperature and weighted energy residuals. Approximate local standard errors use the inverse Jacobian information matrix. These intervals do not capture model-form error, sensor bias, or non-Gaussian uncertainty.

## Appropriate use
Research baselines, data-quality checks, hypothesis generation, calibration workflow development, and preliminary control trade studies.

## Inappropriate use
Safety-critical control, code compliance, equipment sizing, guaranteed savings, fault diagnosis without validation, or production deployment based only on bundled synthetic results.
