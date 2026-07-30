# Agent Playbook

## Gate 1, reproduce baseline
`python -m pip install -e . && pytest -q && python tests/smoke_test.py`

## Gate 2, validate uploaded data
Copy raw data to a new project without modification. Record provenance and timezone. Run `hvac-twin validate-data FILE`. Stop on any error.

## Gate 3, declare split
Use chronological calibration and held-out validation periods. Record dates, exclusions, and reasons.

## Gate 4, calibrate and validate
Run the workflow. Report parameters, uncertainty intervals, train and validation metrics, residual bias, spread, autocorrelation, and range limitations.

## Gate 5, optimize only after validation
Define cost tariff, comfort rule, bounds, and uncertainty. Retain all candidates or Pareto points. If no candidate satisfies the constraint, report infeasibility rather than hiding the constraint.

## Required completion report
State software version, dataset ID, synthetic or measured status, split, model assumptions, metrics, uncertainty method, optimization constraint status, limitations, commands, and output paths.
