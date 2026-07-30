# Data Contract

Each row is one regular time interval. Required columns and units are machine-readable in `schemas/`. `energy_kwh` is interval energy, not cumulative meter reading. `equipment_status` is 0 or 1 and means HVAC disabled or enabled. Setpoints must use the same zone and interval as indoor temperature. Weather and solar must represent a documented source and exposure.

A dataset manifest must record dataset ID, source, license, synthetic or measured status, timezone, daylight-saving handling, nominal interval, missing-data policy, sensor IDs, calibration period, validation period, exclusions, and revisions. Raw uploads are immutable. Transformations create new files with provenance. Missing fields must cause validation failure, not silent imputation.
