# API Overview

- `TwinParameters`: baseline parameter dataclass.
- `simulate(data, params)`: deterministic simulation.
- `calibrate(train, initial)`: bounded parameter fitting and local uncertainty.
- `evaluate(data, params)`: predictions and metrics.
- `residual_diagnostics(data, simulation)`: residual statistics.
- `sensitivity(data, params)`: one-at-a-time parameter sensitivity.
- `optimize_setpoints(data, params, limit)`: grid search with comfort constraint.
- `workflow.run(dataset, output, train_fraction)`: complete reproducible workflow.
