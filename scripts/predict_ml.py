"""Generate surrogate-model predictions for new shelter design/weather inputs.

The predictions use the SAME fitted pipelines (each containing its own
preprocessing) produced by scripts/train_ml_models.py, so the feature ordering
and encodings are identical to training.  This is a surrogate layer on top of
the simulation dataset; the thermal engine is not invoked here.

Two input modes are supported:

1. Existing dataset rows:  --design-id D0000 --weather-scenario-id S01_winter
2. Raw features via JSON:  --config-features config.json --weather-features weather.json

Run from the repository root:

    .venv\\Scripts\\python.exe scripts\\predict_ml.py --design-id D0000 --weather-scenario-id S01_winter
    .venv\\Scripts\\python.exe scripts\\predict_ml.py --config-features my/config.json --weather-features my/weather.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from building_hvac_twin.ml import (  # noqa: E402
    load_dataset_row_features,
    predict_features,
    load_trained_models,
)

DEFAULT_MODELS_DIR = Path("models") / "regression"
DEFAULT_DATASET_PATH = Path("data") / "shelter_ml_dataset.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--design-id", default=None)
    parser.add_argument("--weather-scenario-id", default=None)
    parser.add_argument("--config-features", default=None, help="JSON file with design feature values")
    parser.add_argument("--weather-features", default=None, help="JSON file with weather feature values")
    parser.add_argument("--target", default=None, help="limit predictions to one target")
    args = parser.parse_args()

    if args.design_id is not None or args.weather_scenario_id is not None:
        if not (args.design_id and args.weather_scenario_id):
            parser.error("--design-id and --weather-scenario-id must be given together")
        if args.config_features or args.weather_features:
            parser.error("choose either dataset-row mode or JSON mode, not both")
    elif args.config_features and args.weather_features:
        config = json.loads(Path(args.config_features).read_text(encoding="utf-8"))
        weather = json.loads(Path(args.weather_features).read_text(encoding="utf-8"))
        from building_hvac_twin.ml.features import build_feature_row

        frame = build_feature_row(config, weather)
    else:
        parser.error(
            "provide either --design-id + --weather-scenario-id or "
            "--config-features + --weather-features"
        )
        raise SystemExit(2)

    models_dir = Path(args.models_dir)
    models = load_trained_models(models_dir)

    if args.design_id is not None:
        frame = load_dataset_row_features(
            args.design_id, args.weather_scenario_id, args.dataset
        )

    predictions = predict_features(models, frame, target=args.target)
    print("SHELTER ML SURROGATE MODEL PREDICTIONS")
    print(f"Models: {models_dir}")
    if args.design_id is not None:
        print(f"Input: design {args.design_id} x scenario {args.weather_scenario_id}")
    print()
    print(predictions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())