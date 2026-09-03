"""Command-line interface for the prediction and recommendation layer.

Subcommands:

- predict   ML-predict one design under one scenario.
- recommend generate candidate designs, predict them and rank them.
- compare   cross-check ML against the thermal engine for one design.

Every command prints JSON so results are machine-readable and reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .predictor import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_METRICS_REPORT,
    DEFAULT_MODELS_DIR,
    load_predictors,
    predict_design,
)
from .ranking import DEFAULT_OBJECTIVES, RecommendationObjective, rank_designs
from .validation import compare_prediction_with_physics, load_scenario_weather


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="building_hvac_twin.recommendation",
        description="Predict, recommend and cross-check shelter designs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Predict one design under one scenario.")
    p.add_argument("--scenario", required=True, help="Weather scenario id, e.g. S01_winter.")
    p.add_argument("--design-id", help="Existing dataset design_id (reproducible row).")
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    p.add_argument("--metrics-report", default=str(DEFAULT_METRICS_REPORT))
    p.add_argument("--target", default=None, help="Restrict to one ML target.")

    r = sub.add_parser("recommend", help="Generate, predict and rank candidate designs.")
    r.add_argument("--scenario", required=True)
    r.add_argument("--count", type=int, default=10, help="Number of candidate designs.")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--objective", action="append", default=[], help="target:direction:weight")
    r.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    r.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    r.add_argument("--metrics-report", default=str(DEFAULT_METRICS_REPORT))
    r.add_argument("--primary-only", action="store_true", help="Load one model per target.")

    c = sub.add_parser("compare", help="Cross-check ML against the thermal engine.")
    c.add_argument("--scenario", required=True)
    c.add_argument("--design-id", required=True)
    c.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    c.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    c.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    c.add_argument("--metrics-report", default=str(DEFAULT_METRICS_REPORT))
    return parser


def _parse_objectives(raw: list[str]) -> tuple[RecommendationObjective, ...]:
    if not raw:
        return DEFAULT_OBJECTIVES
    objectives = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"objective must be target:direction:weight, got {item!r}")
        target, direction, weight = parts
        objectives.append(
            RecommendationObjective(target=target, direction=direction, weight=float(weight))
        )
    return tuple(objectives)


def cmd_predict(args: argparse.Namespace) -> int:
    bundle = load_predictors(args.models_dir, primary_only=True, metrics_report=args.metrics_report)
    outcome = predict_design(
        bundle,
        design_id=args.design_id,
        scenario_id=args.scenario,
        target=args.target,
        dataset_path=args.dataset,
        metadata_path=args.metadata,
    )
    print(_json(outcome.to_dict()))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from ..shelter.ml_dataset import generate_designs
    bundle = load_predictors(args.models_dir, primary_only=True, metrics_report=args.metrics_report)
    designs = generate_designs(count=args.count, seed=args.seed)
    outcomes = []
    for index, design in enumerate(designs, start=1):
        config = _design_to_config(design, index)
        outcomes.append(
            predict_design(bundle, config=config, scenario_id=args.scenario, metadata_path=args.metadata)
        )
    objectives = _parse_objectives(args.objective)
    ranked = rank_designs(
        [o.design_id for o in outcomes],
        [o.primary_predictions for o in outcomes],
        objectives,
    )
    print(_json({
        "scenario": args.scenario,
        "count": len(ranked),
        "objectives": [
            {"target": o.target, "direction": o.direction.value, "weight": o.weight}
            for o in objectives
        ],
        "ranking": [
            {
                "rank": rec.rank,
                "design_id": rec.design_id,
                "recommendation_score": rec.recommendation_score,
                "components": rec.components,
                "primary_predictions": rec.primary_predictions,
            }
            for rec in ranked
        ],
        "surrogate_model_disclaimer": (
            "ML models are surrogate models of the thermal engine over the "
            "represented design space and NASA weather scenarios."
        ),
    }))
    return 0


def _design_to_config(design: dict[str, Any], index: int):
    from ..shelter.ml_dataset import build_shelter_config
    name = str(design.get("design_id", f"candidate_{index:04d}"))
    return build_shelter_config(design, name=name)


def cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd
    from ..shelter.ml_dataset import build_shelter_config
    bundle = load_predictors(args.models_dir, primary_only=True, metrics_report=args.metrics_report)
    frame = pd.read_csv(args.dataset)
    row = frame[(frame["design_id"] == args.design_id)]
    if row.empty:
        raise SystemExit(f"design_id {args.design_id!r} not found in {args.dataset}")
    config = build_shelter_config(row.iloc[0].to_dict(), name=args.design_id)
    weather = load_scenario_weather(args.scenario)
    comparison = compare_prediction_with_physics(bundle, config, weather, metadata_path=args.metadata)
    print(_json(comparison))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.cmd == "predict":
            return cmd_predict(args)
        if args.cmd == "recommend":
            return cmd_recommend(args)
        if args.cmd == "compare":
            return cmd_compare(args)
    except SystemExit as exc:
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise SystemExit(f"unknown command {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
