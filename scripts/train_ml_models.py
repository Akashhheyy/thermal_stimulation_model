"""Train and evaluate surrogate ML models for the passive shelter dataset.

Pipeline:

    data/shelter_ml_dataset.csv + metadata
        -> load_dataset (validates structure, provenance, exclusions)
        -> prepare_features (deterministic feature order)
        -> grouped_design_split (no design leakage between train/validation/test)
        -> train one pipeline per (target, model) on the training rows only
        -> evaluate every model on the validation and test rows
        -> save fitted pipelines and reproducible reports

Run from the repository root:

    .venv\\Scripts\\python.exe scripts\\train_ml_models.py --seed 42
    .venv\\Scripts\\python.exe scripts\\train_ml_models.py --seed 42 --target mean_indoor_temperature_c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from building_hvac_twin.ml import (  # noqa: E402
    FEATURE_COLUMNS,
    ML_TARGETS,
    XGBOOST_AVAILABLE,
    baseline_metrics,
    build_models,
    feature_importance,
    grouped_design_split,
    load_dataset,
    model_library,
    physical_sanity_report,
    prepare_features,
    regression_metrics,
    save_model,
    train_models,
)
from building_hvac_twin.ml.train import MODEL_FILE_PATTERN  # noqa: E402

DEFAULT_MODELS_DIR = Path("models") / "regression"
DEFAULT_REPORTS_DIR = Path("reports")


def _build_comparison_rows(fitted, features, target_series, train_index, valid_index, test_index):
    """Metrics rows for every (target, model) on validation and test split."""
    rows = []
    for (target, model), pipeline in sorted(fitted.items()):
        y_train = target_series[target].iloc[train_index].to_numpy(float)
        y_true_test = target_series[target].iloc[test_index].to_numpy(float)
        y_pred_test = pipeline.predict(features.iloc[test_index])
        metrics_test = regression_metrics(y_true_test, y_pred_test)
        y_true_valid = target_series[target].iloc[valid_index].to_numpy(float)
        y_pred_valid = pipeline.predict(features.iloc[valid_index])
        metrics_valid = regression_metrics(y_true_valid, y_pred_valid)
        baseline_row = baseline_metrics(y_train, y_true_test)
        rows.append(
            {
                "target": target,
                "model": model,
                "split": "test",
                "mae": metrics_test["mae"],
                "rmse": metrics_test["rmse"],
                "r2": metrics_test["r2"],
                "explained_variance": metrics_test["explained_variance"],
                "mape_percent": metrics_test.get("mape_percent"),
                "baseline_mae": baseline_row["mae"],
                "beats_baseline_mae": bool(metrics_test["mae"] < baseline_row["mae"]),
            }
        )
        rows.append(
            {
                "target": target,
                "model": model,
                "split": "validation",
                "mae": metrics_valid["mae"],
                "rmse": metrics_valid["rmse"],
                "r2": metrics_valid["r2"],
                "explained_variance": metrics_valid["explained_variance"],
                "mape_percent": metrics_valid.get("mape_percent"),
                "baseline_mae": None,
                "beats_baseline_mae": None,
            }
        )
    return rows


def _write_predictions(frame, fitted, features, test_index, reports_dir):
    """Actual-vs-predicted rows for every target and model on the test split."""
    import numpy as np

    rows = []
    for (target, model), pipeline in sorted(fitted.items()):
        y_true = frame.loc[test_index, target].to_numpy(float)
        y_pred = pipeline.predict(features.iloc[test_index])
        meta = frame.loc[test_index, ["design_id", "weather_scenario_id"]]
        if y_pred.ndim > 1:
            y_pred = np.asarray(y_pred).ravel()
        for (design, scenario), actual, predicted in zip(
            meta.itertuples(index=False, name=None), y_true, y_pred
        ):
            rows.append(
                {
                    "design_id": design,
                    "weather_scenario_id": scenario,
                    "target": target,
                    "model": model,
                    "actual": float(actual),
                    "predicted": float(predicted),
                }
            )
    pd.DataFrame(rows).to_csv(reports_dir / "ml_predictions_test.csv", index=False)


def _write_feature_importance(fitted, reports_dir):
    """Model-derived feature importance for the tree-based pipelines."""
    rows = []
    for (target, model), pipeline in sorted(fitted.items()):
        if model not in ("random_forest", "gradient_boosting"):
            continue
        for row in feature_importance(pipeline, model):
            rows.append({"target": target, "model": model, **row})
    pd.DataFrame(rows).to_csv(reports_dir / "feature_importance.csv", index=False)
    return rows


def _write_metrics_json(frame, fitted, features, test_index, reports_dir):
    """Per (target, model) test metrics plus physical sanity of raw forecasts."""
    import numpy as np

    document: dict = {}
    for (target, model), pipeline in sorted(fitted.items()):
        if target not in document:
            document[target] = {}
        y_true = frame.loc[test_index, target].to_numpy(float)
        y_pred = pipeline.predict(features.iloc[test_index])
        if y_pred.ndim > 1:
            y_pred = np.asarray(y_pred).ravel()
        metrics = regression_metrics(y_true, y_pred)
        document[target][model] = {
            "metrics": metrics,
            "raw_prediction_non_finite_count": int((~np.isfinite(y_pred)).sum()),
            "sanity": physical_sanity_report(target, y_pred),
        }
    (reports_dir / "ml_metrics.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


def _print_report(frame, split, comparison, best, targets) -> None:
    print("=" * 90)
    print("SHELTER ML SURROGATE MODEL TRAINING REPORT")
    print("=" * 90)
    print(
        f"Dataset rows: {len(frame)}  Unique designs: {frame['design_id'].nunique()}"
        f"  Weather scenarios: {frame['weather_scenario_id'].nunique()}"
    )
    print(f"Split (grouped by design_id, seed {split.summary['seed']}):")
    for part in split.summary["splits"]:
        print(
            f"  {part['split']:<11} rows={part['rows']:>5}  "
            f"designs={part['unique_designs']:>4}  scenarios={part['weather_scenarios']}"
        )
    print()
    print("Test-set metrics (RMSE, MAE, R2):")
    header = (
        f"{'target':<32}{'model':<20}{'rmse':>10}{'mae':>10}"
        f"{'r2':>8}{'beatsBase':>12}"
    )
    print(header)
    print("-" * len(header))
    test_rows = comparison[comparison["split"] == "test"].sort_values(["target", "rmse"])
    for row in test_rows.itertuples(index=False):
        beats = "yes" if bool(row.beats_baseline_mae) else "no"
        print(
            f"{row.target:<32}{row.model:<20}{row.rmse:>10.3f}{row.mae:>10.3f}"
            f"{row.r2:>8.3f}{beats:>12}"
        )
    print("-" * len(header))
    print("Best model per target (lowest test RMSE):")
    for row in best.itertuples(index=False):
        print(
            f"  {row.target:<32}{row.model:<20} rmse={row.rmse:.3f} "
            f"mae={row.mae:.3f} r2={row.r2:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default=None, help="train one target only")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--fast", action="store_true", help="small fast models (tests only)")
    args = parser.parse_args()

    seed = args.seed
    targets = [args.target] if args.target else list(ML_TARGETS)
    for target in targets:
        if target not in ML_TARGETS:
            parser.error(f"unsupported target {target!r}; supported: {list(ML_TARGETS)}")

    bundle = load_dataset()
    frame = bundle.frame
    features = prepare_features(frame)
    target_series = {target: frame[target].astype(float) for target in targets}

    split = grouped_design_split(frame, seed=seed)
    train_index, valid_index, test_index = (
        split.train_index,
        split.validation_index,
        split.test_index,
    )

    pipelines = build_models(seed=seed, fast=args.fast)
    model_names = tuple(pipelines)
    fitted = train_models(
        lambda: build_models(seed=seed, fast=args.fast),
        features,
        target_series,
        train_index,
        model_names=model_names,
    )

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for (target, model), pipeline in fitted.items():
        save_model(pipeline, models_dir, target, model)

    comparison_rows = _build_comparison_rows(
        fitted, features, target_series, train_index, valid_index, test_index
    )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["target", "model", "split"]
    ).reset_index(drop=True)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_dir / "ml_model_comparison.csv", index=False)

    test_rows = comparison[comparison["split"] == "test"]
    best = (
        test_rows.sort_values(["target", "rmse", "mae", "model"])
        .groupby("target", sort=True)
        .head(1)
        .reset_index(drop=True)
    )

    _write_feature_importance(fitted, reports_dir)
    _write_predictions(frame, fitted, features, test_index, reports_dir)
    _write_metrics_json(frame, fitted, features, test_index, reports_dir)

    metadata_doc = {
        "seed": int(seed),
        "dataset": {
            "path": str(Path("data") / "shelter_ml_dataset.csv"),
            "row_count": int(len(frame)),
            "unique_designs": int(frame["design_id"].nunique()),
            "weather_scenarios": int(frame["weather_scenario_id"].nunique()),
            "provenance": bundle.metadata.get("nasa_power", {}).get(
                "provenance_statement", ""
            ),
        },
        "features": {"columns": list(FEATURE_COLUMNS), "count": len(FEATURE_COLUMNS)},
        "targets": {"columns": targets},
        "models": model_library(seed=seed, fast=args.fast),
        "xgboost_available": XGBOOST_AVAILABLE,
        "split": split.summary,
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(valid_index)),
        "test_rows": int(len(test_index)),
        "artifacts": {
            "model_files": [
                str(models_dir / MODEL_FILE_PATTERN.format(target=target, model=model))
                for target, model in sorted(fitted)
            ],
            "comparison_csv": str(reports_dir / "ml_model_comparison.csv"),
            "feature_importance_csv": str(reports_dir / "feature_importance.csv"),
            "predictions_csv": str(reports_dir / "ml_predictions_test.csv"),
            "metrics_json": str(reports_dir / "ml_metrics.json"),
        },
    }
    (reports_dir / "training_metadata.json").write_text(
        json.dumps(metadata_doc, indent=2) + "\n", encoding="utf-8"
    )

    _print_report(frame, split, comparison, best, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())