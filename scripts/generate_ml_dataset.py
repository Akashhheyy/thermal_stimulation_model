"""Generate the NASA POWER based ML dataset for the passive shelter model.

Pipeline (reuse only, no new thermal physics and no second NASA client):

    documented scenario dates
        -> existing NASA POWER client (live requests, raw payloads cached on disk)
        -> seeded shelter design space -> validated ShelterConfig objects
        -> existing simulate_shelter() engine
        -> existing design_metrics()
        -> data/shelter_ml_dataset.csv + data/shelter_ml_dataset_metadata.json

Run from the repository root:

    python scripts/generate_ml_dataset.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from building_hvac_twin.shelter.ml_dataset import (  # noqa: E402
    DATASET_COLUMNS,
    TARGET_COLUMNS,
    generate_ml_dataset,
    write_dataset,
    write_metadata,
)


def _print_report(quality: dict) -> None:
    print("=" * 78)
    print("SHELTER ML DATASET GENERATION REPORT")
    print("=" * 78)
    print(f"NASA scenarios requested:            {quality['nasa_scenarios_requested']}")
    print(f"NASA scenarios retrieved:            {quality['nasa_scenarios_retrieved']}")
    print(f"NASA hourly records retrieved:       {quality['nasa_hourly_records_total']}")
    print(f"Shelter designs generated:           {quality['designs_generated']}")
    print(f"Valid designs:                       {quality['designs_valid']}")
    print(f"Final dataset rows:                  {quality['final_row_count']}")
    print(f"Duplicate rows:                      {quality['duplicate_row_count']}")
    print(f"Dataset columns:                     {len(quality['columns'])}")
    print(f"Weather from LIVE NASA POWER:        {quality['weather_from_live_nasa_power']}")
    print()
    missing = quality["missing_value_counts"]
    if missing:
        print("Missing values (optional columns only):")
        for column, count in sorted(missing.items()):
            print(f"  {column}: {count}")
    else:
        print("Missing values: none in any column")
    print()
    print("Target statistics (min / mean / max):")
    for column, stats in quality["target_statistics"].items():
        print(
            f"  {column:<34}"
            f"{stats['min']:>10.3f}{stats['mean']:>10.3f}{stats['max']:>10.3f}"
        )
    print()
    print("Weather scenarios:")
    for scenario in quality["weather_scenario_summary"]:
        replaced = " (fallback date used)" if scenario["date_was_replaced"] else ""
        print(
            f"  {scenario['scenario_id']:<22} {scenario['effective_date']}"
            f"  records={scenario['records']:>3}"
            f"  meanT={scenario['mean_outdoor_temperature_c']:>7.2f} C"
            f"  solarSum={scenario['daily_solar_sum_wh_m2']:>8.1f} Wh/m2"
            f"  [{scenario['retrieval_status']}]{replaced}"
        )
    print()
    print(f"Provenance: {quality['provenance']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", type=int, default=300, help="number of unique designs")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed")
    parser.add_argument("--data-dir", default="data", help="output directory")
    parser.add_argument("--timeout", type=float, default=60.0, help="NASA request timeout seconds")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "nasa_weather_raw"
    print("=" * 78)
    print("SHELTER ML DATASET GENERATION (LIVE NASA POWER WEATHER)")
    print(f"Designs: {args.designs}  Seed: {args.seed}  Cache: {cache_dir}")
    print("=" * 78)

    result = generate_ml_dataset(
        design_count=args.designs,
        seed=args.seed,
        cache_dir=cache_dir,
        timeout_seconds=args.timeout,
    )

    csv_path = write_dataset(result.frame, data_dir / "shelter_ml_dataset.csv")
    metadata_path = write_metadata(
        result.metadata, data_dir / "shelter_ml_dataset_metadata.json"
    )
    result.quality_report["provenance"] = result.metadata["nasa_power"]["provenance_statement"]
    result.quality_report["output_paths"] = {
        "dataset_csv": str(csv_path),
        "metadata_json": str(metadata_path),
        "nasa_raw_cache_dir": str(cache_dir),
    }
    _print_report(result.quality_report)
    print()
    print("Output files:")
    print(f"  {csv_path}")
    print(f"  {metadata_path}")
    print(f"  {cache_dir} ({len(list(cache_dir.glob('*.json')))} cached NASA payloads)")
    if result.failed_scenarios:
        print()
        print("FAILED SCENARIOS (no synthetic substitution was made):")
        for failure in result.failed_scenarios:
            print(f"  {failure['scenario_id']}: {failure['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
