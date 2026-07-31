#!/usr/bin/env python3
import argparse
from pathlib import Path
from building_hvac_twin.measured_validation import run_energy_detective_validation
p=argparse.ArgumentParser(); p.add_argument("dataset_root"); p.add_argument("--output",default="outputs/measured_energy_detective"); a=p.parse_args(); s=run_energy_detective_validation(a.dataset_root,a.output)
print("Scientific status:",s["scientific_status"]); print("Buildings:",s["aggregate"]["buildings"]); print("Median model CVRMSE (%):",round(s["aggregate"]["median_model_cvrmse_percent"],3)); print("Median weekly-persistence CVRMSE (%):",round(s["aggregate"]["median_persistence_cvrmse_percent"],3)); print("Model better than persistence:",s["aggregate"]["model_better_than_persistence_count"],"/",s["aggregate"]["buildings"]); print("Median 95% interval coverage:",round(s["aggregate"]["median_interval_coverage"],3)); print("Output:",Path(a.output).resolve())
