"""Data validation, loading, and deterministic split helpers."""
import json
from pathlib import Path
import pandas as pd
from .model import REQUIRED_COLUMNS

def load_dataset(path):
    d=pd.read_csv(path); d["timestamp"]=pd.to_datetime(d["timestamp"]); return d.sort_values("timestamp").reset_index(drop=True)

def validate_dataset(d):
    errors=[]
    missing=[c for c in REQUIRED_COLUMNS if c not in d.columns]
    if missing: errors.append("Missing columns: "+", ".join(missing)); return errors
    if pd.to_datetime(d.timestamp,errors="coerce").isna().any(): errors.append("timestamp contains invalid values")
    if pd.to_datetime(d.timestamp).duplicated().any(): errors.append("timestamp must be unique")
    numeric=[c for c in REQUIRED_COLUMNS if c not in ("timestamp",)]
    for c in numeric:
        if pd.to_numeric(d[c],errors="coerce").isna().any(): errors.append(f"{c} must be numeric and nonmissing")
    if (d.occupancy_count<0).any(): errors.append("occupancy_count must be nonnegative")
    if (d.energy_kwh<0).any(): errors.append("energy_kwh must be nonnegative")
    if (d.heating_setpoint_c>d.cooling_setpoint_c).any(): errors.append("heating_setpoint_c must not exceed cooling_setpoint_c")
    return errors

def temporal_split(d, train_fraction=0.7):
    k=max(2,min(len(d)-1,int(len(d)*train_fraction))); return d.iloc[:k].copy(),d.iloc[k:].copy()

def write_json(obj,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2)+"\n")
