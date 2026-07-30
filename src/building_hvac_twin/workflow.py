"""End-to-end reference workflow."""
from pathlib import Path
import pandas as pd
from .data import load_dataset,validate_dataset,temporal_split,write_json
from .model import TwinParameters
from .analysis import calibrate,evaluate,residual_diagnostics,sensitivity,optimize_setpoints

def run(dataset,output="outputs/reference_run",train_fraction=.7):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); d=load_dataset(dataset); errors=validate_dataset(d)
    if errors: raise ValueError("; ".join(errors))
    train,val=temporal_split(d,train_fraction); p,cal=calibrate(train,TwinParameters()); train_sim,train_metrics=evaluate(train,p); val_sim,val_metrics=evaluate(val,p)
    train_sim.to_csv(out/"train_predictions.csv",index=False); val_sim.to_csv(out/"validation_predictions.csv",index=False)
    sens=sensitivity(val,p); sens.to_csv(out/"sensitivity.csv",index=False); best,front=optimize_setpoints(val,p); front.to_csv(out/"optimization_candidates.csv",index=False)
    summary={"scientific_status":"Level 0 executable synthetic reference, not production validated","split":{"train_rows":len(train),"validation_rows":len(val),"train_fraction":train_fraction},"parameters":p.to_dict(),"calibration":cal,"train_metrics":train_metrics,"validation_metrics":val_metrics,"residual_diagnostics":residual_diagnostics(val,val_sim),"optimization":{"constraint_degree_hours":12.0,"best":best}}
    write_json(summary,out/"summary.json"); return summary
