"""Command line interface for the building HVAC twin."""
import argparse,json
from pathlib import Path
from .data import load_dataset,validate_dataset
from .synthetic import generate
from .workflow import run

def main(argv=None):
 p=argparse.ArgumentParser(prog="hvac-twin"); sub=p.add_subparsers(dest="cmd",required=True)
 g=sub.add_parser("generate-data"); g.add_argument("--output",default="datasets/example/building_timeseries.csv"); g.add_argument("--days",type=int,default=30); g.add_argument("--seed",type=int,default=42)
 v=sub.add_parser("validate-data"); v.add_argument("dataset")
 r=sub.add_parser("run"); r.add_argument("dataset"); r.add_argument("--output",default="outputs/reference_run"); r.add_argument("--train-fraction",type=float,default=.7)
 a=p.parse_args(argv)
 if a.cmd=="generate-data":
  d,truth=generate(a.days,a.seed); Path(a.output).parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.output,index=False); print(json.dumps({"rows":len(d),"output":a.output,"synthetic_truth":truth.to_dict()},indent=2))
 elif a.cmd=="validate-data":
  e=validate_dataset(load_dataset(a.dataset)); print(json.dumps({"valid":not e,"errors":e},indent=2)); raise SystemExit(1 if e else 0)
 else:
  print(json.dumps(run(a.dataset,a.output,a.train_fraction),indent=2))
if __name__=="__main__": main()
