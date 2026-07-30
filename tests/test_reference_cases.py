import json, subprocess, sys
from pathlib import Path

def test_reference_acceptance_cases():
    run=subprocess.run([sys.executable,'scripts/run_reference_cases.py'],text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    s=json.loads(Path('results/reference-runs/reference_run_summary.json').read_text())
    assert s['parameter_response']['acceptance']=='PASS'
    assert s['parameter_response']['energy_change_percent']>0
    assert s['analytical_convergence']['acceptance']=='PASS'
    assert s['analytical_convergence']['fine_max_abs_error_c']<0.1
    assert s['invalid_input_gate']['acceptance']=='PASS'
