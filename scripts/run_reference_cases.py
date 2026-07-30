"""Run and preserve compact scientific acceptance evidence for release."""
from __future__ import annotations
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import building_hvac_twin
from building_hvac_twin.model import TwinParameters, simulate
from building_hvac_twin.synthetic import generate
from building_hvac_twin.data import validate_dataset

OUT=Path('results/reference-runs'); OUT.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()

# Case 1: smallest meaningful 24 hour baseline.
data,_=generate(days=1,seed=42)
p0=TwinParameters()
s0=simulate(data,p0)
baseline={
 'rows':len(s0),'seed':42,'interval_hours':1.0,
 'energy_kwh':float(s0.predicted_energy_kwh.sum()),
 'heating_kwh_thermal':float(s0.heating_load_kw.sum()),
 'cooling_kwh_thermal':float(s0.cooling_load_kw.sum()),
 'mean_indoor_temp_c':float(s0.predicted_indoor_temp_c.mean()),
 'occupied_comfort_degree_hours':float(s0.comfort_violation_c.sum())}

# Case 2: physical response to a 25 percent envelope UA increase.
p1=TwinParameters(ua_kw_per_k=p0.ua_kw_per_k*1.25)
s1=simulate(data,p1)
response={
 'changed_parameter':'ua_kw_per_k','baseline_value_kw_per_k':p0.ua_kw_per_k,
 'changed_value_kw_per_k':p1.ua_kw_per_k,
 'baseline_energy_kwh':baseline['energy_kwh'],
 'changed_energy_kwh':float(s1.predicted_energy_kwh.sum()),
 'energy_change_percent':float(100*(s1.predicted_energy_kwh.sum()/baseline['energy_kwh']-1)),
 'acceptance':'PASS' if s1.predicted_energy_kwh.sum()>s0.predicted_energy_kwh.sum() else 'FAIL'}

# Case 3: explicit Euler convergence to the analytical free-decay 1R1C solution.
def decay_error(dt_h):
 hours=12; n=int(hours/dt_h)+1; t=pd.date_range('2025-02-01',periods=n,freq=pd.to_timedelta(dt_h,unit='h'))
 d=pd.DataFrame({'timestamp':t,'outdoor_temp_c':np.zeros(n),'solar_w_m2':np.zeros(n),'occupancy_count':np.zeros(n),'indoor_temp_c':np.full(n,20.0),'energy_kwh':np.zeros(n),'equipment_status':np.zeros(n),'heating_setpoint_c':np.zeros(n),'cooling_setpoint_c':np.full(n,40.0),'electricity_price_per_kwh':np.zeros(n)})
 p=TwinParameters(ua_kw_per_k=1.0,capacitance_kwh_per_k=10.0,internal_gain_kw_per_person=0,solar_gain_factor_kw_per_wm2=0,base_power_kw=0,fan_power_kw=0)
 sim=simulate(d,p,initial_temp_c=20.0); analytical=20*np.exp(-(np.arange(n)+1)*dt_h/10.0)
 return float(np.max(np.abs(sim.predicted_indoor_temp_c.to_numpy()-analytical)))
coarse=decay_error(1.0); fine=decay_error(0.25)
convergence={'analytical_limit':'Tin(t)=Tout+(Tin0-Tout)*exp(-UA*t/C)','ua_kw_per_k':1.0,'capacitance_kwh_per_k':10.0,'duration_hours':12,'coarse_dt_hours':1.0,'coarse_max_abs_error_c':coarse,'fine_dt_hours':0.25,'fine_max_abs_error_c':fine,'error_reduction_factor':coarse/fine,'acceptance':'PASS' if fine<coarse/2 else 'FAIL'}

# Case 4: intentional invalid input.
bad=data.drop(columns=['energy_kwh']); errors=validate_dataset(bad)
invalid={'case':'missing required energy_kwh column','errors':errors,'acceptance':'PASS' if errors and 'energy_kwh' in errors[0] else 'FAIL'}

s0.to_csv(OUT/'baseline_predictions.csv',index=False)
pd.DataFrame([baseline,response]).to_csv(OUT/'case_metrics.csv',index=False)
summary={'project':'building-energy-hvac-digital-twin','project_url':'https://github.com/vicena-labs/building-energy-hvac-digital-twin','scientific_status':'Level 0 executable synthetic reference model','simulator':'building_hvac_twin 1R1C explicit Euler','package_version':building_hvac_twin.__version__,'python':platform.python_version(),'baseline':baseline,'parameter_response':response,'analytical_convergence':convergence,'invalid_input_gate':invalid,'remote_compute':{'submitted':False,'reason':'No registered CFD workflow is scientifically applicable to this lumped single-zone 1R1C model. Local deterministic evidence is the correct route.'},'runtime_seconds':time.perf_counter()-started}
(OUT/'reference_run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ['baseline_predictions.csv','case_metrics.csv','reference_run_summary.json']:
 p=OUT/name; print(name,hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(summary,indent=2))
if any(x['acceptance']!='PASS' for x in [response,convergence,invalid]): raise SystemExit('Acceptance failure')
