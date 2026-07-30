"""Calibration, diagnostics, uncertainty, sensitivity, and optimization."""
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .model import TwinParameters, simulate

def metrics(obs,pred):
    o=np.asarray(obs); p=np.asarray(pred); e=p-o
    return {"n":int(len(o)),"mae":float(np.mean(abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(np.mean(e)),"cvrmse_percent":float(100*np.sqrt(np.mean(e*e))/max(abs(np.mean(o)),1e-9)),"nmbe_percent":float(100*np.mean(e)/max(abs(np.mean(o)),1e-9))}

def calibrate(train, initial=None):
    p=initial or TwinParameters(); names=["ua_kw_per_k","capacitance_kwh_per_k","internal_gain_kw_per_person","solar_gain_factor_kw_per_wm2","heating_cop","cooling_cop"]
    x0=np.array([getattr(p,n) for n in names]); lo=[.05,2,.01,.0001,1.2,1.2]; hi=[3,150,.3,.01,6,8]
    def resid(x):
        q=replace(p,**dict(zip(names,x))); s=simulate(train,q)
        return np.r_[s.predicted_indoor_temp_c.to_numpy()-train.indoor_temp_c.to_numpy(),2*(s.predicted_energy_kwh.to_numpy()-train.energy_kwh.to_numpy())]
    r=least_squares(resid,x0,bounds=(lo,hi),max_nfev=250)
    q=replace(p,**dict(zip(names,r.x)))
    jac=r.jac; dof=max(1,len(r.fun)-len(r.x)); sigma2=float(np.sum(r.fun**2)/dof); cov=np.linalg.pinv(jac.T@jac)*sigma2; se=np.sqrt(np.maximum(np.diag(cov),0))
    uncertainty={n:{"estimate":float(v),"standard_error":float(e),"approx_95_percent":[float(v-1.96*e),float(v+1.96*e)]} for n,v,e in zip(names,r.x,se)}
    return q,{"success":bool(r.success),"cost":float(r.cost),"message":r.message,"parameter_uncertainty":uncertainty}

def evaluate(data,p):
    s=simulate(data,p); return s,{"indoor_temperature":metrics(data.indoor_temp_c,s.predicted_indoor_temp_c),"energy":metrics(data.energy_kwh,s.predicted_energy_kwh)}

def residual_diagnostics(data,sim):
    out={}
    for name,obs,pred in [("temperature",data.indoor_temp_c,sim.predicted_indoor_temp_c),("energy",data.energy_kwh,sim.predicted_energy_kwh)]:
        r=np.asarray(pred)-np.asarray(obs); out[name]={"mean":float(r.mean()),"std":float(r.std(ddof=1)),"p05":float(np.quantile(r,.05)),"p95":float(np.quantile(r,.95)),"lag1_autocorrelation":float(pd.Series(r).autocorr(1))}
    return out

def sensitivity(data,p,fraction=.1):
    base=simulate(data,p); base_energy=base.predicted_energy_kwh.sum(); rows=[]
    for name in p.to_dict():
        v=getattr(p,name)
        if not isinstance(v,(int,float)) or v==0: continue
        q=replace(p,**{name:v*(1+fraction)}); s=simulate(data,q); rows.append({"parameter":name,"fractional_change":fraction,"energy_change_percent":100*(s.predicted_energy_kwh.sum()-base_energy)/max(base_energy,1e-9),"comfort_change_degree_hours":s.comfort_violation_c.sum()-base.comfort_violation_c.sum()})
    return pd.DataFrame(rows).sort_values("energy_change_percent",key=abs,ascending=False)

def optimize_setpoints(data,p,comfort_limit_degree_hours=12.0):
    candidates=[]
    for hs in np.arange(18,22.1,1):
      for cs in np.arange(23,28.1,1):
        if hs>cs-1: continue
        d=data.copy(); d.heating_setpoint_c=hs; d.cooling_setpoint_c=cs; s=simulate(d,p)
        candidates.append({"heating_setpoint_c":hs,"cooling_setpoint_c":cs,"energy_kwh":s.predicted_energy_kwh.sum(),"cost":s.cost.sum(),"comfort_degree_hours":s.comfort_violation_c.sum()})
    table=pd.DataFrame(candidates); feasible=table[table.comfort_degree_hours<=comfort_limit_degree_hours]
    best=(feasible if len(feasible) else table).sort_values(["cost","comfort_degree_hours"]).iloc[0].to_dict(); best["constraint_feasible"]=bool(len(feasible))
    return best,table
