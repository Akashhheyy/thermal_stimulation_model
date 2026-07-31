"""Measured HVAC-load benchmarking with chronological, leakage-resistant splits.

This validates hourly HVAC electricity prediction only. It does not validate the
1R1C indoor-temperature model because the source has no indoor measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class MeasuredSplit:
    train_end: str = "2016-06-30 23:00:00"
    calibration_end: str = "2016-12-31 23:00:00"
    test_start: str = "2017-01-01 00:00:00"

def _weather_frame(weather_dir):
    frames=[]
    for path in sorted(Path(weather_dir).glob("shanghai_weather_*.csv")):
        d=pd.read_csv(path,encoding="utf-8-sig").rename(columns={"Timestamp":"timestamp","Air temperatue":"outdoor_temp_c","Dew point temperature":"dewpoint_c","Relative humidity":"rh_percent","Atmospheric pressure":"pressure_raw","Wind speed":"wind_speed_raw"})
        frames.append(d)
    if not frames: raise FileNotFoundError(f"No weather CSV files in {weather_dir}")
    out=pd.concat(frames,ignore_index=True); out["timestamp"]=pd.to_datetime(out.timestamp,errors="raise")
    return out.drop_duplicates("timestamp").sort_values("timestamp")

def load_energy_detective(root,meter_type="W"):
    root=Path(root); rec=pd.read_csv(root/"ref_building"/"ref_building_records.csv"); info=pd.read_csv(root/"ref_building"/"ref_building_info.csv")
    rec=rec.loc[rec.Type.eq(meter_type),["Time","BuildingID","Record"]].copy(); rec.columns=["timestamp","building_id","hvac_kwh"]
    rec["timestamp"]=pd.to_datetime(rec.timestamp,errors="raise"); rec["hvac_kwh"]=pd.to_numeric(rec.hvac_kwh,errors="coerce")
    data=rec.merge(_weather_frame(root/"weather"),on="timestamp",how="inner",validate="many_to_one")
    data=data.merge(info.rename(columns={"BuildingID":"building_id","Area":"area_m2"}),on="building_id",how="left",validate="many_to_one")
    return data.sort_values(["building_id","timestamp"]).reset_index(drop=True),info

def quality_report(data):
    expected=pd.date_range(data.timestamp.min(),data.timestamp.max(),freq="h"); rows=[]
    for bid,g in data.groupby("building_id",sort=True):
        rows.append({"building_id":int(bid),"rows":int(len(g)),"coverage_percent":float(100*g.timestamp.nunique()/len(expected)),"duplicate_timestamps":int(g.timestamp.duplicated().sum()),"missing_hvac":int(g.hvac_kwh.isna().sum()),"negative_hvac":int((g.hvac_kwh<0).sum())})
    return {"time_start":str(data.timestamp.min()),"time_end":str(data.timestamp.max()),"expected_hours_per_building":int(len(expected)),"buildings":rows}

def _design(d):
    ts=d.timestamp; hour=ts.dt.hour.to_numpy(); dow=ts.dt.dayofweek.to_numpy(); month=ts.dt.month.to_numpy(); temp=d.outdoor_temp_c.to_numpy(float); rh=d.rh_percent.to_numpy(float)
    hours=np.eye(24,dtype=float)[hour][:,1:]; weekdays=np.eye(7,dtype=float)[dow][:,1:]; seasons=np.eye(12,dtype=float)[month-1][:,1:]
    heat=np.maximum(18-temp,0); cool=np.maximum(temp-22,0)
    return np.column_stack([np.ones(len(d)),d.weekly_lag_kwh.to_numpy(float),temp,heat,cool,rh,hours,weekdays,seasons,heat*(dow<5),cool*(dow<5)])

def _ridge_fit(x,y,alpha=1e-4):
    scale=np.std(x,axis=0); scale[scale==0]=1; scale[0]=1; xs=x/scale; penalty=np.eye(xs.shape[1])*alpha; penalty[0,0]=0
    return np.linalg.solve(xs.T@xs+penalty,xs.T@y)/scale

def _metrics(obs,pred):
    o=np.asarray(obs,float); p=np.asarray(pred,float); e=p-o; mean=max(abs(float(np.mean(o))),1e-9); ss=float(np.sum((o-o.mean())**2))
    return {"n":int(len(o)),"mae_kwh":float(np.mean(abs(e))),"rmse_kwh":float(np.sqrt(np.mean(e*e))),"cvrmse_percent":float(100*np.sqrt(np.mean(e*e))/mean),"nmbe_percent":float(100*np.mean(e)/mean),"r2":float(1-np.sum(e*e)/ss) if ss else float("nan")}

def validate_building(g,split=MeasuredSplit()):
    history=g[["timestamp","hvac_kwh"]].copy(); history["timestamp"]=pd.to_datetime(history["timestamp"])+pd.offsets.Hour(168); history=history.rename(columns={"hvac_kwh":"weekly_lag_kwh"})
    g=g.merge(history,on="timestamp",how="left")
    need=["hvac_kwh","outdoor_temp_c","rh_percent","weekly_lag_kwh"]
    train=g[g.timestamp<=split.train_end].dropna(subset=need); cal=g[(g.timestamp>split.train_end)&(g.timestamp<=split.calibration_end)].dropna(subset=need); test=g[g.timestamp>=split.test_start].dropna(subset=need)
    if min(len(train),len(cal),len(test))<100: raise ValueError("Each partition needs at least 100 valid observations")
    coef=_ridge_fit(_design(train),train.hvac_kwh.to_numpy()); cal_pred=np.maximum(_design(cal)@coef,0); test_pred=np.maximum(_design(test)@coef,0)
    q=float(np.quantile(np.abs(cal.hvac_kwh.to_numpy()-cal_pred),.95,method="higher")); out=test[["timestamp","building_id","hvac_kwh","outdoor_temp_c","rh_percent"]].copy(); out["predicted_hvac_kwh"]=test_pred; out["prediction_lower_95"]=np.maximum(test_pred-q,0); out["prediction_upper_95"]=test_pred+q
    coverage=float(np.mean((out.hvac_kwh>=out.prediction_lower_95)&(out.hvac_kwh<=out.prediction_upper_95)))
    out["weekly_persistence_kwh"]=test.weekly_lag_kwh.to_numpy(); vp=out.dropna(subset=["weekly_persistence_kwh"])
    return {"building_id":int(g.building_id.iloc[0]),"train_rows":int(len(train)),"calibration_rows":int(len(cal)),"test_rows":int(len(test)),"model":_metrics(out.hvac_kwh,out.predicted_hvac_kwh),"weekly_persistence":_metrics(vp.hvac_kwh,vp.weekly_persistence_kwh),"absolute_residual_95_kwh":q,"empirical_interval_coverage":coverage},out

def run_energy_detective_validation(root,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); data,_=load_energy_detective(root); quality=quality_report(data); summaries=[]; predictions=[]
    for _,g in data.groupby("building_id",sort=True): s,p=validate_building(g); summaries.append(s); predictions.append(p)
    table=pd.DataFrame([{"building_id":s["building_id"],**{f"model_{k}":v for k,v in s["model"].items()},**{f"persistence_{k}":v for k,v in s["weekly_persistence"].items()},"interval_coverage":s["empirical_interval_coverage"]} for s in summaries]); table.to_csv(output/"per_building_metrics.csv",index=False); pd.concat(predictions,ignore_index=True).to_csv(output/"test_predictions_2017.csv",index=False)
    result={"scientific_status":"Measured-data Level 1 validation for hourly HVAC electricity prediction only","prediction_task":"One-week-ahead HVAC electricity prediction using weather, calendar, and the value from 168 hours earlier","claim_boundary":"Does not validate indoor temperature, comfort, equipment physics, causal control savings, or new buildings without target history","dataset":{"name":"EnergyDetective 2020 dataset v2","doi":"10.5281/zenodo.6590976","raw_sha256":"ae43c6fb205fb9442ab7ac8da7156afdf91767e06ed4a3252a2fa94c662c2fde"},"split":MeasuredSplit().__dict__,"quality":quality,"aggregate":{"buildings":int(len(table)),"median_model_cvrmse_percent":float(table.model_cvrmse_percent.median()),"median_persistence_cvrmse_percent":float(table.persistence_cvrmse_percent.median()),"model_better_than_persistence_count":int((table.model_rmse_kwh<table.persistence_rmse_kwh).sum()),"median_interval_coverage":float(table.interval_coverage.median())},"per_building":summaries}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n"); return result
