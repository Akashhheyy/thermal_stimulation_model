"""Transparent single-zone 1R1C building and HVAC reference model."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass
class TwinParameters:
    ua_kw_per_k: float = 0.42
    capacitance_kwh_per_k: float = 18.0
    internal_gain_kw_per_person: float = 0.10
    solar_gain_factor_kw_per_wm2: float = 0.0022
    heating_cop: float = 3.0
    cooling_cop: float = 3.4
    fan_power_kw: float = 0.45
    max_heating_kw: float = 18.0
    max_cooling_kw: float = 18.0
    deadband_c: float = 0.5
    base_power_kw: float = 0.30

    def to_dict(self): return asdict(self)

REQUIRED_COLUMNS = ["timestamp","outdoor_temp_c","solar_w_m2","occupancy_count","indoor_temp_c","energy_kwh","equipment_status","heating_setpoint_c","cooling_setpoint_c","electricity_price_per_kwh"]

def simulate(data: pd.DataFrame, params: TwinParameters, initial_temp_c=None) -> pd.DataFrame:
    d = data.copy().reset_index(drop=True)
    ts = pd.to_datetime(d.timestamp)
    dt = ts.diff().dt.total_seconds().div(3600).fillna(ts.diff().dt.total_seconds().div(3600).median())
    dt = dt.fillna(1.0).clip(lower=1/60).to_numpy()
    n=len(d); tin=np.zeros(n); heat=np.zeros(n); cool=np.zeros(n); power=np.zeros(n); energy=np.zeros(n); comfort=np.zeros(n)
    tin[0] = float(initial_temp_c if initial_temp_c is not None else d.indoor_temp_c.iloc[0])
    for i in range(n):
        if i>0: tin[i]=tin[i-1]
        gains=params.internal_gain_kw_per_person*float(d.occupancy_count.iloc[i])+params.solar_gain_factor_kw_per_wm2*float(d.solar_w_m2.iloc[i])
        free=params.ua_kw_per_k*(float(d.outdoor_temp_c.iloc[i])-tin[i])+gains
        predicted=tin[i]+dt[i]*free/params.capacitance_kwh_per_k
        hs=float(d.heating_setpoint_c.iloc[i]); cs=float(d.cooling_setpoint_c.iloc[i])
        enabled=float(d.equipment_status.iloc[i])>0
        if enabled and predicted < hs-params.deadband_c:
            heat[i]=min(params.max_heating_kw,(hs-predicted)*params.capacitance_kwh_per_k/dt[i])
        elif enabled and predicted > cs+params.deadband_c:
            cool[i]=min(params.max_cooling_kw,(predicted-cs)*params.capacitance_kwh_per_k/dt[i])
        tin[i]=tin[i]+dt[i]*(free+heat[i]-cool[i])/params.capacitance_kwh_per_k
        power[i]=params.base_power_kw+(params.fan_power_kw if enabled and (heat[i]+cool[i])>0 else 0)+heat[i]/params.heating_cop+cool[i]/params.cooling_cop
        energy[i]=power[i]*dt[i]
        comfort[i]=(max(hs-tin[i],0)+max(tin[i]-cs,0)) if float(d.occupancy_count.iloc[i])>0 else 0.0
    return pd.DataFrame({"timestamp":ts,"predicted_indoor_temp_c":tin,"heating_load_kw":heat,"cooling_load_kw":cool,"equipment_power_kw":power,"predicted_energy_kwh":energy,"comfort_violation_c":comfort,"cost":energy*d.electricity_price_per_kwh.to_numpy()})
