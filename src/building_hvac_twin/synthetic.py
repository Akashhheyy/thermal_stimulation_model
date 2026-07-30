"""Generate deterministic synthetic building telemetry."""
import numpy as np
import pandas as pd
from .model import TwinParameters, simulate

def generate(days=30,seed=42):
    rng=np.random.default_rng(seed); n=days*24; t=pd.date_range("2025-01-01",periods=n,freq="h"); hour=t.hour.to_numpy(); day=np.arange(n)/24
    outdoor=9+7*np.sin(2*np.pi*(hour-8)/24)+2*np.sin(2*np.pi*day/7)+rng.normal(0,.5,n)
    solar=np.maximum(0,550*np.sin(np.pi*(hour-6)/12))+rng.normal(0,10,n); solar=np.maximum(solar,0)
    occupied=((t.dayofweek<5)&(hour>=8)&(hour<18)).astype(int); occ=occupied*rng.integers(5,20,n)
    hs=np.where(occupied,20.5,17.0); cs=np.where(occupied,24.5,28.0); status=occupied.astype(int)
    price=np.where((hour>=16)&(hour<21),.32,.15)
    d=pd.DataFrame({"timestamp":t,"outdoor_temp_c":outdoor,"solar_w_m2":solar,"occupancy_count":occ,"indoor_temp_c":20.0,"energy_kwh":0.0,"equipment_status":status,"heating_setpoint_c":hs,"cooling_setpoint_c":cs,"electricity_price_per_kwh":price})
    truth=TwinParameters(ua_kw_per_k=.48,capacitance_kwh_per_k=22,heating_cop=3.2,cooling_cop=3.6)
    s=simulate(d,truth,20.0); d["indoor_temp_c"]=s.predicted_indoor_temp_c+rng.normal(0,.18,n); d["energy_kwh"]=np.maximum(0,s.predicted_energy_kwh+rng.normal(0,.04,n)); return d,truth
