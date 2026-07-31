import numpy as np
import pandas as pd
from building_hvac_twin.measured_validation import MeasuredSplit,quality_report,validate_building

def fixture():
    t=pd.date_range("2015-01-01","2017-12-31 23:00",freq="h"); temp=15+12*np.sin(2*np.pi*np.arange(len(t))/(24*365.25)); y=20+2*np.maximum(18-temp,0)+3*np.maximum(temp-22,0)+5*(t.dayofweek<5)
    return pd.DataFrame({"timestamp":t,"building_id":1,"hvac_kwh":y,"outdoor_temp_c":temp,"rh_percent":55.0})

def test_chronological_measured_validation():
    s,p=validate_building(fixture(),MeasuredSplit()); assert s["train_rows"]>1000 and s["test_rows"]==8760; assert s["model"]["cvrmse_percent"]<1; assert 0<=s["empirical_interval_coverage"]<=1; assert p.timestamp.min()==pd.Timestamp("2017-01-01")

def test_quality_negative():
    d=fixture(); d.loc[0,"hvac_kwh"]=-1; assert quality_report(d)["buildings"][0]["negative_hvac"]==1
