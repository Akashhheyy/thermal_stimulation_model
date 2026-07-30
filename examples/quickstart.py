from building_hvac_twin.data import load_dataset, temporal_split
from building_hvac_twin.analysis import calibrate, evaluate
from building_hvac_twin.model import TwinParameters
d=load_dataset("datasets/example/building_timeseries.csv"); train,val=temporal_split(d); p,_=calibrate(train,TwinParameters()); _,m=evaluate(val,p)
print("Held-out temperature RMSE, degC:",round(m["indoor_temperature"]["rmse"],3))
print("Held-out energy CVRMSE, percent:",round(m["energy"]["cvrmse_percent"],2))
