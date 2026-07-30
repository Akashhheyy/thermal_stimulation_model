from building_hvac_twin.synthetic import generate
from building_hvac_twin.data import validate_dataset,temporal_split
def test_valid_synthetic():
 d,_=generate(3); assert validate_dataset(d)==[]; a,b=temporal_split(d); assert len(a)+len(b)==len(d) and len(b)>0
def test_invalid_missing_column():
 d,_=generate(1); assert validate_dataset(d.drop(columns=["energy_kwh"]))
def test_invalid_setpoints():
 d,_=generate(1); d.loc[0,"heating_setpoint_c"]=30; assert validate_dataset(d)
