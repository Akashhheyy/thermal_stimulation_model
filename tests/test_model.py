import numpy as np
from building_hvac_twin.synthetic import generate
from building_hvac_twin.model import TwinParameters,simulate
def test_model_outputs_are_finite():
 d,_=generate(2); s=simulate(d,TwinParameters()); assert np.isfinite(s.select_dtypes("number")).all().all(); assert (s.predicted_energy_kwh>=0).all()
def test_disabled_system_has_no_hvac_load():
 d,_=generate(1); d.equipment_status=0; s=simulate(d,TwinParameters()); assert s.heating_load_kw.sum()==0 and s.cooling_load_kw.sum()==0
