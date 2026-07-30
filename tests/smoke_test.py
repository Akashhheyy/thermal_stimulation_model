from building_hvac_twin.synthetic import generate
from building_hvac_twin.model import TwinParameters,simulate
d,_=generate(1); s=simulate(d,TwinParameters()); assert len(s)==24
print("SMOKE PASS",round(s.predicted_energy_kwh.sum(),3),"kWh synthetic")
