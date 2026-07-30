from building_hvac_twin.synthetic import generate
from building_hvac_twin.workflow import run
def test_end_to_end(tmp_path):
 d,_=generate(5); f=tmp_path/"d.csv"; d.to_csv(f,index=False); summary=run(f,tmp_path/"out"); assert summary["calibration"]["success"]; assert (tmp_path/"out/summary.json").exists(); assert summary["split"]["validation_rows"]>0
