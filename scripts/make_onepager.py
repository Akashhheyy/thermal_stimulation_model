"""Generate the A4 landscape one-page overview from verified synthetic outputs."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
s=json.load(open("outputs/reference_run/summary.json")); pred=pd.read_csv("outputs/reference_run/validation_predictions.csv")
fig=plt.figure(figsize=(11.69,8.27),facecolor="#f4f7fb"); gs=fig.add_gridspec(4,4,height_ratios=[.6,1,2.2,.8],hspace=.48,wspace=.35)
ax=fig.add_subplot(gs[0,:]); ax.set_facecolor("#10243e"); ax.text(.03,.64,"BUILDING ENERGY + HVAC DIGITAL TWIN",color="white",fontsize=20,fontweight="bold"); ax.text(.03,.18,"Vendor-neutral, calibratable open-source R&D reference model  |  MIT  |  vicena.ai",color="#bfe2ff",fontsize=10); ax.set_xticks([]); ax.set_yticks([])
cards=[("Transparent physics","Single-zone 1R1C heat balance"),("Calibration","Bounded least squares with uncertainty"),("Validation","Temporal held-out metrics and residuals"),("Optimization","Cost and comfort setpoint search")]
for i,(a,b) in enumerate(cards):
 x=fig.add_subplot(gs[1,i]); x.set_facecolor("white"); x.text(.06,.68,a,fontweight="bold",fontsize=11,color="#10243e"); x.text(.06,.28,b,fontsize=8,wrap=True); x.set_xticks([]); x.set_yticks([])
a=fig.add_subplot(gs[2,:2]); a.plot(pd.to_datetime(pred.timestamp).iloc[:120],pred.predicted_indoor_temp_c.iloc[:120],color="#157f9a"); a.set_title("Held-out predicted indoor temperature, synthetic example"); a.set_ylabel("degC"); a.tick_params(axis='x',rotation=25,labelsize=7); a.grid(alpha=.2)
b=fig.add_subplot(gs[2,2:]); b.scatter(pred.comfort_violation_c,pred.predicted_energy_kwh,s=12,alpha=.55,color="#e47d31"); b.set_title("Energy and comfort outcomes, synthetic example"); b.set_xlabel("comfort violation, degree C per interval"); b.set_ylabel("energy, kWh"); b.grid(alpha=.2)
m=fig.add_subplot(gs[3,:]); m.set_facecolor("#10243e"); vm=s["validation_metrics"]; values=[f"{s['split']['train_rows']} calibration rows",f"{s['split']['validation_rows']} held-out rows",f"{vm['indoor_temperature']['rmse']:.3f} degC temperature RMSE",f"{vm['energy']['cvrmse_percent']:.1f}% energy CVRMSE"]
m.text(.02,.70,"VERIFIED SYNTHETIC REFERENCE METRICS",color="white",fontweight="bold",fontsize=10); m.text(.02,.30,"   |   ".join(values),color="#bfe2ff",fontsize=9); m.set_xticks([]); m.set_yticks([])
fig.suptitle("Workflow: validate data  >  split  >  calibrate  >  held-out validation  >  sensitivity and constrained optimization",y=.02,fontsize=9)
Path("assets").mkdir(exist_ok=True); fig.savefig("assets/building-energy-hvac-digital-twin-onepager.png",dpi=170,bbox_inches="tight"); fig.savefig("Building_Energy_HVAC_Digital_Twin_OnePager.pdf",bbox_inches="tight"); print("One-page metrics:",*values,sep="\n")
