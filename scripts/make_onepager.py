"""Regenerate the Vicena Research Twins A4 landscape PDF and PNG."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

project=Path(__file__).resolve().parents[1]; assets=project/'assets'; docs_assets=project/'docs/assets'
summary=json.loads((project/'results/reference-runs/reference_run_summary.json').read_text())
workflow=json.loads((project/'outputs/reference_run/summary.json').read_text())
pred=pd.read_csv(project/'outputs/reference_run/validation_predictions.csv')
GOLD='#F8C73A'; DARK='#10151C'; MID='#394452'; LIGHT='#F4F6F8'; BLUE='#3F7CAC'; GREEN='#2A9D8F'; RED='#E76F51'
fig=plt.figure(figsize=(11.69,8.27),facecolor='white'); ax=fig.add_axes([0,0,1,1]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.add_patch(FancyBboxPatch((0,.83),1,.17,boxstyle='square,pad=0',facecolor=DARK,edgecolor='none'))
logo=Image.open(assets/'vicena-logo.png'); la=fig.add_axes([.035,.852,.09,.12]); la.imshow(logo); la.axis('off')
ax.text(.135,.935,'BUILDING ENERGY + HVAC DIGITAL TWIN',fontsize=21.5,fontweight='bold',color='white',va='center')
ax.text(.137,.885,'Vendor-neutral thermal simulation, calibration, validation and control optimization, built with Vicena',fontsize=10.2,color='#D9E0E8',va='center')
ax.text(.965,.93,'OPEN SOURCE',fontsize=9,fontweight='bold',ha='right',color=GOLD); ax.text(.965,.887,'MIT  •  Level 0 synthetic reference',fontsize=8.5,ha='right',color='white')
ax.text(.04,.785,'Turn building telemetry into an inspectable, calibratable HVAC research workflow.',fontsize=15.2,fontweight='bold',color=DARK)
ax.text(.04,.748,'Validate time series, fit transparent physics, test held-out periods, quantify uncertainty, then optimize cost against comfort constraints.',fontsize=9.7,color=MID)
cards=[('THERMAL BASELINE','1R1C envelope • gains • zone temperature\nHeating + cooling loads'),('ENERGY + COMFORT','COP • fan • base power • tariffs\nOccupied comfort degree-hours'),('CALIBRATION + VALIDATION','Chronological split • residuals\nLocal uncertainty • analytical check'),('AGENT ADAPTABLE','Schemas • units • playbooks • tests\nVendor-neutral project structure')]
for (title,body),x in zip(cards,[.04,.275,.51,.745]):
 ax.add_patch(FancyBboxPatch((x,.615),.215,.105,boxstyle='round,pad=.009,rounding_size=.012',facecolor=LIGHT,edgecolor='#DCE2E8')); ax.add_patch(FancyBboxPatch((x,.700),.215,.020,boxstyle='round,pad=.001,rounding_size=.010',facecolor=GOLD,edgecolor='none')); ax.text(x+.012,.674,title,fontsize=8.5,fontweight='bold',color=DARK); ax.text(x+.012,.635,body,fontsize=8.0,color=MID,linespacing=1.35)
# Evidence 1: held-out temperature
ap=fig.add_axes([.055,.305,.31,.255]); x=pd.to_datetime(pred.timestamp).iloc[:120]; ap.plot(x,pred.predicted_indoor_temp_c.iloc[:120],color=BLUE,lw=2,label='Predicted'); ap.set_title('Held-out zone-temperature prediction',loc='left',fontsize=10,fontweight='bold',color=DARK,pad=9); ap.set_ylabel('Indoor temperature [degC]',fontsize=8); ap.tick_params(labelsize=7,axis='x',rotation=25); ap.grid(alpha=.22); ap.legend(frameon=False,fontsize=7)
for sp in ['top','right']: ap.spines[sp].set_visible(False)
# Evidence 2: parameter response and analytical convergence
ar=fig.add_axes([.41,.305,.31,.255]); labels=['Baseline energy','UA +25% energy']; vals=[summary['parameter_response']['baseline_energy_kwh'],summary['parameter_response']['changed_energy_kwh']]; ar.bar(np.arange(2),vals,color=[BLUE,RED],width=.55); ar.set_xticks(np.arange(2),labels,fontsize=7); ar.set_ylabel('Electricity [kWh/day]',fontsize=8); ar.set_title('Physical response and acceptance evidence',loc='left',fontsize=10,fontweight='bold',color=DARK,pad=9); ar.grid(axis='y',alpha=.22)
for i,v in enumerate(vals): ar.text(i,v+.5,f'{v:.2f}',ha='center',fontsize=7)
ar.text(.02,.92,f"Fine-step analytical max error: {summary['analytical_convergence']['fine_max_abs_error_c']:.3f} degC",transform=ar.transAxes,fontsize=7.2,color=GREEN)
for sp in ['top','right']: ar.spines[sp].set_visible(False)
# Metrics
ax.add_patch(FancyBboxPatch((.755,.305),.205,.255,boxstyle='round,pad=.012,rounding_size=.012',facecolor=DARK,edgecolor='none'))
metrics=[(f"{summary['baseline']['energy_kwh']:.2f} kWh",'24-hour synthetic baseline'),(f"+{summary['parameter_response']['energy_change_percent']:.1f}%",'Energy response to UA +25%'),(f"{summary['analytical_convergence']['fine_max_abs_error_c']:.3f} degC",'Fine-step analytical max error'),(f"{workflow['validation_metrics']['indoor_temperature']['rmse']:.3f} degC",'Synthetic held-out temperature RMSE')]
y=.522
for value,label in metrics: ax.text(.775,y,value,fontsize=12.5,fontweight='bold',color=GOLD); ax.text(.775,y-.026,label,fontsize=7.2,color='white'); y-=.057
ax.text(.04,.245,'MODEL LADDER: SYNTHETIC REFERENCE TO MEASURED, HELD-OUT R&D TWIN',fontsize=10,fontweight='bold',color=DARK)
steps=[('UPLOAD','weather + telemetry'),('VALIDATE','schema + units'),('CALIBRATE','training period'),('VALIDATE','held-out period'),('OPTIMIZE','cost + comfort')]
centers=np.linspace(.105,.895,5)
for i,((title,sub),cx) in enumerate(zip(steps,centers)):
 ax.add_patch(FancyBboxPatch((cx-.075,.135),.15,.075,boxstyle='round,pad=.007,rounding_size=.012',facecolor=LIGHT,edgecolor='#D7DDE3')); ax.text(cx,.180,title,ha='center',fontsize=8.2,fontweight='bold',color=DARK); ax.text(cx,.151,sub,ha='center',fontsize=7,color=MID)
 if i<4: ax.add_patch(FancyArrowPatch((cx+.078,.172),(centers[i+1]-.078,.172),arrowstyle='-|>',mutation_scale=11,color=GOLD,lw=1.8))
ax.add_patch(FancyBboxPatch((0,0),1,.075,boxstyle='square,pad=0',facecolor=LIGHT,edgecolor='none')); ax.text(.04,.042,'github.com/vicena-labs/building-energy-hvac-digital-twin',fontsize=9.4,fontweight='bold',color=DARK,va='center'); ax.text(.96,.042,'vicena.ai  •  Synthetic examples, validate with measured evidence before operational use',fontsize=8.2,color=MID,ha='right',va='center')
pdf=project/'Building_Energy_HVAC_Digital_Twin_OnePager.pdf'; png=assets/'building-energy-hvac-digital-twin-onepager.png'; fig.savefig(pdf,format='pdf',bbox_inches='tight',pad_inches=0); fig.savefig(png,dpi=160,bbox_inches='tight',pad_inches=0); docs_assets.mkdir(parents=True,exist_ok=True); shutil.copy2(png,docs_assets/png.name)
print('24-hour baseline energy, kWh:',summary['baseline']['energy_kwh']); print('UA response, percent:',summary['parameter_response']['energy_change_percent']); print('fine analytical error, degC:',summary['analytical_convergence']['fine_max_abs_error_c']); print('held-out synthetic temperature RMSE, degC:',workflow['validation_metrics']['indoor_temperature']['rmse']); print('Saved:',pdf); print('Saved:',png)
