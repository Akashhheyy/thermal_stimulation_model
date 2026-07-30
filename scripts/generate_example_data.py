from building_hvac_twin.cli import main
if __name__=="__main__": main(["generate-data","--output","datasets/example/building_timeseries.csv","--days","30","--seed","42"])
