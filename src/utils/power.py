import pandas as pd

power_curves = pd.read_csv("../processed/teads/instancelines.csv")
all_runs = pd.read_csv("../processed/scout/average_utils/averages.csv")
#print(power_curves.loc[["c4.2xlarge"]:])
#print(power_curves["cluster_type"])
power_curves = power_curves.set_index("cluster_type")
dfs=[]

for index, row in all_runs.iterrows():
    avgutil=row['avgcpu']
    instance=row['cluster_type']
    #print(row['cluster_type'],row['avgcpu'])
    hourpower=power_curves.loc[instance,'slope']* avgutil + power_curves.loc[instance,'intercept']
    powerused=(hourpower/3600)*row['elapsed_time']
    df = pd.DataFrame(data={'cluster_type' : row['name'],'power':powerused}, index=['cluster_type'])
    dfs.append(df)
all=pd.concat(dfs)
dfpath = Path(f"../processed/power.csv")
all.to_csv(dfpath, index=False)
