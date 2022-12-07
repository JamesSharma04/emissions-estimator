import streamlit as st
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt

plt.style.use('dark_background')

st.set_page_config(
    page_title="Cloud Carbon Calculator",
    page_icon="🖩",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': "http://localhost:8501",
        'Report a bug': "mailto:2469702s@student.gla.ac.uk",
        'About': "# Made By James Sharma for L4 Project"
    }
)

st.title("Carbon Emission Estimator for Distributed Data Jobs using Runtime Information")


# create sidebar with hyperlinks to page sections
with st.sidebar:
    st.markdown('# [Calculator](#section-1)')
    st.markdown('# [Dataset Statistics](#dataset-statistics)')
    st.markdown('# [Methodology](#methodology)')
    st.write('##')

st.markdown("# Calculator")

def success_msg():
    st.sidebar.success("File Uploaded")

upload_runs = st.sidebar.file_uploader("Upload your own runs with utilisation information",help="Must be in same format as scout data")
upload_power_curves = st.sidebar.file_uploader("Upload your own power curves",help="Must be in same format as teads data")

#st.session_state['upload_file']=upload_runs

if upload_runs is not None:
    #st.session_state['upload_file']=upload_file
    try:
        inputdf = pd.read_csv(upload_runs)
        #compute_avg_cpu_util(inputdf)
        st.session_state['df'] = inputdf
        success_msg()
    except Exception as e:
        st.sidebar.exception(f"Failed. Error {e}")
if upload_runs is None:
    df = pd.read_csv('../data/processed/scout/average_utils/averages.csv')
    st.session_state['df'] = df
chosen_file = 'Scout' if not upload_runs else upload_runs.name
st.session_state['chosen_file'] = chosen_file
st.sidebar.info(f"Using data from {chosen_file}")

if upload_power_curves is not None:
    try:
        inputdf = pd.read_csv(upload_power_curves)
        st.session_state['power_df'] = inputdf
        success_msg()
    except Exception as e:
        st.sidebar.exception(f'Failed. Error {e}')
if upload_power_curves is None:
    power_df = pd.read_csv("../data/processed/teads/instancelines.csv")
    st.session_state['power_df'] = power_df
chosen_power_file = 'Teads' if not upload_runs else upload_power_curves.name
st.sidebar.info(f"Using data from {chosen_power_file}")






locations=['United Kingdom','Germany','France']

df = st.session_state['df']


jobs=df["workload"].unique()
instancetypes=df["cluster_type"].unique()
nodecounts=df["node_count"].unique()
aaa = nodecounts.sort()

def node_interpolation(df,nodecounts,jobs):
    dfs = []
    x = nodecounts
    for j in jobs:
        filter_ = df['workload']==j
        y = df[filter_].groupby(['node_count'])['avgcpu'].mean(numeric_only=True)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        dfs.append(pd.DataFrame(data={'workload':[j],'slope':[slope],'intercept':[intercept],'error':[std_err],'r_value':[r_value],'p_value':[p_value],},index=[j]))
    all=pd.concat(dfs)
    return all 

def get_power(power_df,df):
    power_df.set_index("cluster_type")
    dfs = []
    for index, row in df.iterrows():
        avgutil=row['avgcpu']
        instance=row['cluster_type']
        #print(row['cluster_type'],row['avgcpu'])
        hourpower=power_df.loc[instance,'slope']* avgutil + power_df.loc[instance,'intercept']
        powerused=(hourpower/3600)*row['elapsed_time']
        df = pd.DataFrame(data={'cluster_type' : row['name'],'power':powerused}, index=['cluster_type'])
        dfs.append(df)
    all=pd.concat(dfs)
    return all 

def get_carbon(power_results):
    dfs=[]
    for index, row in power_results.iterrows():
        power=row['power']
        instance=row['cluster_type']
        pue_power =power*PUE
        carbon=(pue_power*CARBON_INTENSITY)/1000
        df = pd.DataFrame(data={'cluster_type' : instance,'carbon':carbon}, index=['cluster_type'])
        dfs.append(df)
    all=pd.concat(dfs)
    
all=node_interpolation(df,nodecounts,jobs)
st.write(all)

#possibly convert into dict 
nodes=st.slider("Node Count", min_value=2,max_value=100)
instance=st.radio("Instance Type",instancetypes,horizontal=True)
taskload=st.radio("Job",jobs,horizontal=True)
location=st.radio("Location",locations,horizontal=True)

with st.expander("Override preset data"):
    # could allow more overrides (elapsed time, avg cpu util, power curve etc but would probably need to be on different page?)
    PUE = st.number_input("Power Usage Effectiveness of Data Center",min_value=1.0,value=1.135, help="Minumum value of 1")
    CARBON_INTENSITY = st.number_input("Grid Carbon Intensity (g/CO2/kWh)",value=228,min_value=0)
    AVG_UTIL = st.number_input("Use Constant CPU Utilisation Level",value=0.5,min_value=0.0)
        # incl rest of data processing
        # try to interpolate between scaleouts assuming sparse dataset 

scope3=st.checkbox("Include Scope 3 Emissions")

run=st.button("Calculate")
recommendation=st.button("Show Recommendation")



if run:
    try:
        power = 1000
        power_result = get_power(power_df, df)
        # call some functions to calculate result 
        carbon_result = get_carbon(power_result)

        # compute by looking up carbon_result, using node interpolation/extrapolation if config not there
        carbon_total=100

        # store this somewhere 
        avg_load=0.8
        # use Teads data to make a DF with this info and look up from there
        scope3val=5
        scopeexp=f", including {scope3val}g/CO2/kWh scope 3 emissions." if scope3 else "."

        st.success(f"Carbon Footprint Result is: {carbon_total} gCO2eq.")
        st.info(f"Calculation: {power} watt device at {avg_load*100}% load, multiplied by {PUE} PUE and {CARBON_INTENSITY}g/CO2/kWh{scopeexp}")
        

    except Exception as e:
        st.exception(f"Failed. Error {e}")

if recommendation:
    rec_cluster='c4.large'  
    improvement=50
    st.success(f"Your recommended EC2 Cluster is: {rec_cluster}",icon="✔️")


st.markdown("# Dataset Statistics")
st.write(df.describe())

def plot_avg_util_by_noof_nodes(df):
    aaa=df.groupby('node_count')['avgcpu'].mean(numeric_only=True)
    fig, ax = plt.subplots()
    ax.plot(aaa)
    plt.title("Average CPU Utilisation by Number of Nodes")
    plt.xlabel("Node Count")
    plt.ylabel("Utilisation (%)")
    st.write(fig)

def plot_avg_util_by_workload(df):
    aaa = df.groupby('workload')['avgcpu'].mean(numeric_only=True)
    fig, ax = plt.subplots()
    #
    plt.title("Average CPU Utilisation by Number of Nodes")
    plt.xlabel("Node Count")
    plt.ylabel("Utilisation (%)")
    st.write(fig)

plot_avg_util_by_noof_nodes(df)

st.markdown("# Methodology")

st.sidebar.write("\n ")

