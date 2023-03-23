import datetime
import streamlit as st
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
import utils
from datetime import timedelta
from scipy.optimize import minimize
import itertools

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
    st.markdown('# [Calculator](#calculator)')
    st.markdown('# [Dataset Statistics](#dataset-statistics)')
    st.write('##')

st.markdown("## Calculator")
st.markdown("""---""")

def success_msg():
    st.sidebar.success("File Uploaded")

upload_runs = st.sidebar.file_uploader("Upload your own Trace Runs with utilisation information",help="Must be in same format as scout data")
upload_power_curves = st.sidebar.file_uploader("Upload your own Power Curves",help="Must be in same format as teads data")
upload_grid = st.sidebar.file_uploader("Upload your own Grid Emission Factors",help="Must be in same format as cloudcarbonfootprint data")
#st.session_state['upload_file']=upload_runs

if upload_runs is not None:
    #st.session_state['upload_file']=upload_file
    try:
        inputdf = pd.read_csv(upload_runs)
        #compute_avg_cpu_util(inputdf)
        st.session_state['df'] = inputdf
        original_df = inputdf.copy()
        df = original_df
        with st.spinner("Training model"):
            util_reg,time_reg,predicted_cpu_time, encoded_features = utils.train_gbr(inputdf)
        success_msg()
    except Exception as e:
        st.sidebar.exception(f"Failed. Error {e}")
if upload_runs is None:
    df = pd.read_csv('../data/processed/scout/average_utils/averages.csv')
    st.session_state['df'] = df
    original_df = df.copy()
    with st.spinner("Training model"):
        util_reg,time_reg,predicted_cpu_time, encoded_features = utils.train_gbr(df)
chosen_file = 'Scout' if not upload_runs else upload_runs.name
st.session_state['chosen_file'] = chosen_file
st.sidebar.info(f"Using data from {chosen_file}")

if upload_power_curves is not None:
    try:
        inputdf = pd.read_csv(upload_power_curves)
        st.session_state['power_df'] = inputdf
        power_df=inputdf
        success_msg()
    except Exception as e:
        st.sidebar.exception(f'Failed. Error {e}')
if upload_power_curves is None:
    power_df = pd.read_csv("../data/processed/teads/instancelines.csv")
    st.session_state['power_df'] = power_df
chosen_power_file = 'Teads' if not upload_power_curves else upload_power_curves.name
st.sidebar.info(f"Using data from {chosen_power_file}")

if upload_grid is not None:
    try:
        inputdf = pd.read_csv(upload_grid)
        st.session_state['grid_df'] = inputdf
        grid_carbon = inputdf
        success_msg()
    except Exception as e:
        st.sidebar.exception(f'Failed. Error {e}')
if upload_grid is None:
    grid_df = pd.read_csv("../data/raw/grid_carbon.md", sep="|").tail(-1)
    grid_df.rename(columns=lambda x: x.strip(), inplace=True)
    grid_carbon=pd.DataFrame(data={"Region": grid_df["Region"].str.replace(' ', ''), "CO2": grid_df["CO2e (metric ton/kWh)"].astype(float)*1000000})
    st.session_state['grid_df'] = grid_carbon
chosen_grid_file = 'CloudCarbonFootprint' if not upload_grid else upload_grid.name
st.sidebar.info(f"Using data from {chosen_grid_file}")



def format_string(string: str) -> str:
    # Replace underscores and dashes with spaces and capitalise first letters
    string = string.replace("_", " ").replace("-", " ")
    formatted_string = string.title()
    return formatted_string

def get_features(df):
    # set up all feature objects
    features = list(df.columns)
    #print(features)
    # remove features we don't want user selection for 
    features_to_remove=['avgcpu','elapsed_time','name']
    for f in features_to_remove:
        features.remove(f)

    # fill feature input array with streamlit objects containing user selection 

    # take likely user-adjustable features aside to populate later
    user_features_to_remove=['node_count','instance_type']
    user_features=[]
    for uf in user_features_to_remove:
        user_features.append(uf)
        features.remove(uf)

    # initialise empty array the size of the features we want
    feature_input=[None]*(len(features))

    for ind,val in enumerate(features):
        if df[val].dtype == 'int64':
            # could add help thing showing range in data? 
            feature_input[ind] = st.number_input(label=format_string(features[ind]), min_value=4, step=1)
        elif df[val].dtype == 'float64':
            feature_input[ind] = st.number_input(label=format_string(features[ind]), min_value=0)
        elif df[val].dtype == 'object':
            possible_vals=df[val].unique()
            if len(possible_vals)<10:
                feature_input[ind] = st.radio(label=format_string(features[ind]),options=possible_vals,horizontal=True)
            else:
                feature_input[ind] = st.selectbox(label=format_string(features[ind]),options=possible_vals)

    grid_df = pd.read_csv("../data/raw/grid_carbon.md", sep="|").tail(-1)
    grid_df.rename(columns=lambda x: x.strip(), inplace=True)
    grid_carbon=pd.DataFrame(data={"Region": grid_df["Region"].str.replace(' ', ''), "CO2": grid_df["CO2e (metric ton/kWh)"].astype(float)*1000000})
    locations=grid_carbon["Region"]
    location=st.radio("Location",locations,horizontal=True)
    global CARBON_INTENSITY
    CARBON_INTENSITY = grid_carbon[grid_carbon['Region']==location]["CO2"].values[0]
    st.markdown("""---""")

    # messy - find another way of doign this without as much code duplication - use function with params?
    for ind,val in enumerate(user_features):
        if df[val].dtype == 'int64':
            # could add help thing showing range in data? 
            feature_input.append(st.number_input(label=format_string(user_features[ind]), min_value=2, step=1))
        elif df[val].dtype == 'float64':
            feature_input.append(st.number_input(label=format_string(user_features[ind]), min_value=0))
        elif df[val].dtype == 'object':
            possible_vals=df[val].unique()
            if len(possible_vals)<10:
                feature_input.append(st.radio(label=format_string(user_features[ind]),options=possible_vals,horizontal=True))
            else:
                feature_input.append(st.selectbox(label=format_string(user_features[ind]),options=possible_vals))
    feature_inputdf=pd.DataFrame(columns=features+user_features, data = [feature_input])
    return feature_inputdf


feature_inputdf = get_features(original_df)


# more overrides, options etc are possible but entail a technical overhead for not much use
with st.expander("Override preset data"):
    
    PUE = st.number_input("Power Usage Effectiveness of Data Center",min_value=1.0,value=1.135, help="Minumum value of 1")
    intensity_override = st.number_input("Grid Carbon Intensity (g/CO2/kWh)", value=0.0, min_value=0.0, help="Override applies if value not set to 0")
    if intensity_override >0:
        CARBON_INTENSITY = intensity_override




def get_power(power_df,e_util,e_time,instance_type,nodes):
    instance=instance_type.values[0]
    intercept = power_df.loc[power_df['instance_type']==instance,'intercept'].values[0]   
    gradient= power_df.loc[power_df['instance_type']==instance,'slope'].values[0]  
    hourpower_single= (gradient * e_util) + intercept
    hourpower=hourpower_single*nodes
    max_power=power_df.loc[power_df['instance_type']==instance,'max_power'].values[0]
    powerused=(hourpower/3600)*e_time
    return (powerused,max_power)

def get_carbon(powerused):
    power=powerused
    pue_power =power*PUE
    carbon=(pue_power*CARBON_INTENSITY)/1000
    return carbon

st.markdown("")
col1,col2 = st.columns([.2,1],gap='small')
with col1:
    run_prediction=st.button("Estimate Emissions")
with col2:
    recommendation=st.button("Recommend Optimal Scale-out")

def predict_with_trace(feature_inputdf, predicted_cpu_time):
    job_input = feature_inputdf['job_input'].values[0]
    engine_type = feature_inputdf['engine_type'].values[0]
    data_size = feature_inputdf['data_size'].values[0]
    node_count = feature_inputdf['node_count'].values[0]
    instance_type = feature_inputdf['instance_type'].values[0]

    # sometimes useful to view what trace datatset info we have
    #st.write(predicted_cpu_time)

    match = predicted_cpu_time.query("job_input == @job_input and engine_type == @engine_type and data_size == @data_size and node_count == @node_count and instance_type == @instance_type")
    if match.empty:
        return None
    return match

def show_gradboost_prediction(e_util, e_time, power_result, carbon_result):
    st.subheader("Estimate using Gradient Boosting")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resource Utilisation", f"{round(e_util[0],2)}%")
    col2.metric("Runtime Duration", utils.time_format(seconds=e_time[0]))
    col3.metric("Power Usage", f"{round(power_result[0],2)}w")
    col4.metric("Environmental Footprint", f"{round(carbon_result,2)} gCO2eq")

def show_match_prediction(match_util, match_time, match_power_result, match_carbon_result, e_util, e_time, power_result, carbon_result):
    st.subheader("Estimate using Trace Dataset Match")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resource Utilisation", f"{np.round(match_util)}%", delta=f"{round(((match_util-e_util[0])/e_util[0])*100,2)}%")
    col2.metric("Runtime Duration", utils.time_format(seconds=match_time), delta=f'{round(((match_time-e_time[0])/e_time[0])*100,2)}%')
    col3.metric("Power Usage", f"{round(match_power_result,2)}w", delta = f"{round(((match_power_result-power_result[0])/power_result[0])*100,2)}%")
    col4.metric("Environmental Footprint", f"{round(match_carbon_result,2)} gCO2eq", delta=f"{round(((match_carbon_result-carbon_result)/carbon_result)*100,2)}%")
    st.caption("Deltas show the percentage change of the metric compared to the initial estimate ")

if run_prediction:
    # put try block in here
    #st.write(feature_inputdf)
    nodes=feature_inputdf['node_count'].values[0]
    chosen_features = feature_inputdf.copy()
    e_util, e_time, instance_type = utils.predict_gbr(util_reg,time_reg,feature_inputdf, encoded_features.copy())
    power_result,max_power = get_power(power_df,e_util,e_time,instance_type,nodes)
    # call some functions to calculate result 
    carbon_result = get_carbon(power_result)[0]

    # use Teads data to make a DF with this info and look up from there
    #scope3val=get_scope_3(power_df)
    #scopeexp=f", including {scope3val}g/CO2/kWh scope 3 emissions." if scope3 else "."

        # show dataframe of both results    

    #comparison = predicted_cpu_time.compare(feature_inputdf)
    #print("comparison", comparison)
    #st.write(comparison)
    #match_df=predicted_cpu_time.where((predicted_cpu_time['job_input'].equals(feature_inputdf['job_input'])))
    #print("match df",match_df)
    show_gradboost_prediction(e_util, e_time, power_result, carbon_result)

    match = predict_with_trace(chosen_features, predicted_cpu_time)
    
    if match is not None:
        match_util, match_time = np.array(match['true_cpu'].values[0]), np.array(match['true_time'].values[0])
        match_power_result,_ = get_power(power_df,match_util,match_time,instance_type,nodes)
        match_carbon_result = get_carbon(match_power_result)
        show_match_prediction(match_util, match_time, match_power_result, match_carbon_result, e_util, e_time, power_result, carbon_result)
    #st.success(f"Carbon Footprint Estimate using Gradient Boosting is: {round(carbon_result,2)} gCO2eq.")
    #st.info(f"Calculation: {max_power} watt instance at {round(e_util[0])}% load for {round(e_time[0])} seconds, multiplied by {PUE} PUE and {CARBON_INTENSITY}g/CO2/kWh{scopeexp}")


if recommendation:
    selected_job=feature_inputdf['job_input'].values[0]
    selected_engine = feature_inputdf['engine_type'].values[0]
    selected_size=feature_inputdf['data_size'].values[0]
    # placeholder - need to perform optimisatio n based on models
    #predicted_cpu_time["instance_max_power"] = power_df.loc[power_df['instance_type']==predicted_cpu_time[],'max_power'].values[0]
    constrained_df=predicted_cpu_time[(predicted_cpu_time['job_input']==selected_job) & (predicted_cpu_time['engine_type']==selected_engine) & (predicted_cpu_time['data_size']==selected_size)]

    all_df = constrained_df.merge(power_df[['instance_type','max_power', 'slope', 'intercept']])
    
    all_df["estimated_gCO2eq"] = ((((all_df['slope']*all_df['true_cpu']+all_df['intercept'])/3600)*all_df['true_time']*all_df['node_count'])*PUE*CARBON_INTENSITY)/1000
    sorted_best=all_df.sort_values(by='estimated_gCO2eq')
    best_one = sorted_best.head(1)

    job_input = encoded_features["job_input"].transform([selected_job])
    engine_type = encoded_features["engine_type"].transform([selected_engine])
    data_size = encoded_features["data_size"].transform([selected_size])
    node_count = [4,8,16,32,64]
    cluster_type=[0,1,2,3,4,5,6,7,8]
    index = pd.MultiIndex.from_product([job_input, engine_type,data_size,node_count,cluster_type], names = ["job_input", "engine_type", "data_size", "node_count", "instance_type"])
    configs_df = pd.DataFrame(index = index).reset_index()
    e_util_configs = util_reg.predict(configs_df)
    e_time_configs = time_reg.predict(configs_df)
    for col in configs_df.columns:
        if col == 'node_count':
            configs_df[col] = configs_df[col]
        else:
            configs_df[col] = encoded_features[col].inverse_transform(configs_df[col])
    configs_power = configs_df.merge(power_df[['instance_type','max_power', 'slope', 'intercept']])
    configs_power["e_util"] = e_util_configs
    configs_power["e_time"] = e_time_configs
    configs_power["estimated_power"] = (((configs_power['slope']*configs_power['e_util']+configs_power['intercept'])/3600)*configs_power['e_time']*configs_power['node_count'])
    configs_power["estimated_gCO2eq"] = (configs_power["estimated_power"]*PUE*CARBON_INTENSITY)/1000
    sorted_e_best=configs_power.sort_values(by='estimated_gCO2eq')
    e_best_one=sorted_e_best.head(1)

    

    st.subheader("Optimal Scale-out using Gradient Boosting")
    e_col1, e_col2, _ = st.columns(3)
    e_col1.metric("Node Count", e_best_one["node_count"])
    e_col2.metric("Instance Type", e_best_one["instance_type"].values[0])
    #e_best_one
    if not best_one.empty:
        st.subheader("Optimal Scale-out using Trace Dataset Match")
        col1, col2, _ = st.columns(3)
        col1.metric("Node Count", best_one["node_count"])
        col2.metric("Instance Type", best_one["instance_type"].values[0])
        #util_values = {}
        #for combination in combinations:
        #    util_values.append((combination, util_reg.predict(np.array(combination).reshape(1, -1))))
        #time_values = []
        #for combination in combinations:
        #    time_values.append((combination, util_reg.predict(np.array(combination).reshape(1, -1))))

        #print(values)
        #best = [c for c,v in values if v == min([v for c,v in values])]
        #print("best: ", best)
    #hourpower= (gradient * e_util) + intercept
    #max_power=power_df.loc[power_df['instance_type']==instance,'max_power'].values[0]
    #print(f'hour power: {hourpower}')
    #powerused=(hourpower/3600)*e_time
    #rec_cluster='c4.large'  
    #improvement=50
    #st.success(f"Your recommended EC2 Cluster is: {rec_cluster}",icon="✔️")


st.markdown("""---""")
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

plot_avg_util_by_noof_nodes(original_df)

st.sidebar.write("\n ")

