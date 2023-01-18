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
    st.markdown('# [Methodology](#methodology)')
    st.write('##')

st.markdown("## Calculator")
st.markdown("""---""")

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
        original_df = inputdf.copy()
        with st.spinner("Training model"):
            util_reg,time_reg = utils.train_gbr(inputdf)
        success_msg()
    except Exception as e:
        st.sidebar.exception(f"Failed. Error {e}")
if upload_runs is None:
    df = pd.read_csv('../data/processed/scout/average_utils/averages.csv')
    st.session_state['df'] = df
    original_df = df.copy()
    with st.spinner("Training model"):
        util_reg,time_reg = utils.train_gbr(df)
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

def format_string(string: str) -> str:
    # Replace underscores and dashes with spaces and capitalise first letters
    string = string.replace("_", " ").replace("-", " ")
    formatted_string = string.title()
    return formatted_string


#df = st.session_state['df']

def get_features(df):
    # set up all feature objects
    features = list(df.columns)
    print(features)
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
            feature_input[ind] = st.number_input(label=format_string(features[ind]), min_value=2, step=1)
        elif df[val].dtype == 'float64':
            feature_input[ind] = st.number_input(label=format_string(features[ind]), min_value=0)
        elif df[val].dtype == 'object':
            possible_vals=df[val].unique()
            if len(possible_vals)<10:
                feature_input[ind] = st.radio(label=format_string(features[ind]),options=possible_vals,horizontal=True)
            else:
                feature_input[ind] = st.selectbox(label=format_string(features[ind]),options=possible_vals)
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


# show semantic split between user inputted calculator data and other presets
st.markdown("""---""")

locations=['United Kingdom','Germany','France']

location=st.radio("Location",locations,horizontal=True)

with st.expander("Override preset data"):
    # could allow more overrides (elapsed time, avg cpu util, power curve etc but would probably need to be on different page?)
    PUE = st.number_input("Power Usage Effectiveness of Data Center",min_value=1.0,value=1.135, help="Minumum value of 1")
    CARBON_INTENSITY = st.number_input("Grid Carbon Intensity (g/CO2/kWh)",value=228,min_value=0)
    AVG_UTIL = st.number_input("Use Constant CPU Utilisation Level",value=0.5,min_value=0.0)

scope3=st.checkbox("Include Scope 3 Emissions")


def predict_gbr(util_reg,time_reg,feature_inputdf):
    #save cluster type to use later
    instance_type = feature_inputdf["instance_type"]
    #encode feature labels
    cat_cols_f = feature_inputdf.select_dtypes(include='object').columns
    d_f = defaultdict(preprocessing.LabelEncoder)
    feature_inputdf[cat_cols_f] = feature_inputdf[cat_cols_f].apply(lambda x: d_f[x.name].fit_transform(x.astype(str)))

    # attempt to make df of resultant data
    e_util = util_reg.predict(feature_inputdf)
    e_time = time_reg.predict(feature_inputdf)
    return (e_util, e_time, instance_type)

def get_power(power_df,e_util,e_time,instance_type):
    #power_df.set_index("instance_type")
    instance=instance_type.values[0]
    hourpower=power_df.loc[power_df['instance_type']==instance,'slope']* e_util + power_df.loc[power_df['instance_type']==instance,'intercept']
    powerused=(hourpower/3600)*e_time
    return powerused

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

if run_prediction:
    try:
        with st.spinner("Computing Prediction"):
            e_util, e_time, instance_type = predict_gbr(util_reg,time_reg,feature_inputdf)
            
        # placeholder, will get from teads dataset 
        max_power = 100
        power_result = get_power(power_df,e_util,e_time,instance_type)
        # call some functions to calculate result 
        carbon_result = get_carbon(power_result).values[0]

        # use Teads data to make a DF with this info and look up from there
        scope3val=5
        scopeexp=f", including {scope3val}g/CO2/kWh scope 3 emissions." if scope3 else "."

        st.success(f"Carbon Footprint Result is: {round(carbon_result,2)} gCO2eq.")
        st.info(f"Calculation: {max_power} watt instance at {round(e_util[0])}% load for {round(e_time[0])} seconds, multiplied by {PUE} PUE and {CARBON_INTENSITY}g/CO2/kWh{scopeexp}")

    except Exception as e:
        st.exception(f"Failed. Error {e}")
if recommendation:
    # placeholder - need to perform optimisation based on models 
    rec_cluster='c4.large'  
    improvement=50
    st.success(f"Your recommended EC2 Cluster is: {rec_cluster}",icon="✔️")


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
st.markdown("""---""")
st.markdown("# Methodology")

st.sidebar.write("\n ")

