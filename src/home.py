import streamlit as st
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn import preprocessing

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

def format_string(string: str) -> str:
    # Replace underscores and dashes with spaces and capitalise first letters
    string = string.replace("_", " ").replace("-", " ")
    formatted_string = string.title()
    return formatted_string




locations=['United Kingdom','Germany','France']

df = st.session_state['df']

# set up calculator objects
features = list(df.columns)
#  might it be useful to sometimes estimate emissions based on these? could make user configurable?
features_to_remove=['avgcpu','elapsed_time','name']
for f in features_to_remove:
    features.remove(f)

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
feature_inputdf=pd.DataFrame(columns=features, data = [feature_input])

st.text("")
location=st.radio("Location",locations,horizontal=True)

with st.expander("Override preset data"):
    # could allow more overrides (elapsed time, avg cpu util, power curve etc but would probably need to be on different page?)
    PUE = st.number_input("Power Usage Effectiveness of Data Center",min_value=1.0,value=1.135, help="Minumum value of 1")
    CARBON_INTENSITY = st.number_input("Grid Carbon Intensity (g/CO2/kWh)",value=228,min_value=0)
    AVG_UTIL = st.number_input("Use Constant CPU Utilisation Level",value=0.5,min_value=0.0)
        # incl rest of data processing
        # try to interpolate between scaleouts assuming sparse dataset 

scope3=st.checkbox("Include Scope 3 Emissions")

def train_gbr(df,feature_inputdf):
    #save cluster type to use later
    cluster_type = feature_inputdf["cluster_type"]
    #encode feature labels
    cat_cols_f = feature_inputdf.select_dtypes(include='object').columns
    d_f = defaultdict(preprocessing.LabelEncoder)
    feature_inputdf[cat_cols_f] = feature_inputdf[cat_cols_f].apply(lambda x: d_f[x.name].fit_transform(x.astype(str)))
    print(feature_inputdf)
    # labels
    y1 = df.pop('avgcpu')
    y2 = df.pop('elapsed_time')
    # drop runtime/util if not already - not sure if this is correct?
    #df.drop(columns=['avgcpu','elapsed_time'], inplace=True, errors='ignore')
    # feature vector
    name = df.pop("name")
    X = df
    unencoded_X=X.copy()
    # get columns containing text, apply label encoder and transform text to numbers
    cat_cols = X.select_dtypes(include='object').columns
    d = defaultdict(preprocessing.LabelEncoder)
    X[cat_cols] = X[cat_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))

    # split dataset
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0)

    # set regression model parameters to tweak later and see how results change
    params = {
        "n_estimators":500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }

    reg1 = GradientBoostingRegressor(**params)
    reg2 = GradientBoostingRegressor(**params)
    reg1.fit(X, y1)
    reg2.fit(X, y2)

    # attempt to make df of resultant data
    y1_pred = reg1.predict(feature_inputdf)
    y2_pred = reg2.predict(feature_inputdf)
    return (y1_pred,y2_pred,cluster_type)

def get_power(power_df,e_util,e_time,cluster_type):
    #power_df.set_index("cluster_type")
    instance=cluster_type.values[0]
    hourpower=power_df.loc[power_df['cluster_type']==instance,'slope']* e_util + power_df.loc[power_df['cluster_type']==instance,'intercept']
    powerused=(hourpower/3600)*e_time
    return powerused

def get_carbon(powerused):
    power=powerused
    pue_power =power*PUE
    carbon=(pue_power*CARBON_INTENSITY)/1000
    return carbon

run_prediction=st.button("Estimate Emissions")

if run_prediction:
    try:
        e_util, e_time, cluster_type = train_gbr(df,feature_inputdf)
        # placeholder
        max_power = 100
        power_result = get_power(power_df,e_util,e_time,cluster_type)
        # call some functions to calculate result 
        carbon_result = get_carbon(power_result).values[0]

        # use Teads data to make a DF with this info and look up from there
        scope3val=5
        scopeexp=f", including {scope3val}g/CO2/kWh scope 3 emissions." if scope3 else "."

        st.success(f"Carbon Footprint Result is: {round(carbon_result,2)} gCO2eq.")
        st.info(f"Calculation: {max_power} watt instance at {round(e_util[0])}% load for {round(e_time[0])} seconds, multiplied by {PUE} PUE and {CARBON_INTENSITY}g/CO2/kWh{scopeexp}")

    except Exception as e:
        st.exception(f"Failed. Error {e}")


jobs=df["workload"].unique()
instancetypes=df["cluster_type"].unique()
nodecounts=df["node_count"].unique()
aaa = nodecounts.sort()

def scale_out_fit_linregress(df,nodecounts,jobs):
    dfs = []
    x = nodecounts
    for j in jobs:
        filter_ = df['workload']==j
        y = df[filter_].groupby(['node_count'])['avgcpu'].mean(numeric_only=True)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        dfs.append(pd.DataFrame(data={'workload':[j],'slope':[slope],'intercept':[intercept],'error':[std_err],'r_value':[r_value],'p_value':[p_value],},index=[j]))
    all=pd.concat(dfs)
    return all 

def scale_out_fit_gradboost(df,nodecounts,jobs):
    dfs = []
    x=sorted(nodecounts)
    oneDx = np.array(x)
    x=oneDx.reshape(-1, 1)
    for j in jobs:
        filter_ = df['workload']==j
        y = (df[filter_].groupby(['node_count'])['avgcpu'].mean(numeric_only=True))
        gbr = GradientBoostingRegressor()
        gbr.fit(x, y)
        #predvals = np.arange(min(x),max(x)+1).reshape(-1,1)
        y_pred=gbr.predict(x)
        jarray=[j for i in range(len(y_pred))]
        dfs.append(pd.DataFrame(data={'noof_nodes':oneDx, 'predicted_util':y_pred},index=jarray))
    all=pd.concat(dfs)
    return all 

def get_gradboost_models(df,nodecounts,jobs):
    dfs = []
    x=sorted(nodecounts)
    oneDx = np.array(x)
    x=oneDx.reshape(-1, 1)
    for j in jobs:
        filter_ = df['workload']==j
        y = (df[filter_].groupby(['node_count'])['avgcpu'].mean(numeric_only=True))
        gbr = GradientBoostingRegressor()
        gbr.fit(x, y)
        dfs.append(pd.DataFrame(data={'workload':j, 'model':gbr},index=j))
    all=pd.concat(dfs)
    return all 
    
#linregressresult=scale_out_fit_linregress(df,nodecounts,jobs)
#st.write(linregressresult)
#gradboostresult = scale_out_fit_gradboost(df,nodecounts,jobs)
#st.write(gradboostresult)

def plot_linregress(linregressresult):

    plt.scatter(x, y)
    plt.show()



#possibly convert into dict 
nodes=st.slider("Node Count", min_value=2,max_value=100)
instance=st.radio("Instance Type",instancetypes,horizontal=True)
taskload=st.radio("Job",jobs,horizontal=True)
# indicate difference between earlier info which is based on user input and location which is always present 

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

#plot_avg_util_by_noof_nodes(df)

st.markdown("# Methodology")

st.sidebar.write("\n ")

