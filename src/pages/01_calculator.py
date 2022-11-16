import streamlit as st
import pandas as pd 

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

st.title("Cloud Carbon Calculator")






df = st.session_state['df']


tasks=df["workload"].unique()
locations=['United Kingdom','Germany','France']
instancetypes=df["cluster_type"].unique()
nodecounts=df["node_count"].unique()
nodecounts.sort()


#possibly convert into dict 
nodes=st.select_slider("Node Count", nodecounts)
instance=st.radio("Instance Type",instancetypes,horizontal=True)
location=st.radio("Location",locations,horizontal=True)
taskload=st.radio("Task",tasks,horizontal=True)
scope3=st.checkbox("Include Scope 3 Emissions")
with st.expander("Override preset data"):
    # could allow more overrides (elapsed time, avg cpu util, power curve etc but would probably need to be on different page?)
    PUE = st.number_input("Power Usage Effectiveness of data center",min_value=1.0,value=1.135, help="Minumum value of 1")
    CARBON_INTENSITY = st.number_input("Grid Carbon Intensity (g/CO2/kWh)",value=228,min_value=0)
    AVG_UTIL = st.number_input("Average CPU Utilisation",value=0.5,min_value=0.0)
        # incl rest of data processing
        # try to interpolate between scaleouts assuming sparse dataset 
run=st.button("Calculate")

if run:
    try:
        # call some functions to calculate result 
        power=2
        amount=100
        avg_load=0.8
        scope3val=5
        rec_cluster='c4.large'
        scopeexp=f", including {scope3val}g/CO2/kWh scope 3 emissions." if scope3 else "."

        st.success(f"Carbon Footprint Result is: {amount} gCO2eq. Your recommended EC2 Cluster is: {rec_cluster}",icon="✔️")
        st.info(f"Calculation: {power} watt device at {avg_load*100}% load, multiplied by {PUE} PUE and {CARBON_INTENSITY}g/CO2/kWh{scopeexp}")
        

    except Exception as e:
        st.exception(f"Failed. Error {e}",icon="❌")

