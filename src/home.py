import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

#st.set_page_config(layout="wide")
st.set_page_config(
    page_title="Cloud Carbon Calculator",
    page_icon="🖩",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Made By James Sharma for L4 Project"
    }
)

plt.style.use('dark_background')

upload_file = st.sidebar.file_uploader("Upload your own runs with utilisation information", type = ['csv'])

df = pd.read_csv('../data/processed/scout/average_utils/averages.csv')
# look into accepting multiple files
if upload_file is not None:
    df = pd.read_csv(upload_file)
st.session_state['df'] = df

st.title('Cloud Carbon Calculator')
st.text('This is a web app to allow carbon emission estimation of big data jobs')

