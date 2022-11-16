import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



df = st.session_state['df']

st.header('Statistics of Dataset')
st.write(df.describe())


aaa=df.groupby('node_count')['avgcpu'].mean(numeric_only=True)
fig, ax = plt.subplots()
ax.plot(aaa)
plt.title("Average CPU Utilisation by Number of Nodes")
plt.xlabel("Node Count")
plt.ylabel("Utilisation (%)")
fig
