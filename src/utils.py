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

@st.cache(show_spinner=False)
def train_gbr(df):
    # remove features to predict
    y1 = df.pop('avgcpu')
    y2 = df.pop('elapsed_time')

    # unnceccessary feature 
    name = df.pop("name")
    X = df
    unencoded_X=X.copy()
    # get columns containing text, apply label encoder and transform text to numbers
    cat_cols = X.select_dtypes(include='object').columns
    d = defaultdict(preprocessing.LabelEncoder)
    X[cat_cols] = X[cat_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))

    # split dataset
    #X1_train, X1_test, y1_train, y1_test = train_test_split(X,y1,test_size=0.1, random_state=0)
    #X2_train, X2_test, y2_train, y2_test = train_test_split(X,y2,test_size=0.1, random_state=0)

    # set regression model parameters to tweak later and see how results change
    param_grid = {
        "n_estimators":[500,1000],
        "max_depth": [4,8],
        "min_samples_split": [2,4],
        "learning_rate": [0.005, 0.01, 0.05],
        "loss": ["squared_error"]
    }

    gbr1 = GradientBoostingRegressor()
    gbr2 = GradientBoostingRegressor()

    cpu_reg = GridSearchCV(gbr1, param_grid, cv=2)
    time_reg = GridSearchCV(gbr2, param_grid, cv=2)
    cpu_reg.fit(X, y1)
    time_reg.fit(X, y2)
    return (cpu_reg,time_reg)
