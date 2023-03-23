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

@st.cache(show_spinner=False, suppress_st_warning=True)
def train_gbr(df):
    # remove features to predict
    y1 = df.pop('avgcpu')
    y2 = df.pop('elapsed_time')

    # unnceccessary feature 
    unencoded_X=df.copy()
    name = df.pop("name")
    X = df
    # get columns containing text, apply label encoder and transform text to numbers
    #cat_cols = X.select_dtypes(include='object').columns
    #d = defaultdict(preprocessing.LabelEncoder)
    #X[cat_cols] = X[cat_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    encoded_features={}
    #st.write(unencoded_X)
    for col in X.columns:
        if col != 'node_count':
            le = preprocessing.LabelEncoder()
            encoded_features[col] = le.fit(X[col].values.astype(str))
            X[col] = encoded_features[col].transform(X[col].values.astype(str))


    #for col in X.columns:
    #    X[col] = encoded_features[col].transform(X[col].values)

    #le = preprocessing.LabelEncoder()
    #le.fit(X)
    #X[cat_cols] = X[cat_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))

    #le.transform(X)
    # split dataset
    #X1_train, X1_test, y1_train, y1_test = train_test_split(X,y1,test_size=0.1, random_state=0)
    #X2_train, X2_test, y2_train, y2_test = train_test_split(X,y2,test_size=0.1, random_state=0)

    # set regression model parameters to tweak later and see how results change
    param_grid = {
        "n_estimators":[250],
        "max_depth": [4],
        "min_samples_split": [4],
        "subsample": [0.8,1],
        "max_depth": [5, 20],
        "learning_rate": [0.05,0.2]

    }

    gbr1 = GradientBoostingRegressor()
    gbr2 = GradientBoostingRegressor()
    with st.spinner("Training Resource Utilisation Estimator Model"):
        cpu_reg = GridSearchCV(gbr1, param_grid, cv=2)
    with st.spinner("Training Elapsed Time Estimatior Model"):
        time_reg = GridSearchCV(gbr2, param_grid, cv=2)
    #print(X)
    cpu_reg.fit(X, y1)
    time_reg.fit(X, y2)

    cpu_pred = cpu_reg.predict(X)
    time_pred = time_reg.predict(X)
    predicted_cpu_time=pd.DataFrame(data={"true_cpu":y1,"predicted_cpu":cpu_pred,"true_time": y2, "predicted_time":time_pred, "name":name})
    #predicted_cpu_time["node_count"], predicted_cpu_time["instance_type"], predicted_cpu_time["job_input"], predicted_cpu_time["engine_type"], predicted_cpu_time["data_size"], _ = predicted_cpu_time["name"].str.split("_", expand=True) 
    #name_info = {"node_count": node_count, "instance_type":instance_type, "job_input": job_input, "engine_type": engine_type, "data_size": data_size}
    #predicted_cpu_time.merge(unencoded_X, on='name')
    #print(predicted_cpu_time)
    #st.write(predicted_cpu_time)
    #st.write(X)
    for col in X.columns:
        if col == 'node_count':
            predicted_cpu_time[col] = X[col]
        else:
            predicted_cpu_time[col] = encoded_features[col].inverse_transform(X[col])
    return (cpu_reg,time_reg,predicted_cpu_time,encoded_features)


def predict_gbr(util_reg,time_reg,chosen_features, encoded_features):
    #save cluster type to use later
    instance_type = chosen_features["instance_type"]
    #encode feature labels
    # get columns containing text, apply label encoder and transform text to numbers
    #cat_cols = X.select_dtypes(include='object').columns
    #d = defaultdict(preprocessing.LabelEncoder)
    #chosen_Features[cat_cols] = chosen_Features[cat_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    for col in chosen_features:
        if col != 'node_count':
            chosen_features[col] = encoded_features[col].transform(chosen_features[col].values)
    e_util = util_reg.predict(chosen_features)
    e_time = time_reg.predict(chosen_features)
    #print(f'prediction: {e_util}, type: {type(e_util)}' )
    return (e_util, e_time, instance_type)

def time_format(seconds: int) -> str:
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'


