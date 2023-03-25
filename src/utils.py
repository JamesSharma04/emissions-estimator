import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


@st.cache(show_spinner=False, suppress_st_warning=True)
def train_gbr(df):
    # remove features to predict
    y1 = df.pop('avgcpu')
    y2 = df.pop('elapsed_time')

    name = df.pop("name")
    X = df
    # get columns containing text, apply label encoder and transform text to numbers
    encoded_features = {}
    for col in X.columns:
        if col != 'node_count':
            le = preprocessing.LabelEncoder()
            encoded_features[col] = le.fit(X[col].values.astype(str))
            X[col] = encoded_features[col].transform(X[col].values.astype(str))

    param_grid = {
        "n_estimators": [250],
        "min_samples_split": [4],
        "subsample": [0.8, 1],
        "max_depth": [5, 20],
        "learning_rate": [0.05, 0.2]

    }

    gbr1 = GradientBoostingRegressor()
    gbr2 = GradientBoostingRegressor()
    with st.spinner("Training Resource Utilisation Estimator Model"):
        cpu_reg = GridSearchCV(gbr1, param_grid, cv=2)
    with st.spinner("Training Elapsed Time Estimatior Model"):
        time_reg = GridSearchCV(gbr2, param_grid, cv=2)

    cpu_reg.fit(X, y1)
    time_reg.fit(X, y2)

    cpu_pred = cpu_reg.predict(X)
    time_pred = time_reg.predict(X)
    # construct df of results
    predicted_cpu_time = pd.DataFrame(
        data={"true_cpu": y1, "predicted_cpu": cpu_pred, "true_time": y2, "predicted_time": time_pred, "name": name})

    # inverse transform for presentation to user
    for col in X.columns:
        if col == 'node_count':
            predicted_cpu_time[col] = X[col]
        else:
            predicted_cpu_time[col] = encoded_features[col].inverse_transform(
                X[col])
    return (cpu_reg, time_reg, predicted_cpu_time, encoded_features)


def predict_gbr(util_reg, time_reg, chosen_features, encoded_features):
    # for use in power computation
    instance_type = chosen_features["instance_type"]
    # transform features for use in regressor
    for col in chosen_features:
        if col != 'node_count':
            chosen_features[col] = encoded_features[col].transform(
                chosen_features[col].values)

    e_util = util_reg.predict(chosen_features)
    e_time = time_reg.predict(chosen_features)
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
