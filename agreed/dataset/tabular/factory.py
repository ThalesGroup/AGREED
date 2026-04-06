import pandas as pd
import os
import numpy as np

# Sklearn
from ucimlrepo import fetch_ucirepo
from scipy.io import arff
from pyfd.features import Features
from pyfd.data import get_data_california_housing, get_data_marketing, get_data_credit


def get_data_nomao():
    # Load dataset
    data, _ = arff.loadarff(os.path.join("data", "nomao.arff"))
    df = pd.DataFrame(data)

    # Generate Features object
    ordinal_features = ['V7', 'V8', 'V15', 'V16', 'V23', 'V24', 'V31', 'V32', 'V39', 'V40', 'V47',
                            'V48', 'V55', 'V56', 'V63', 'V64', 'V71', 'V72', 'V79', 'V80', 'V87', 'V88',
                            'V92', 'V96', 'V100', 'V104', 'V108', 'V112', 'V116', 'Class']
    df[ordinal_features] = df[ordinal_features].apply(lambda x : x.str.decode('utf8'))
    df[ordinal_features] = df[ordinal_features].astype('float')
    X = df.to_numpy()[:, :-1]
    y = df['Class'].map({2: 1, 1: 0}).to_numpy()
    feature_types = ["num"] * X.shape[1]
    feature_names = list(df.columns[:-1])
    features = Features(X, feature_names, feature_types)

    return X, y, features


def get_data_spam():
    # fetch dataset
    fetch_data = fetch_ucirepo(id=94)
    X = fetch_data.data.features.to_numpy().astype(np.float64)
    y = fetch_data.data.targets.to_numpy().ravel()

    # Generate Features object
    feature_types = ["sparse_num"] * 57
    feature_names = list(fetch_data.variables["name"][:-1])
    features = Features(X, feature_names, feature_types)

    return X, y, features


def get_data_diabetes():
    # fetch dataset
    fetch_data = fetch_ucirepo(id=891)

    X = fetch_data.data.features.to_numpy().astype(np.float64)
    y = fetch_data.data.targets.to_numpy().ravel()
    feature_names = list(fetch_data.variables["name"][2:])
    feature_types = ["bool"] * 3 +  ["num"] + ["bool"] * 9 + ["num_int"] * 3 + ["bool"] * 2 +\
                    [["nominal", "Female", "Male"]] + ["num_int"] * 2
    features = Features(X, feature_names, feature_types)
    return X, y, features



def get_data_academic():
    # fetch dataset
    fetch_data = fetch_ucirepo(id=697)

    # data (as pandas dataframes)
    X = fetch_data.data.features.to_numpy().astype(np.float64)
    y = fetch_data.data.targets
    y = (y == 'Dropout').to_numpy().astype(int).ravel()

    feature_names = list(fetch_data.variables["name"][:-1])
    # TODO put categorical features
    feature_types = ["num_int"] * 36

    features = Features(X, feature_names, feature_types)
    return X, y, features


DATASET_MAPPING = {
    "california": get_data_california_housing,
    "marketing": get_data_marketing,
    "default_credit": get_data_credit,
    "spam": get_data_spam,
    "diabetes": get_data_diabetes,
    "academic": get_data_academic,
    "nomao": get_data_nomao,
}

TASK_MAPPING = {
    "california" : "regression",
    "marketing": "classification",
    "spam": "classification",
    "diabetes": "classification",
    "default_credit": "classification",
    "academic": "classification",
    "nomao": "classification"
}
