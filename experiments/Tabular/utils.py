"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import numpy as np
from copy import deepcopy

import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from pyfd.decompositions import get_components_tree, get_components_ebm
from pyfd.decompositions import get_interventional_from_anchored, get_PDP_PFI_importance
from pyfd.shapley import interventional_treeshap, shap_from_decomposition
from pyfd.utils import ravel


class CallableModel(object):
    def __init__(self, model):

        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            self.callable = lambda x : model.predict_proba(x)[:, 1]
        else:
            self.callable = model.decision_function
        self.n_calls = 0

    def reset_n_calls(self):
        self.n_calls = 0

    def __call__(self, X):
        N = X.shape[0]
        self.n_calls += N
        results = self.callable(X)
        return results


def get_disagreement_global(attr1: np.ndarray, attr2: np.ndarray):
    l2_disagreement = np.sum((attr1 - attr2)**2)
    cosine_disagreement = np.sum(attr1 * attr2) / (np.linalg.norm(attr1)* np.linalg.norm(attr2))
    return l2_disagreement, cosine_disagreement


def get_disagreement_local(attr1: np.ndarray, attr2: np.ndarray):
    N = attr1.shape[0]
    l2_disagreement = 0
    cosine_disagreement = 0
    for i in range(N):
        l2_delta, cosine_delta = get_disagreement_global(attr1[i], attr2[i])
        l2_disagreement += l2_delta
        cosine_disagreement += cosine_delta
    return l2_disagreement / N, cosine_disagreement / N



# Custom train/test split for reproducability (random_state is always 42 !!!)
def custom_train_test_split(X, y, task):
    N = X.shape[0]
    ratio = 0.1 if N > 6000 else 0.2
    if task == "regression":
        return train_test_split(X, y, test_size=ratio, random_state=42)
    else:
        return train_test_split(X, y, test_size=ratio, random_state=42, stratify=y)



def get_cross_validator(k, task, split_seed, split_type):
    # Train / Test split and cross-validator. Dont look at the test yet...
    if task == "regression":
        if split_type == "Shuffle":
            cross_validator = ShuffleSplit(n_splits=k,
                                           test_size=0.1,
                                           random_state=split_seed)
        elif split_type == "K-Fold":
            cross_validator = KFold(n_splits=k, shuffle=True,
                                    random_state=split_seed)
        else:
            raise ValueError("Wrong type of cross-validator")

    # Binary Classification
    else:
        if split_type == "Shuffle":
            cross_validator = StratifiedShuffleSplit(n_splits=k,
                                                     test_size=0.1,
                                                     random_state=split_seed)
        elif split_type == "K-Fold":
            cross_validator = StratifiedKFold(n_splits=k,
                                              shuffle=True,
                                              random_state=split_seed)
        else:
            raise ValueError("Wrong type of cross-validator")

    return cross_validator



def get_hp_grid(filename):

    def to_eval(string):
        if type(string) == str:
            split = string.split("_")
            if len(split) == 2:
                return split[1]
            else:
                return None
        else:
            return None

    hp_dict = json.load(open(filename, "r"))
    for key, value in hp_dict.items():
        # Iterate over list
        if type(value) == list:
            for i, element in enumerate(value):
                str_to_eval = to_eval(element)
                if str_to_eval is not None:
                    value[i] = eval(str_to_eval)
        # Must be evaluated
        if type(value) == str:
            str_to_eval = to_eval(value)
            if str_to_eval is not None:
                hp_dict[key] = eval(str_to_eval)
    return hp_dict



def get_background(X, background_size, random_state):
    np.random.seed(random_state)
    idx_choose = np.random.choice(range(X.shape[0]), background_size, replace=False)
    background = X[idx_choose]
    return background


MODEL = {
        "ebm" : {"regression": ExplainableBoostingRegressor(),
                "classification": ExplainableBoostingClassifier()
                },
        "rf" : {"regression": RandomForestRegressor(),
                "classification": RandomForestClassifier()
                },
        "gbt" : {"regression": HistGradientBoostingRegressor(),
                "classification": HistGradientBoostingClassifier()
                }
        }


def get_sensitivity_n(phi, callable_model, decomposition, background, features, n=1, n_samples=10):
    N, d = background.shape
    N = min(100, N)
    D = len(features)
    fidelity_data = np.zeros((N, n_samples))
    background_copy = deepcopy(background)
    for i in range(N):
        for j in range(n_samples):
            group_subset = np.random.choice(a=range(D), size=(n,), replace=False)
            u = ravel([features.Imap_inv[k] for k in group_subset])
            not_u = [k for k in range(d) if k not in u]
            background_copy[:, not_u] = background[i, not_u]
            fidelity_data[i, j] = (decomposition[()][i] - callable_model(background_copy).mean()
                                        - sum([phi[i, j] for j in group_subset])) ** 2
            background_copy[:, not_u] = background[:, not_u]
    fidelity = np.mean(fidelity_data)
    return fidelity

def get_INFD(phi, callable_model, decomposition, background, features, n_samples=10):
    N, d = background.shape
    N = min(100, N)
    D = len(features)
    fidelity_data = np.zeros((N, n_samples))
    background_copy = deepcopy(background)
    for i in range(N):
        for j in range(n_samples):
            group_subset = np.random.randint(low=0, high=2, size=(D,))
            u = ravel([features.Imap_inv[k] for k in range(D) if group_subset[k]])
            not_u = [k for k in range(d) if k not in u]
            background_copy[:, not_u] = background[i, not_u]
            fidelity_data[i, j] = (decomposition[()][i] - callable_model(background_copy).mean()
                                        - sum([phi[i, j] for j in range(D) if group_subset[j]])) ** 2
            background_copy[:, not_u] = background[:, not_u]
    fidelity = np.mean(fidelity_data)
    return fidelity


def get_SWF(phi, callable_model, decomposition, background, features, n_samples=10):
    N, d = background.shape
    N = min(100, N)
    D = len(features)
    fidelity_data = np.zeros((N, n_samples))
    background_copy = deepcopy(background)
    for i in range(N):
        for j in range(n_samples):
            p = 1/(np.arange(1, D) * (D - np.arange(1, D)))
            p = p / p.sum()
            subset_size = np.random.choice(range(1, D), size=(1,), p=p)[0]
            group_subset = np.random.choice(a=range(D), size=(subset_size,), replace=False)
            u = ravel([features.Imap_inv[k] for k in group_subset])
            background_copy[:, u] = background[i, u]
            fidelity_data[i, j] = (callable_model(background_copy).mean() - decomposition[()].mean()
                                        - sum([phi[i, j] for j in group_subset])) ** 2
            background_copy[:, u] = background[:, u]
    fidelity = np.mean(fidelity_data)
    return fidelity

def get_disagreements_and_fidelity(model_name, model, background, features):
    # Compute explanations
    if model_name in ["rf", "gbt"]:
        if model_name == "rf":
            callable_model = lambda x: model.predict_proba(x)[:, 1]
        else:
            callable_model = model.decision_function
        decomposition = get_components_tree(model, background, background, features, anchored=True)
        shap_values = interventional_treeshap(model, background, background, features)
    else:
        callable_model = model.decision_function
        decomposition = get_components_ebm(model, background, background, features, anchored=True)
        shap_values = shap_from_decomposition(get_interventional_from_anchored(decomposition))
    I_PDP, I_PFI = get_PDP_PFI_importance(decomposition)

    # Compute disagreements local and global
    disagreement = {}
    # disagreement["l2_global"], disagreement["cosine_global"] = get_disagreement_global(I_PDP, I_PFI)
    partial_dependence = np.zeros((background.shape[0], len(features)))
    pfi = np.zeros((background.shape[0], len(features)))
    for i in range(len(features)):
        partial_dependence[:, i] = decomposition[(i,)].mean(1)
        pfi[:, i] = -1 * decomposition[(i,)].mean(0)
    disagreement["l2_pdp_vs_shap"], _ = get_disagreement_local(partial_dependence, shap_values)
    disagreement["l2_pdp_vs_pfi"], _ = get_disagreement_local(partial_dependence, pfi)
    disagreement["l2_shap_vs_pfi"], _ = get_disagreement_local(pfi, shap_values)

    fidelity = {}
    # Sensitivity-1
    fidelity["sensitivity_1_pdp"] = get_sensitivity_n(partial_dependence, callable_model, decomposition, background, features)
    fidelity["sensitivity_1_shap"] = get_sensitivity_n(shap_values, callable_model, decomposition, background, features)
    fidelity["sensitivity_1_pfi"] = get_sensitivity_n(pfi, callable_model, decomposition, background, features)

    # INFD
    fidelity["INFD_pdp"]  = get_INFD(partial_dependence, callable_model, decomposition, background, features)
    fidelity["INFD_shap"] = get_INFD(shap_values, callable_model, decomposition, background, features)
    fidelity["INFD_pfi"]  = get_INFD(pfi, callable_model, decomposition, background, features)

    # SWF
    fidelity["SWF_pdp"]  = get_SWF(partial_dependence, callable_model, decomposition, background, features)
    fidelity["SWF_shap"] = get_SWF(shap_values, callable_model, decomposition, background, features)
    fidelity["SWF_pfi"]  = get_SWF(pfi, callable_model, decomposition, background, features)

    return disagreement, fidelity
