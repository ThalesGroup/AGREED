"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pyfd.decompositions import get_components_tree, get_PDP_PFI_importance
from pyfd.shapley import interventional_treeshap, get_SHAP_importance
from pyfd.plots import bar

from agreed.dataset.tabular.factory import get_data_california_housing
from agreed.explainers.tabular.grouping import TabularAGREED

X, y, features = get_data_california_housing()
print(features.summary())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = HistGradientBoostingRegressor(random_state=42).fit(X_train, y_train)
print(model.score(X_test, y_test))

# Explain predictions without grouping
n_background_samples = 1000
n_background_samples = 200
background = X_test[:n_background_samples]
decomposition = get_components_tree(
    model,
    background,
    background,
    features,
    anchored=True
)
shap_values = interventional_treeshap(
    model,
    background,
    background,
    features
)
I_PDP, I_PFI = get_PDP_PFI_importance(decomposition)
I_SHAP = get_SHAP_importance(shap_values)

bar([I_PFI, I_SHAP, I_PDP], features.names())
plt.xlim(0, 0.8)
plt.savefig("california_nogroups.pdf", bbox_inches='tight')

# Run AGREED
partition = TabularAGREED(
    model.predict,
    background,
    threshold=0.4,
    max_group_size=3,
    init_decomposition=decomposition,
    verbose=True
)
grouped_features = features.group([group for group in partition.Imap_inv if len(group) > 1])
print(grouped_features.summary())

# Explain predictions with grouping
decomposition = get_components_tree(
    model,
    background,
    background,
    grouped_features,
    anchored=True
)
shap_values = interventional_treeshap(
    model,
    background,
    background,
    grouped_features
)
I_PDP, I_PFI = get_PDP_PFI_importance(decomposition)
I_SHAP = get_SHAP_importance(shap_values)

bar([I_PFI, I_SHAP, I_PDP], grouped_features.names())
plt.xlim(0, 0.8)
plt.savefig("california_groups.pdf", bbox_inches='tight')
