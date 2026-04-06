import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/glaberge/Desktop/Pytorch/AGREED/')

# Src imports
from src.explainers.tabular.grouping import TabularRecurse, TabularIGreedy


def generate_problem(N, seed, rho_12, rho_45, alpha, beta):
    # Generate input
    np.random.seed(seed)
    cov = np.eye(5)
    cov[0, 1] = rho_12
    cov[1, 0] = rho_12
    cov[3, 4] = rho_45
    cov[4, 3] = rho_45
    X = np.random.multivariate_normal(np.zeros(5), cov=cov, size=(N,))

    # Model to explain
    def h(X):
        return alpha * X[:, 0] * X[:, 1]  + beta * X[:, 3] * X[:, 4]

    return X, h

# We generate the data and models. The data X is stored in a (N, d) numpy array
X, h = generate_problem(1000, 42, 0.5, 0.75, 1, 2)
# res = TabularRecurse(h, X, 1e-2)
res = TabularIGreedy(h, X, 1e-2)
print(res)