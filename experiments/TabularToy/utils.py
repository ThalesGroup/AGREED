"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from itertools import combinations
from typing import List, Callable

from agreed.explainers.tabular.decomposition import TabularPartition


class InteractionBlock(metaclass=ABCMeta):
    """ An interaction block between a group of variables """

    def __init__(self, feature_group: List[int]):
        self.feature_group = feature_group

    def __len__(self):
        return len(self.feature_group)

    @abstractmethod
    def predict(self, x):
        pass


class ProductBlock(InteractionBlock):

    def __init__(self, feature_group: List[int]):
        super().__init__(feature_group)

    def predict(self, x: np.ndarray):
        x_subset = x[:, self.feature_group]
        N, d = x_subset.shape
        output = np.zeros(N)
        for u in combinations(range(d), 2):
            output += x_subset[:, u[0]] * x_subset[:, u[1]]
        return output


class GaussianBlock(InteractionBlock):

    def __init__(self, feature_group: List[int]):
        super().__init__(feature_group)

    def predict(self, x: np.ndarray):
        x_subset = x[:, self.feature_group]
        return np.exp(-0.5 * np.sum(x_subset**2, axis=1))


def relu(x):
    return x * (x > 0)

NON_LINEARITIES = [np.sin, np.cos, np.tanh, relu]

class HiddenUnitBlock(InteractionBlock):

    def __init__(self, feature_group: List[int]):
        super().__init__(feature_group)
        # Choose a random weight vector
        self.w = np.random.normal(size=(len(feature_group)))
        # Choose a Random Non-Linearity
        self.non_lin: Callable = NON_LINEARITIES[np.random.randint(0, 4)]


    def predict(self, x: np.ndarray):
        x_subset = x[:, self.feature_group]
        return self.non_lin( np.sum(self.w*x_subset, axis=1) )


INTERACTION_BLOCKS = [ProductBlock, GaussianBlock, HiddenUnitBlock]


def generate_data(N: int, d: int, rho: float, seed: int):
    np.random.seed(seed)
    X = np.zeros((N, d))
    remaining_groups = list(range(d))
    while len(remaining_groups) > 0:
        group_size = np.random.randint(2, 5)
        current_group = remaining_groups[:group_size]
        remaining_groups = remaining_groups[group_size:]
        d_group = len(current_group)
        cov = rho * np.ones(d_group) + (1 - rho) * np.eye(d_group)
        X[:, current_group] = np.random.multivariate_normal(mean=np.zeros(d_group,), cov=cov, size=(N,))
    X = X[:, np.random.permutation(range(d))]
    return X

def generate_blocks(d: int, seed: int):
    np.random.seed(seed)
    feature_groups = list(range(d))
    interaction_blocks: List[InteractionBlock] = []
    while len(feature_groups) > 0:
        group_size = np.random.randint(5, 11)
        interaction_block = INTERACTION_BLOCKS[np.random.randint(0, 3)]
        interaction_blocks.append(interaction_block(feature_groups[:group_size]))
        feature_groups = feature_groups[group_size:]
    return interaction_blocks

def get_ground_truth_partition(d: int, blocks: list[InteractionBlock]) -> TabularPartition:
    Imap = np.zeros(d, dtype=np.int32)
    Imap_inv: list[list[int]] = [[]] * len(blocks)
    D = len(blocks)
    for b, block in enumerate(blocks):
        Imap_inv[b] = block.feature_group
        Imap[block.feature_group] = b
    return TabularPartition(Imap, Imap_inv, d, D)


class Model(object):
    """ A model that keeps track of the number of times it is called """

    def __init__(self, blocks: List[InteractionBlock]):
        self.blocks = blocks
        self.n_calls = 0

    def reset_n_calls(self):
        self.n_calls = 0

    def __call__(self, X: np.ndarray):
        N = X.shape[0]
        self.n_calls += N
        results = np.zeros(N)
        for block in self.blocks:
            results += block.predict(X)
        return results
