"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Defining functional decompositions using feature partitions

from typing import Callable
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass

from agreed.explainers.utils import ravel, powerset

# A decomposition maps an element u of the lattice space to an array storing
# the anchored decomposition h_u. Averaging these NxN matrices leads to
# the Marginal Decomposition (Equation 3 in the paper).
Decomposition = dict[tuple[int, ...], np.ndarray]


@dataclass
class TabularPartition(object):
    Imap: np.ndarray  # (d,) array that maps every feature to its group index
    Imap_inv: list[list[int]]  # Imap_inv[i] stores the features in group i
    d: int  # The number of features
    D: int  # The number of groups


def numpy_get_anchored_components_u(
    decomposition: Decomposition,
    model: Callable,
    key: Iterable,
    Imap_inv: list[list[int]],
    background: np.ndarray,
) -> np.ndarray:
    """
    Compute the Anchored Decompositions h_u at a given key u

    Parameters
    ---------
    decomposition: Decomposition
        The functional decomposition containing precomputed main effects and interactions.
    model: Callable
        The ML model to explain.
    key: Iterable[int]
        The interaction we want to compute. For instance, to compute the pairwise
        interaction between groups 3 and 4, the corresponding would be (3, 4).
    Imap_inv: list[list[int]]
        The inverse partition map (i.e. maps a group index to the features within that group).
    background: np.ndarray
        Dataset used to mask missing features.

    Returns
    ------
    result: np.ndarray
        (N, N) array storing the h_u components for the chosen interaction.
    """

    N = background.shape[0]
    result = np.zeros((N, N))
    data_temp = np.copy(background)
    U = ravel([Imap_inv[j] for j in key])
    # Compute h_{u,k}(x) for all x to eval
    for i in range(N):
        # Annihilation property
        annihilated = np.zeros((N,))
        for j in key:
            annihilated += (
                background[i, Imap_inv[j]] == background[:, Imap_inv[j]]
            ).all(axis=1)
        not_annihilated = np.where(annihilated == 0)[0]

        # Compute the component
        if len(not_annihilated) > 0:
            data_temp[not_annihilated.reshape((-1, 1)), U] = background[i, U]
            h_replace = model(data_temp[not_annihilated])
            result[i, not_annihilated] += h_replace
            # Remove all contributions of subsets to get the interaction
            for subset in powerset(key):
                if len(subset) > 0:
                    result[i, not_annihilated] -= decomposition[subset][
                        i, not_annihilated
                    ]
                else:
                    result[i, not_annihilated] -= decomposition[subset][not_annihilated]
    return result


def get_Psi(decomposition: Decomposition) -> np.ndarray:
    """Return the potential of each additive key (i,) in a decomposition"""
    Psi = []
    additive_keys = [key for key in decomposition.keys() if len(key) == 1]
    for key in additive_keys:
        Psi.append(
            np.abs(decomposition[key].mean(1) + decomposition[key].mean(0)).mean()
        )
    return np.array(Psi)


def get_disagreements_from_decomposition(
    decomposition: Decomposition, metric: str = "cosine"
):
    """Compute the most extreme disagreement -> between pure and full individual effects"""
    additive_keys = [key for key in decomposition.keys() if len(key) == 1]
    d = len(additive_keys)
    N = len(decomposition[()])
    attr_1 = np.zeros((N, d))
    attr_2 = np.zeros((N, d))
    for i, key in enumerate(additive_keys):
        attr_1[:, i] = decomposition[key].mean(1)
        attr_2[:, i] = -1 * decomposition[key].mean(0)
    # TODO
    if metric == "cosine":
        raise Exception()
        # TODO where was this function implemented before?
        # return ((1 - agreement(attr_1, attr_2)) / 2).mean()
    else:
        return ((attr_1 - attr_2) ** 2).sum(1).mean()


def get_VIN_clique(
    significant_interactions: Iterable[tuple[int, ...]], d: int
) -> list[list[int]]:
    """Iterate over all pairwise interactions and return the graphe's cliques"""

    cliques: list[list[int] | None] = [[i] for i in range(d)]  # The feature cliques
    clique_idx = [i for i in range(d)]  # In which clique a feature is part of
    for u in significant_interactions:
        if len(u) < 2:
            continue
        # Fuse right's clique with left's clique
        left_clique_idx = clique_idx[u[0]]
        right_clique_idx = clique_idx[u[1]]
        # We do not need to fuse if they are already part of the same clique
        if left_clique_idx == right_clique_idx:
            continue
        # Fuse cliques  left <- left + right
        for right_element in cliques[right_clique_idx]:
            clique_idx[right_element] = left_clique_idx
        cliques[left_clique_idx] = cliques[left_clique_idx] + cliques[right_clique_idx]
        cliques[right_clique_idx] = None
    return [sorted(clique) for clique in cliques if clique is not None]


def get_pixel_groups_from_clique(
    cliques: list[tuple], init_pixel_groups: list[list[int]]
) -> list[list[int]]:
    pixel_groups: list[list[int]] = []
    for clique in cliques:
        pixel_groups.append([])
        for j in clique:
            pixel_groups[-1] += init_pixel_groups[j]
    return pixel_groups


def update_partition(
    indices_to_fuse: tuple[int, int],
    decomposition: Decomposition,
    partition: TabularPartition,
):
    """Update the partition by fusing the groups with index i and j"""

    i, j = indices_to_fuse
    # Create a new decomposition and pixel_group
    new_decomposition: Decomposition = {(): decomposition[()]}
    new_Imap_inv = []
    new_Imap = -1 * np.ones(partition.d, dtype=np.int32)

    c = 0
    for k in range(partition.D):
        if (not i == k) and (not j == k):
            new_decomposition[(c,)] = decomposition[(k,)]
            new_Imap_inv.append(partition.Imap_inv[k])
            new_Imap[partition.Imap_inv[k]] = c
            c += 1
    # Transform the interaction into a joint main effect for (i,j)
    joint_effect = (
        decomposition[indices_to_fuse] + decomposition[(i,)] + decomposition[(j,)]
    )
    new_group = partition.Imap_inv[i] + partition.Imap_inv[j]
    new_decomposition[(c,)] = joint_effect
    new_Imap_inv.append(new_group)
    new_Imap[new_group] = c
    new_partition = TabularPartition(
        new_Imap, new_Imap_inv, partition.d, partition.D - 1
    )
    return new_decomposition, new_partition


def get_TabularPartition_from_Imap_inv(Imap_inv: list[list[int]]):
    D = len(Imap_inv)
    d = max(ravel(Imap_inv)) + 1
    Imap = np.zeros(d, dtype=np.int32)
    for g, group in enumerate(Imap_inv):
        Imap[group] = g
    return TabularPartition(Imap, Imap_inv, d, D)
