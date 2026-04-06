"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
from typing import TYPE_CHECKING

from agreed.explainers.tabular.decomposition import (
    numpy_get_anchored_components_u,
    get_Psi,
    get_VIN_clique,
    get_TabularPartition_from_Imap_inv,
)
from agreed.explainers.tabular.decomposition import (
    get_disagreements_from_decomposition,
    update_partition,
    TabularPartition,
)

if TYPE_CHECKING:
    from agreed.explainers.tabular.decomposition import Decomposition


def lasso_term(partition, lambd):
    c = np.array(list(map(lambda x: len(x), partition)))
    c = c * (c - 1) / 2
    return lambd * np.sum(c)


def precompute_pairwise_interactions(
    decomposition: Decomposition,
    model: Callable,
    background: np.ndarray,
    interactions_graph: nx.Graph,
) -> dict[tuple[int, ...], dict[tuple[int, ...], np.ndarray]]:

    # Maintain buffered joint effects to avoid recomputing
    buffered_merge_values: dict[tuple[int, ...], dict[tuple[int, ...], np.ndarray]] = {}

    # if interactions_graph is None:
    # TODO this algorithm can be accelated by using a sparse interaction graph
    d = background.shape[1]
    for i in range(d):
        connections = interactions_graph.edges(i)
        if len(connections) > 0:
            buffered_merge_values[(i,)] = {}
        for node_1, node_2 in connections:
            other_node = node_1 if node_2 == i else node_2
            if other_node < i:
                continue
            new_coalition = [node_1, node_2]
            # A bit hacky here...
            value_function = numpy_get_anchored_components_u(
                decomposition, model, (0,), [new_coalition], background
            ).mean(1)
            buffered_merge_values[(i,)][(other_node,)] = value_function
    return buffered_merge_values


def TabularIGreedy(
    model: Callable,
    background: np.ndarray,
    threshold: float = 0.1,
    verbose: bool = True,
) -> list[list[int]]:
    """
    Baseline from paper
    Xu, S., Cuppers, J., & Vreeken, J. (2024). Succinct Interaction-Aware
    Explanations. arXiv preprint arXiv:2402.05566.

    Code adapted from https://github.com/Schascha1/iSHAP/blob/main/ishap.py
    """

    # TODO this could be replaced by a sparse graph
    d = background.shape[1]
    Imap_inv = [[i] for i in range(d)]
    interactions_graph = nx.complete_graph(d)

    value_target = model(background)
    decomposition: Decomposition = {}
    decomposition[()] = value_target

    # compute the individual effect of each singleton
    curr_component_sum = value_target.mean()
    partition_values = {}
    for i in range(d):
        value_function = numpy_get_anchored_components_u(
            decomposition, model, (i,), Imap_inv, background
        ).mean(1)
        partition_values[(i,)] = value_function
        curr_component_sum += value_function
    curr_loss = np.mean((curr_component_sum - value_target) ** 2)
    lambd = threshold * curr_loss
    curr_regul_term = lasso_term(Imap_inv, lambd)
    curr_obj = curr_loss + curr_regul_term

    # Merge the coalitions: greedy merging until no more merges are
    # possible either due to size or threshold
    final_Imap_inv = deepcopy(Imap_inv)
    buffered_merge_values = precompute_pairwise_interactions(
        decomposition,
        model,
        background,
        interactions_graph,
    )

    while len(Imap_inv) > 1:
        if verbose:
            print(f"Objective : {curr_obj:.2e}")

        best_obj = curr_obj
        best_merge = None
        best_component_sum = 0.0
        best_loss = 0.0
        best_regul_term = 0.0
        for set_1 in buffered_merge_values.keys():
            for set_2 in buffered_merge_values[set_1].keys():
                new_component_sum = (
                    curr_component_sum
                    - partition_values[set_1]
                    - partition_values[set_2]
                    + buffered_merge_values[set_1][set_2]
                )
                new_regul_term = (
                    curr_regul_term
                    - lasso_term([set_1, set_2], lambd)
                    + lasso_term([set_1 + set_2], lambd)
                )
                new_loss = np.mean((value_target - new_component_sum) ** 2)
                new_obj = new_loss + new_regul_term
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_loss = new_loss
                    best_regul_term = new_regul_term
                    best_component_sum = new_component_sum
                    best_merge = (set_1, set_2)
        if best_merge is None:
            if verbose:
                print("Not able to decrease the objective further\n")
            break
        else:

            # Update scores and partition
            curr_obj = best_obj
            curr_loss = best_loss
            curr_regul_term = best_regul_term
            curr_component_sum = best_component_sum
            new_coalition = best_merge[0] + best_merge[1]
            Imap_inv.remove(list(best_merge[0]))
            Imap_inv.remove(list(best_merge[1]))
            Imap_inv.append(list(new_coalition))
            partition_values.pop(best_merge[0], None)
            partition_values.pop(best_merge[1], None)
            partition_values[new_coalition] = buffered_merge_values[best_merge[0]][
                best_merge[1]
            ]
            final_Imap_inv = deepcopy(Imap_inv)

            # update buffered merge values
            buffered_merge_values.pop(best_merge[0], None)
            buffered_merge_values.pop(best_merge[1], None)
            for set_1 in buffered_merge_values.keys():
                buffered_merge_values[set_1].pop(best_merge[0], None)
                buffered_merge_values[set_1].pop(best_merge[1], None)
            for set_1 in buffered_merge_values.keys():
                compute_merge = False
                for element in set_1:
                    edges = interactions_graph.edges(element)
                    for node_1, node_2 in edges:
                        other_node = node_1 if node_2 == element else node_2
                        if other_node in new_coalition:
                            compute_merge = True
                            break
                    if compute_merge:
                        break
                if compute_merge:
                    u = list(new_coalition + set_1)
                    value_function = numpy_get_anchored_components_u(
                        decomposition, model, (0,), [u], background
                    ).mean(1)
                    buffered_merge_values[set_1][new_coalition] = value_function
            buffered_merge_values[new_coalition] = {}

    return get_TabularPartition_from_Imap_inv(final_Imap_inv)


# TODO
def TabularPairwise(
    model: Callable,
    background: np.ndarray,
    threshold: float = 0.1,
    init_decomposition: Decomposition | None = None,
    verbose: bool = True,
    return_decomposition: bool = False,
) -> TabularPartition:
    """
    Baseline from paper
    Tsang, M., Rambhatla, S., & Liu, Y. (2020). How does this interaction affect me?
    interpretable attribution for feature interactions. Advances in neural
    information processing systems, 33, 6147-6159.

    Code was adapted from https://github.com/mtsang/archipelago to use our
    functional decomposition API.
    """
    # Setup
    N, d = background.shape
    feature_groups = [[i] for i in range(d)]
    main_effects = [(i,) for i in range(d)]
    pairwise_interactions = [u for u in combinations(range(d), 2)]

    # Compute all pair-wise interactions
    key: tuple[int, ...]
    if init_decomposition is None:
        decomposition: Decomposition = {}
        decomposition[()] = model(background)
        # We start by computing the main effects
        for key in main_effects:
            decomposition[key] = numpy_get_anchored_components_u(
                decomposition, model, key, feature_groups, background
            )
        # We then compute pairwise interactions
        for key in tqdm(
            pairwise_interactions,
            desc="Computing all pair-wise interactions",
            disable=not verbose,
        ):
            decomposition[key] = numpy_get_anchored_components_u(
                decomposition, model, key, feature_groups, background
            )
    else:
        decomposition = init_decomposition

    # Only keep the significant interactions
    significant_interactions = []
    model_var = decomposition[()].var()
    for key in pairwise_interactions:
        if (
            decomposition[key].mean(0).var() + decomposition[key].mean(1).var()
            > 2 * threshold * model_var
        ):
            significant_interactions.append(key)
    final_Imap_inv = get_VIN_clique(significant_interactions, d)
    return get_TabularPartition_from_Imap_inv(final_Imap_inv)


def TabularRecurse(
    model: Callable,
    background: np.ndarray,
    threshold: float = 0.1,
    init_decomposition: Decomposition | None = None,
) -> TabularPartition:
    """
    Baseline from paper
    Sivill, T., & Flach, P. (2023). Shapley sets: Feature attribution via
    recursive function decomposition. arXiv preprint arXiv:2307.01777.

    Code was done from scratch since to previous implementation was available.
    """
    # Setup
    N, d = background.shape
    decomposition: Decomposition = {}
    decomposition[()] = model(background)
    additive_keys = [(i,) for i in range(d)]
    Imap_inv_new = []
    Imap_inv = [[i] for i in range(d)]

    key: tuple[int, ...]
    if init_decomposition is None:
        # We start by computing the main effects
        for key in additive_keys:
            decomposition[key] = numpy_get_anchored_components_u(
                decomposition, model, key, Imap_inv, background
            )
    else:
        decomposition = init_decomposition

    # Order variables relative to their potential Psi
    Psi = get_Psi(decomposition)
    feature_order = list(np.argsort(-Psi))

    G_1 = feature_order[0:1]
    G_2 = feature_order[1:]

    def RecurseInteract(G_1, G_2, model, background, threshold, null_component):
        Imap_inv = [G_1, G_2]
        decomposition = {}
        decomposition[()] = null_component
        for key in [(0,), (1,), (0, 1)]:
            decomposition[key] = numpy_get_anchored_components_u(
                decomposition, model, key, Imap_inv, background
            )
        var = decomposition[()].var()
        interaction_strength = 0.5 * (
            decomposition[(0, 1)].mean(1).var() + decomposition[(0, 1)].mean(0).var()
        )
        if interaction_strength > threshold * var:
            N_2 = len(G_2)
            if N_2 == 1:
                G_1 = G_1 + G_2
            else:
                G_1_fused_left = RecurseInteract(
                    G_1, G_2[: N_2 // 2], model, background, threshold, null_component
                )
                G_1_fused_right = RecurseInteract(
                    G_1, G_2[N_2 // 2 :], model, background, threshold, null_component
                )
                G_1 = set(G_1_fused_left) | set(G_1_fused_right)
                G_1 = list(G_1)
        return G_1

    while len(G_2) > 0:

        G_1_new = RecurseInteract(
            G_1, G_2, model, background, threshold, decomposition[()]
        )

        # G_1 does not interact with any features in G_2
        if sorted(G_1) == sorted(G_1_new):
            Imap_inv_new.append(G_1_new)
            G_1 = G_2[0:1]
            G_2 = G_2[1:]
        # G_1 interacts with some features in G_2 so we expand G_1
        else:
            G_1 = G_1_new
            G_2 = [g for g in G_2 if g not in G_1]

        # At the end, we put the remaining features in a group
        if len(G_2) == 0:
            Imap_inv_new.append(G_1)

    return get_TabularPartition_from_Imap_inv(Imap_inv_new)


def get_interactions_candidate(
    decomposition: Decomposition,
    partition: TabularPartition,
    features_to_ignore: list[int] | None = None,
    min_potential: float = 0.0,
    max_group_size: int | None = None,
) -> list[tuple[int, int]]:
    """
    Return a list of pairwise interactions to investigate while respecting the various
    partition hard constraints. If no interaction matches the constraints,
    an empty list is returned.
    """
    D, d = partition.D, partition.d
    groups_to_ignore: list[tuple[int, ...]] | np.ndarray
    if max_group_size is None:
        max_group_size = d
    if features_to_ignore is None:
        groups_to_ignore = []
    else:
        groups_to_ignore = [partition.Imap[i] for i in features_to_ignore]

    # Find the group that interact most some others
    Psi = get_Psi(decomposition)
    Psi[groups_to_ignore] = -np.inf

    # Further prune the search
    group_sizes = np.array([len(group) for group in partition.Imap_inv])
    # Groups that break constraints are ignored
    groups_to_ignore = np.where(
        (Psi < min_potential) | (group_sizes > max_group_size)
    )[0]
    Psi[groups_to_ignore] = -np.inf

    # Main loop, try each candidate and stop if they can be fused with at least another group
    candidate = int(np.argmax(Psi))
    while Psi[candidate] >= 0.0:
        Psi[candidate] = -np.inf
        interaction_candidates = [(k, candidate) for k in range(D) if Psi[k] >= 0]
        interaction_candidates = [
            (i, j)
            for (i, j) in interaction_candidates
            if (
                len(partition.Imap_inv[i]) + len(partition.Imap_inv[j])
                <= max_group_size
            )
        ]
        if len(interaction_candidates) > 0:
            return interaction_candidates
        candidate = int(np.argmax(Psi))
    return []


def TabularAGREED(
    model: Callable,
    background: np.ndarray,
    threshold: float = 0.1,
    features_to_ignore: list[int] | None = None,
    min_potential: float = 0.0,
    max_group_size: int | None = None,
    init_decomposition: Decomposition | None = None,
    verbose: bool = False,
) -> TabularPartition:
    """
    Adaptive Grouping to ReducE Explanation Disagreements:
    This approach iteratively group features while dynamically
    keeping track of all relevant between-groups interactions.
    Early convergence implies that this algorithm will not compute
    the O(d^2) interactions between all features

    Parameters
    ----------
    model: Callable
        The ML model to explain
    background: np.ndarray
        The (N, d) background dataset used to mask missing features
    threshold: float, default=0.1
        The algorithm converges when disagreements (relative to no groups)
        fall bellow this value.
    features_to_ignore: list[int] | None, default=None
        Certain features can be excempt from the partitioning algorithm.
    min_potential: float, default=0.0
        Only consider fusing group i with another if its potential is higher than
        this value. This allows to aggressively prune the search space.
    max_group_size: int | None, default=None
        Only consider partitions where each group has a bounded sizes. Groups of size
        at most three remain interpretable for tabular data.
    init_decomposition: Decomposition | None, default=None
        A precomputed functional decomposition containing all main effects
        can be provided. Otherwise, this decomposition will be calculated before
        entering the main loop.
    verbose: bool, default=False
        Print the disagreements during the runtime of AGREED.
    """
    # Setup
    N, d = background.shape
    decomposition: Decomposition = {}
    decomposition[()] = model(background)
    additive_keys = [(i,) for i in range(d)]
    Imap = np.arange(d, dtype=np.int32)
    Imap_inv = [[i] for i in range(d)]
    partition = TabularPartition(Imap, Imap_inv, d, d)
    key: tuple[int, ...]

    if verbose:
        print("\n=================================================")
        print("Running AGREED with settings:")
        print(f"    features_to_ignore: {features_to_ignore}")
        print(f"    min_potential: {min_potential}")
        print(f"    max_group_size: {max_group_size}")

    if init_decomposition is None:
        # We start by computing the main effects
        for key in additive_keys:
            decomposition[key] = numpy_get_anchored_components_u(
                decomposition, model, key, partition.Imap_inv, background
            )
    else:
        decomposition = init_decomposition
    disagreement = get_disagreements_from_decomposition(decomposition, metric="l2")
    init_disagreement = disagreement
    curr_partition = partition

    if verbose:
        print(f"Init Disagreement : {disagreement: .2e}")
    # Main loop, we iteratively group patches until the gap error falls bellow a threshold
    while disagreement > threshold * init_disagreement:

        interaction_candidates = get_interactions_candidate(
            decomposition,
            curr_partition,
            features_to_ignore,
            min_potential,
            max_group_size,
        )

        # Early stopping when no more groups can be fused while respecting contraints
        if len(interaction_candidates) == 0:
            if verbose:
                print("Exited because no more groups could be fused")
                print("=================================================")
            return curr_partition

        # Compute interaction with all other patches
        interaction_strength = []
        for key in interaction_candidates:
            decomposition[key] = numpy_get_anchored_components_u(
                decomposition, model, key, curr_partition.Imap_inv, background
            )
            interaction_strength.append(
                np.abs(decomposition[key].mean(1)).mean()
                + np.abs(decomposition[key].mean(0)).mean()
            )
        groups_to_fuse = interaction_candidates[
            np.argmax(np.array(interaction_strength))
        ]

        # Fuse two patches into one
        decomposition, curr_partition = update_partition(
            groups_to_fuse, decomposition, curr_partition
        )

        # Recompute the Gap error
        disagreement = get_disagreements_from_decomposition(decomposition, metric="l2")

        if verbose:
            print(f"Disagreement : {disagreement: .2e}")

    if verbose:
        print("Exited because disagreements fell bellow threshold")
        print("=================================================\n")
    return curr_partition
