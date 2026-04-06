"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Grouping algorithms that are iterable -> to efficiently try a sequence of thresholds

from __future__ import annotations
from typing import Callable
import numpy as np
import networkx as nx
from collections.abc import Iterable
from typing import TYPE_CHECKING

from agreed.explainers.tabular.decomposition import (
    get_Psi,
    get_disagreements_from_decomposition,
    get_VIN_clique,
    update_partition,
)
from agreed.explainers.tabular.decomposition import (
    numpy_get_anchored_components_u,
    TabularPartition,
)

if TYPE_CHECKING:
    from agreed.explainers.tabular.decomposition import Decomposition


# def lasso_term(partition, lambd):
#     c = np.array(list(map(lambda x: len(x),partition)))
#     c = c*(c-1)/2
#     return lambd*np.sum(c)


# def GeneratorTabularIGreedy(model: Callable, background: np.ndarray, thresholds: List[float] = [0.1],
#                             verbose: bool = True, lambd: float = 0.001) -> List[List[int]]:
#     """
#     Baseline from paper
#     Xu, S., Cuppers, J., & Vreeken, J. (2024). Succinct Interaction-Aware Explanations. arXiv preprint arXiv:2402.05566.

#     adapted from https://github.com/Schascha1/iSHAP/blob/main/ishap.py"""

#     partition = [(i,) for i in range(background.shape[1])]
#     value_target = model(background)
#     decomposition = {}
#     decomposition[()] = value_target

#     # compute the individual effect of each singleton
#     curr_component_sum = value_target.mean()
#     partition_values = {}
#     for coalition in partition:
#         value_function = numpy_get_anchored_components_u(decomposition, model, coalition, partition, background).mean(1)
#         partition_values[coalition] = value_function
#         curr_component_sum += value_function
#     curr_loss = np.mean((curr_component_sum - value_target)**2)
#     lambd = lambd * curr_loss
#     curr_regul_term = lasso_term(partition, lambd)
#     curr_obj = curr_loss + curr_regul_term

#     # Maintain buffered joint effects to avoid recomputing
#     buffered_merge_values = {}
#     # if interactions_graph is None:
#     # TODO this algorithm can be accelated by using a sparse interaction graph
#     interactions_graph = nx.complete_graph(background.shape[1])
#     for i in range(len(partition)):
#         connections = interactions_graph.edges(i)
#         if len(connections) > 0:
#             buffered_merge_values[(i,)] = {}
#         for node_1,node_2 in connections:
#             other_node = node_1 if node_2 == i else node_2
#             if other_node < i:
#                 continue
#             new_coalition = [node_1, node_2]
#             value_function = numpy_get_anchored_components_u(decomposition, model, (0,), [new_coalition], background).mean(1)
#             buffered_merge_values[(i,)][(other_node,)] = value_function

#     # Merge the coalitions: greedy merging until no more merges are possible either due to size or threshold
#     final_partition = partition.copy()

#     while len(partition) > 1:
#         if verbose:
#             print(f"Objective : {curr_obj:.2e}")

#         best_obj = curr_obj
#         best_merge = None
#         best_component_sum = 0
#         best_loss = 0
#         best_regul_term = 0
#         for set_1 in buffered_merge_values.keys():
#             for set_2 in buffered_merge_values[set_1].keys():
#                 new_component_sum = curr_component_sum - partition_values[set_1] - partition_values[set_2] + buffered_merge_values[set_1][set_2]
#                 new_regul_term = curr_regul_term - lasso_term([set_1,set_2], lambd) + lasso_term([set_1+set_2], lambd)
#                 new_loss = np.mean((value_target - new_component_sum)**2)
#                 new_obj = new_loss + new_regul_term
#                 if new_obj < best_obj:
#                     best_obj = new_obj
#                     best_loss = new_loss
#                     best_regul_term = new_regul_term
#                     best_component_sum = new_component_sum
#                     best_merge = (set_1, set_2)
#         if best_merge is None:
#             break

#         # Update scores and partition
#         curr_obj = best_obj
#         curr_loss = best_loss
#         curr_regul_term = best_regul_term
#         curr_component_sum = best_component_sum
#         new_coalition = best_merge[0] + best_merge[1]
#         partition.remove(best_merge[0])
#         partition.remove(best_merge[1])
#         partition.append(new_coalition)
#         partition_values.pop(best_merge[0], None)
#         partition_values.pop(best_merge[1], None)
#         partition_values[new_coalition] = buffered_merge_values[best_merge[0]][best_merge[1]]
#         final_partition = partition.copy()


#         # update buffered merge values
#         buffered_merge_values.pop(best_merge[0],None)
#         buffered_merge_values.pop(best_merge[1],None)
#         for set_1 in buffered_merge_values.keys():
#             buffered_merge_values[set_1].pop(best_merge[0],None)
#             buffered_merge_values[set_1].pop(best_merge[1],None)
#         for set_1 in buffered_merge_values.keys():
#             compute_merge = False
#             for element in set_1:
#                 edges = interactions_graph.edges(element)
#                 for node_1, node_2 in edges:
#                     other_node = node_1 if node_2 == element else node_2
#                     if other_node in new_coalition:
#                         compute_merge = True
#                         break
#                 if compute_merge:
#                     break
#             if compute_merge:
#                 u = list(new_coalition + set_1)
#                 value_function = numpy_get_anchored_components_u(decomposition, model, (0,), [u], background).mean(1)
#                 buffered_merge_values[set_1][new_coalition] = value_function

#         buffered_merge_values[new_coalition] = {}

#         yield [list(p) for p in final_partition]


# def GeneratorTabularPairwise(model: Callable, background: np.ndarray, thresholds: List[float] = [0.1], verbose: bool = True) -> List[List[int]]:
#     # Setup
#     N, d = background.shape
#     feature_groups = [[i] for i in range(d)]
#     main_effects = [(i,) for i in range(d)]
#     pairwise_interactions = [u for u in combinations(range(d), 2)]

#     # Compute all pair-wise interactions
#     decomposition = {}
#     decomposition[()] = model(background)
#     # We start by computing the main effects
#     for key in main_effects:
#         decomposition[key] = numpy_get_anchored_components_u(decomposition, model, key, feature_groups, background)
#     # We then compute pairwise interactions
#     for key in tqdm(pairwise_interactions, desc="Computing all pair-wise interactions", disable= not verbose):
#         decomposition[key] = numpy_get_anchored_components_u(decomposition, model, key, feature_groups, background)

#     for threshold in thresholds:
#         print(threshold)
#         # Only keep the significant interactions
#         significant_interactions = []
#         model_var = decomposition[()].var()
#         for key in pairwise_interactions:
#             if decomposition[key].mean(0).var() + decomposition[key].mean(1).var() > 2 * threshold * model_var:
#                 significant_interactions.append(key)
#         yield get_VIN_clique(significant_interactions, d)


# def RecurseInteract(G_1, G_2, model, background, threshold, null_component):
#     Imap_inv = [G_1, G_2]
#     decomposition = {}
#     decomposition[()] = null_component
#     for key in [(0,), (1,), (0, 1)]:
#         decomposition[key] = numpy_get_anchored_components_u(decomposition, model, key, Imap_inv, background)
#     var = decomposition[()].var()
#     interaction_strength = 0.5 * (decomposition[(0, 1)].mean(1).var() + decomposition[(0, 1)].mean(0).var())
#     if interaction_strength > threshold * var:
#         N_2 = len(G_2)
#         if N_2 == 1:
#             G_1 = G_1 + G_2
#         else:
#             G_1_fused_left  = RecurseInteract(G_1, G_2[:N_2//2], model, background, threshold, null_component)
#             G_1_fused_right = RecurseInteract(G_1, G_2[N_2//2:], model, background, threshold, null_component)
#             G_1 = set(G_1_fused_left) | set(G_1_fused_right)
#             G_1 = list(G_1)
#     return G_1

# def GeneratorTabularRecurse(model: Callable, background: np.ndarray, thresholds: List[float] = [0.1]) -> List[List[int]]:
#     """
#     TODO
#     """
#     # Setup
#     N, d = background.shape
#     decomposition = None
#     additive_keys = [(i,) for i in range(d)]
#     Imap_inv = [[i] for i in range(d)]

#     for threshold in thresholds:
#         print(threshold)
#         if decomposition is None:
#             # We start by computing the main effects
#             decomposition = {}
#             decomposition[()] = model(background)
#             for key in additive_keys:
#                 decomposition[key] = numpy_get_anchored_components_u(decomposition, model, key, Imap_inv, background)

#         # Order variables relative to their potential Psi
#         Psi = get_Psi(decomposition)
#         feature_order = list(np.argsort(-Psi))
#         Imap_inv_new = []

#         G_1 = feature_order[0:1]
#         G_2 = feature_order[1:]

#         while len(G_2) > 0:

#             G_1_new = RecurseInteract(G_1, G_2, model, background, threshold, decomposition[()])

#             # G_1 does not interact with any features in G_2
#             if sorted(G_1) == sorted(G_1_new):
#                 Imap_inv_new.append(G_1_new)
#                 G_1 = G_2[0:1]
#                 G_2 = G_2[1:]
#             # G_1 interacts with some features in G_2 so we expand G_1
#             else:
#                 G_1 = G_1_new
#                 G_2 = [g for g in G_2 if not g in G_1]

#             # At the end, we put the remainder features in a group
#             if len(G_2) == 0:
#                 Imap_inv_new.append(G_1)

#         yield Imap_inv_new


def GeneratorTabularAGREED(
    model: Callable,
    background: np.ndarray,
    thresholds: tuple[float] = (0.1,),
    init_decomposition: Decomposition | None = None,
    limit_candidates: bool = False,
    verbose: bool = True,
) -> Iterable[TabularPartition]:
    """
    Adaptive Grouping to ReducE Explaination Disagreements:
    This approach will iteratively group features while dynamically
    keeping track of all relevant between-groups interactions.
    This will not compute the O(d^2) interactions between all features
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

    if init_decomposition is None:
        decomposition[()] = model(background)
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

    for threshold in thresholds:

        # Main loop, we iteratively group patches until the gap error falls bellow a threshold
        while disagreement > threshold * init_disagreement:

            if verbose:
                print(f"Disagreement : {disagreement: .2e}")

            # identify the patch that interact most with any other
            Psi = get_Psi(decomposition)
            candidate = int(np.argmax(Psi))
            if limit_candidates:
                interaction_candidates = [
                    (int(k), candidate)
                    for k in np.where(Psi > 0.01)[0]
                    if not k == candidate
                ]
            else:
                interaction_candidates = [
                    (int(k), candidate)
                    for k in range(curr_partition.D)
                    if not k == candidate
                ]

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

            # Recompute the disagreements
            disagreement = get_disagreements_from_decomposition(
                decomposition, metric="l2"
            )

            if verbose:
                print(f"Disagreement : {disagreement: .2e}")

        yield curr_partition
