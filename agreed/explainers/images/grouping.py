"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import numpy as np
import torch
from tqdm import tqdm
from skimage.segmentation import find_boundaries

from itertools import combinations

from agreed.explainers.images.decomposition import (
    get_arch_occ_from_decomposition,
    get_disagreements,
    get_Psi,
    get_VIN_clique,
    get_pixel_groups_from_clique,
)
from agreed.explainers.images.decomposition import (
    torch_get_anchored_components_u,
    TorchModel,
    ImagePartition,
    update_partition,
)

if TYPE_CHECKING:
    from agreed.explainers.images.decomposition import Decomposition


def ImagePairwise(
    model: TorchModel,
    background: torch.Tensor,
    init_partition: ImagePartition,
    thresholds: tuple[float] = (0.1,),
) -> Iterable[tuple[ImagePartition, tuple[torch.Tensor|None, torch.Tensor|None]]]:

    # Setup
    decomposition: Decomposition = {}
    decomposition[()] = model(background)

    # The background becomes (N, n_channels, w*w)
    w = background.shape[-1]
    background = setup_image_agreed(background)
    key: tuple[int, ...]

    def iterator_():
        for cardinality in [1, 2]:
            for key in tqdm(combinations(range(init_partition.D), r=cardinality)):
                yield key

    # We then compute pairwise interactions
    for key in tqdm(iterator_(), desc="All pairwise interactions"):
        decomposition[key] = torch_get_anchored_components_u(
            model, background, init_partition, key, decomposition
        )

    model_var = decomposition[()].var()
    for threshold in thresholds:

        print(threshold)

        # Only keep the significant interactions
        significant_interactions: list[list[int]] = []
        for key in combinations(range(init_partition.D), r=2):
            if (
                decomposition[key].mean(0).var() + decomposition[key].mean(1).var()
                > 2 * threshold * model_var
            ):
                significant_interactions.append(list(key))
        # Return the clique of the pairwise interaction graph
        cliques = get_VIN_clique(significant_interactions, init_partition.D)

        new_Imap_inv = get_pixel_groups_from_clique(cliques, init_partition.Imap_inv)
        new_Imap = np.zeros_like(init_partition.Imap).ravel()
        for j, pixel_group in enumerate(new_Imap_inv):
            new_Imap[pixel_group] = j
        new_Imap = new_Imap.reshape(w, w)
        new_partition = ImagePartition(
            Imap=new_Imap, Imap_inv=new_Imap_inv, D=len(new_Imap_inv)
        )
        yield new_partition, (None, None)


def setup_image_agreed(background: torch.Tensor):
    """
    This function will flatten the background into (N, n_channels, -1) to make
    it easier to replace groups of pixels.
    """
    N, n_channels, w, _ = background.shape
    # Reshape background
    background = background.view(N, n_channels, -1)
    return background


def get_interaction_candidates(candidate: int, partition: ImagePartition):
    boolean_image = partition.Imap == candidate
    boundaries = find_boundaries(boolean_image, mode="outer")
    boundary_groups = list(np.unique(partition.Imap[boundaries]))
    return [(candidate, k) for k in boundary_groups if not k == -1]


def ImageAGREED(
    model: TorchModel,
    background: torch.Tensor,
    init_partition: ImagePartition,
    thresholds: tuple[float, ...] = (0.1,),
):
    """
    Adaptive Grouping to ReducE Explaination Disagreements: This approach
    iteratively groups features while dynamically keeping track of all relevant
    between-groups interactions. This will not compute the O(d^2) interactions
    between all features
    """
    # Setup
    decomposition: Decomposition = {}
    decomposition[()] = model(background)
    additive_keys = [(i,) for i in range(init_partition.D)]

    # The background becomes (N, n_channels, w*w)
    background = setup_image_agreed(background)

    # We start by computing the main effects for all pixel groups
    for key in tqdm(additive_keys, desc="Main Effects"):
        decomposition[key] = torch_get_anchored_components_u(
            model, background, init_partition, key, decomposition
        )
    # Disagreements are computed to measure our progress
    disagreement = get_disagreements(decomposition, init_partition, metric="l2")
    init_disagreement = disagreement
    curr_partition = init_partition

    for threshold in thresholds:
        print(threshold)
        # Main loop, we iteratively group patches until disagreements fall bellow a threshold
        while disagreement > threshold * init_disagreement:

            print(f"Disagreement : {disagreement: .2e}")

            # identify the patch that interact most with any other
            Psi = get_Psi(decomposition)
            candidate = int(torch.argmax(Psi).item())
            interaction_candidates = get_interaction_candidates(
                candidate, curr_partition
            )

            # Compute interaction with all other patches
            interaction_strength = []
            for key in interaction_candidates:
                decomposition[key] = torch_get_anchored_components_u(
                    model, background, curr_partition, key, decomposition
                )
                interaction_strength.append(
                    torch.abs(decomposition[key].mean(1)).mean()
                    + torch.abs(decomposition[key].mean(0)).mean()
                )
            groups_to_fuse = interaction_candidates[
                torch.argmax(torch.tensor(interaction_strength))
            ]

            # Fuse two patches into one
            decomposition, curr_partition = update_partition(
                groups_to_fuse, decomposition, curr_partition
            )

            # Recompute the Gap error
            disagreement = get_disagreements(decomposition, curr_partition, metric="l2")
        print(f"Disagreement : {disagreement: .2e}")
        yield curr_partition, get_arch_occ_from_decomposition(
            decomposition, curr_partition
        )
