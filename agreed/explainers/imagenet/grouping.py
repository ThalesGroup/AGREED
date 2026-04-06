"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Fun the AGREED algorithm on ImageNet Classifiers

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

import torch
from torchvision import transforms
import numpy as np
from math import sqrt
from skimage.segmentation import find_boundaries

from agreed.explainers.images.decomposition import (
    TorchModel,
    get_disagreements,
    get_Psi,
    update_partition,
    get_arch_occ_from_decomposition,
)
from agreed.explainers.imagenet.decomposition import compute_anchored_decomposition

if TYPE_CHECKING:
    from agreed.explainers.images.decomposition import Decomposition, ImagePartition


# Assume we want to fuse patch i with another patch. If there exists a pixel from another
# group which is connected to MAX_CONNECTIVITY or more points from group i, then we
# are forced to fuse group i with said pixel.
# For example, let MAX_CONNECTIVITY=6 and let us fuse group 1 with another.
#    1 1 1
#    1 3 2
#    1 1 2
# Regardless of interactions with group 2, group 1 will be fused with group 3 since this
# pixel is connected to 6 pixels from group 1. This constraint ensures that patche shapes
# do not degenerate.
MAX_CONNECTIVITY = 6
Integer_Mask = np.ndarray[int, np.dtype[np.int32]]

def setup_image_agreed(x: torch.Tensor, z: torch.Tensor):
    """
    Flatten the images into (1, n_channels, -1) to make
    it easier to replace groups of pixels.
    """
    _, n_channels, w, _ = x.shape
    x = x.view(1, n_channels, -1)
    z = z.view(1, n_channels, -1)
    return x, z


def get_boundary_and_connect(
    Imap: Integer_Mask, candidate: int
) -> tuple[Integer_Mask, Integer_Mask]:
    boolean_image = Imap == candidate
    boundary = find_boundaries(boolean_image, mode="outer")
    # Find the connectivity between the group and the border
    connectivity = np.zeros(Imap.shape, dtype=np.int32)
    connectivity[:, 1:] += boolean_image[:, :-1]
    connectivity[:, :-1] += boolean_image[:, 1:]
    connectivity[1:, :] += boolean_image[:-1, :]
    connectivity[:-1, :] += boolean_image[1:, :]
    connectivity[1:, 1:] += boolean_image[:-1, :-1]
    connectivity[1:, :-1] += boolean_image[:-1, 1:]
    connectivity[:-1, 1:] += boolean_image[1:, :-1]
    connectivity[:-1, :-1] += boolean_image[1:, 1:]
    return boundary, connectivity


def get_interaction_candidates(
    decomposition: Decomposition,
    partition: ImagePartition,
    init_patch_size: int,
    max_patch_size: int | None = None,
) -> list[tuple[int, int]]:
    """
    Return a list of pairwise interactions to investigate while respecting the various
    partition hard constraints. If no interaction matches the constraints,
    an empty list is returned.
    """

    h, w = partition.Imap.shape
    if max_patch_size is None:
        max_patch_size = h * w

    # Identify the patch that interact most with any other
    Psi = get_Psi(decomposition)

    # Tag groups that should not be extended
    group_sizes = torch.tensor([len(group) for group in partition.Imap_inv])
    groups_to_ignore = torch.where(group_sizes > max_patch_size)[0]
    Psi[groups_to_ignore] = -torch.inf

    # Main loop, try each candidate and stop if they can be fused with at least another group
    candidate = int(torch.argmax(Psi).item())
    while Psi[candidate] >= 0.0:
        Psi[candidate] = -torch.inf

        # Use one pixel per initial patch
        Imap_smaller = partition.Imap[::init_patch_size, ::init_patch_size]
        boundary, connectivity = get_boundary_and_connect(Imap_smaller, candidate)
        # Max connectivity is between 1 and 8
        max_connectivity_boundary = connectivity[boundary].max()

        # If a boundary pixel has a large connectivity, we MUST fuse it
        if max_connectivity_boundary > MAX_CONNECTIVITY:
            boundary_groups = list(np.unique(
                Imap_smaller[boundary & (connectivity == max_connectivity_boundary)]
                )
            )
            interaction_candidates = [
                (candidate, k)
                for k in boundary_groups
                if len(partition.Imap_inv[candidate]) + len(partition.Imap_inv[k]) < max_patch_size
            ]
            if len(interaction_candidates) > 0:
                return interaction_candidates
        # If we could not fuse with a large connectivity group, or no such group
        # exists, we simply try to fuse with all neighbors
        boundary_groups = list(np.unique(Imap_smaller[boundary]))
        interaction_candidates = [
            (candidate, k)
            for k in boundary_groups
            if len(partition.Imap_inv[candidate]) + len(partition.Imap_inv[k]) < max_patch_size
        ]
        if len(interaction_candidates) > 0:
            return interaction_candidates
        candidate = int(torch.argmax(Psi).item())
    return []


def ImagenetAGREED(
    model: TorchModel,
    x: torch.Tensor,
    z: torch.Tensor,
    thresholds: tuple[float, ...],
    partition: ImagePartition,
    max_patch_size: int | None = None,
    verbose: bool = False,
) -> Iterable[ tuple[ ImagePartition, tuple[torch.Tensor, torch.Tensor] ] ]:
    """
    Adaptive Grouping to ReducE Explanation Disagreements:
    This approach iteratively group features while dynamically
    keeping track of all relevant between-groups interactions.
    Early convergence implies that this algorithm will not compute
    the O(d^2) interactions between all features

    Parameters
    ----------
    model: TorchModel
        A wrapper to a torch model that returns a singl output (e.g. logit for class c)
    x: torch.Tensor
        The (1, C, H, W) input image to explain.
    z: torch.Tensor
        The (1, C, H, W) image used as a reference.
    thresholds: tuple[float, ...]
        The algorithm iterates over the values of thresholds. For each value, AGREED
        runs until the disagreement falls bellow said threshold. This allows to run
        AGREED with different threshold while reusing previous computations.
    partition: ImagePartition
        The initial partition of the image. AGREED will fuse these initial patches
        to get the final ones.
    max_patch_size: int | None, default=None
        Only consider partitions where each patch has at most this number of pixels.
    verbose: bool, default=False
        Print the disagreements during the runtime of AGREED.

    Returns
    ---------
    partition: ImagePartition
        The partition of the image i.e. the patches of pixels
    (arch, occ): tuple[ torch.Tensor, torch.Tensor ]
        The ArchAttribute and Occlusion saliency map come for-free after finding
        the partition.
    """
    # Setup
    x, z = setup_image_agreed(x, z)

    if verbose:
        print("\n=================================================")
        print("Running AGREED with settings:")
        print(f"    thresholds: {thresholds}")
        print(f"    max_patch_size: {max_patch_size}")

    assert ( (np.array(thresholds) > 0) & (np.array(thresholds) <= 1.0) ).all(),\
                                            "Thresholds must be between 0.0 and 1.0 "
    interactions = [()] + [(i,) for i in range(partition.D)]
    init_patch_size = int(sqrt(len(partition.Imap_inv[0])))
    decomposition: Decomposition = {}
    for key in interactions:
        decomposition[key] = torch.zeros(2, 2)

    # The main effects
    compute_anchored_decomposition(
        model,
        x,
        z,
        partition,
        interactions,
        decomposition,
    )
    disagreement = get_disagreements(decomposition, partition, metric="l2")
    init_disagreement = disagreement
    curr_partition = partition

    if verbose:
        print(f"Init Disagreement : {init_disagreement: .2e}")

    # Outer loop: we try different thresholds while reusing previous computations
    for threshold in thresholds:

        # Inner loop, we iteratively group patches until loss falls bellow the threshold
        while disagreement >= threshold * init_disagreement:

            interaction_candidates = get_interaction_candidates(
                decomposition,
                curr_partition,
                init_patch_size,
                max_patch_size
            )

            # Early stopping when no more groups can be fused while respecting contraints
            if len(interaction_candidates) == 0:
                break

            # Compute interaction with the boundary patches
            compute_anchored_decomposition(
                model,
                x,
                z,
                curr_partition,
                interaction_candidates,
                decomposition,
            )
            interaction_strength = []
            for u in interaction_candidates:
                interaction_strength.append(
                    torch.abs(decomposition[u][0, 1]) + torch.abs(decomposition[u][1, 0])
                )
            groups_to_fuse = interaction_candidates[
                torch.argmax(torch.Tensor(interaction_strength))
            ]

            # Fuse two patches into one
            decomposition, curr_partition = update_partition(
                groups_to_fuse, decomposition, curr_partition
            )

            # Recompute the Gap error
            disagreement = get_disagreements(decomposition, curr_partition, metric="l2")

            if verbose:
                print(f"Disagreement : {disagreement: .2e}")

        if verbose:
            if len(interaction_candidates) == 0:
                print("Exited because no more groups could be fused")
            else:
                print(f"Exited because loss fell bellow threshold: {threshold:.2e}")
        yield curr_partition, get_arch_occ_from_decomposition(decomposition, curr_partition)

    print("=================================================\n")
