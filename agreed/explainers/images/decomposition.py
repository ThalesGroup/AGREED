"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Utility functions to compute partitions and decompositions

from typing import Callable
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
import numpy as np
from math import sqrt, ceil
from dataclasses import dataclass
from copy import deepcopy

from agreed.explainers.utils import ravel, powerset

# A decomposition maps an element u of the lattice space to an array storing
# the anchored decomposition h_u. For images, each element is a 2x2 matrix whose
# non-diagonal elements are the anchored decomposition at z (resp x) evaluated
# at x (resp z)
Decomposition = dict[tuple[int, ...], torch.Tensor]

@dataclass
class ImagePartition(object):
    Imap: np.ndarray            # (w, h) array that maps every pixel to its group index
    Imap_inv: list[ list[int] ] # Imap_inv[i] are pixels in the flat image from group i
    D: int                      # The number of groups


def get_baseline(
        baseline: str,
        im_size: tuple[int, int],
        n_channels: int,
        x: torch.Tensor | None = None,
        model: nn.Module | None = None,
        device: str | None = None,
        label: int | None = None
        ) -> torch.Tensor:
    # Get the baseline
    if baseline == "zero":
        z = torch.zeros((1,) + (n_channels,) + im_size)
    elif baseline == "mean":
        assert x is not None
        z = torch.zeros((1,) + (n_channels,) + im_size)
        for c in range(n_channels):
            z[:, c, ...] = x[0, c, ...].mean(-1).mean(-1)
    elif baseline == "blur":
        assert x is not None
        assert model is not None
        assert device is not None
        assert label is not None
        # Blur until the classification changes
        # TODO avoid hard coded values
        for kernel_size in range(33, 223, 30):
            z = gaussian_blur(x, kernel_size=kernel_size, sigma=None)
            with torch.no_grad():
                logits = model(z.to(device)).cpu()
            prediction = int(torch.argmax(logits, dim=1)[0])
            if not prediction == label:
                break
    else:
        raise Exception("Invalid value for baseline")
    return z


def get_class_logit_margin(
    batch: torch.Tensor,
    model: nn.Module,
    batch_size: int,
    device: torch.device | str,
    class_1: int,
    class_0: int | None = None,
    keep_gradients: bool = False,
) -> torch.Tensor:
    outputs = []
    for b in range(ceil(batch.shape[0] / batch_size)):
        X = batch[b * batch_size : (b + 1) * batch_size]
        if keep_gradients:
            out = model(X.to(device))
        else:
            with torch.no_grad():
                out = model(X.to(device))
        if class_0 is None:
            out = out[:, class_1]
        else:
            out = out[:, class_1] - out[:, class_0]
        outputs.append(out)
    return torch.cat(outputs, dim=0).cpu()


class TorchModel(object):
    """
    A simple wrapper that accumulates counts of the number of
    samples on which the model was run. Moreover, it reports the logit margin
    between two classes chosen beforehand. Thus, this model is applicable
    to explanations of predictions at a fixed input x.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device | str,
        class_1: int,
        class_0: int | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.class_1 = class_1
        self.class_0 = class_0
        self.n_calls = 0

    def reset_n_calls(self):
        self.n_calls = 0

    def __call__(self, X: torch.Tensor, keep_gradients: bool = False):
        N = X.shape[0]
        self.n_calls += N
        return get_class_logit_margin(
            X,
            self.model,
            self.batch_size,
            self.device,
            self.class_1,
            self.class_0,
            keep_gradients=keep_gradients,
        )


def torch_get_anchored_components_u(
    model: Callable,
    background: torch.Tensor,
    partition: ImagePartition,
    key: tuple[int, ...],
    decomposition: Decomposition,
):
    N, n_channels, w_squared = background.shape
    w = int(sqrt(w_squared))
    result = torch.zeros((N, N))
    data_temp = deepcopy(background)
    U = ravel([partition.Imap_inv[j] for j in key])
    # Compute h_{u,k}(x) for all x to eval
    for i in range(N):
        # Annihilation property
        annihilated = torch.zeros((N,))
        for j in key:
            annihilated += (
                (
                    background[i, :, partition.Imap_inv[j]]
                    == background[..., partition.Imap_inv[j]]
                )
                .all(dim=1)
                .all(dim=1)
            )
        not_annihilated = torch.where(annihilated == 0)[0]

        # Compute the component using all anchors that do not annihilate
        if len(not_annihilated) > 0:
            # Broadcasting dark magic happens here
            data_temp[
                not_annihilated.view(-1, 1, 1),
                torch.arange(n_channels).view(1, -1, 1),
                U,
            ] = background[i, :, U]
            queries = data_temp[not_annihilated].view(-1, n_channels, w, w)
            h_replace = model(queries)
            result[i, not_annihilated] += h_replace
            # Remove all contributions of subsets to get the interaction
            for subset in powerset(key):
                if subset not in decomposition:
                    raise Exception(
                        "The provided interaction set is not closed downward"
                    )
                if len(subset) > 0:
                    result[i, not_annihilated] -= decomposition[subset][
                        i, not_annihilated
                    ]
                else:
                    result[i, not_annihilated] -= decomposition[subset][not_annihilated]
    return result


def get_Psi(decomposition: Decomposition) -> torch.Tensor:
    Psi  = []
    additive_keys = [key for key in decomposition.keys() if len(key) == 1]
    for key in additive_keys:
        Psi.append(
            torch.abs(decomposition[key].mean(1) + decomposition[key].mean(0)).mean()
        )
    return torch.tensor(Psi)


def get_avg_patch_size(partition: ImagePartition):
    return np.mean([len(pixel_group) for pixel_group in partition.Imap_inv])


def get_patch_partition(patch_size: int, im_size: tuple[int, int]):
    """
    Define a partition based on (patch_size, patch_size) patches from an image
    of size (h, w).
    """
    n_patches_h = int(np.ceil(im_size[0] / patch_size))
    n_patches_w = int(np.ceil(im_size[1] / patch_size))
    Imap_inv: list[list[int]] = [[] for _ in range(n_patches_h*n_patches_w)]
    curr_idx = 0
    # Create Imap_inv
    for i in range(im_size[0]):
        for j in range(im_size[1]):
            patch_row_idx = int(np.floor(i / patch_size))
            patch_col_idx = int(np.floor(j / patch_size))
            Imap_inv[n_patches_h * patch_row_idx + patch_col_idx].append(curr_idx)
            curr_idx += 1
    # Create Imap
    Imap = np.zeros(im_size[0] * im_size[1], dtype=np.int32)
    for j, pixel_group in enumerate(Imap_inv):
        Imap[pixel_group] = j
    Imap = Imap.reshape(*im_size)
    return ImagePartition(Imap, Imap_inv, len(Imap_inv))


def get_arch_occ_from_decomposition(
    decomposition: Decomposition, partition: ImagePartition
) -> tuple[torch.Tensor, torch.Tensor]:
    attr_1 = torch.zeros(partition.D)
    for i in range(partition.D):
        attr_1[i] = decomposition[(i,)][0, 1]
    attr_2 = torch.zeros(partition.D)
    for i in range(partition.D):
        attr_2[i] = -1 * decomposition[(i,)][1, 0]
    return attr_1, attr_2


def cosinedisagreement(attr_1: torch.Tensor, attr_2: torch.Tensor):
    return 0.5 * float(
        (
            1 - torch.sum(attr_1 * attr_2)
            / (torch.linalg.norm(attr_1) * torch.linalg.norm(attr_2))
        )
    )


def l2disagreement(attr_1: torch.Tensor, attr_2: torch.Tensor):
    return float(torch.linalg.norm(attr_1 - attr_2))


def get_disagreements(decomposition: Decomposition, partition: ImagePartition, metric="l2"):
    attr_1, attr_2 = get_arch_occ_from_decomposition(decomposition, partition)
    if metric == "l2":
        return l2disagreement(attr_1, attr_2)
    else:
        return cosinedisagreement(attr_1, attr_2)



def get_VIN_clique(
    significant_interactions: list[list[int]], d: int
) -> list[list[int]]:
    """Iterate over all pairwise interactions and return the graphe's cliques"""

    cliques = [[i] for i in range(d)]  # The feature cliques
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
        cliques[right_clique_idx] = [-1]
    return [sorted(clique) for clique in cliques if not clique == [-1]]


def get_pixel_groups_from_clique(
    cliques: list[list[int]], init_pixel_groups: list[list[int]]
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
    partition: ImagePartition,
):

    u, v = indices_to_fuse
    # Transform the interaction into a joint main effect for (u,v)
    joint_effect = (
        decomposition[indices_to_fuse] + decomposition[(u,)] + decomposition[(v,)]
    )

    # Create a new decomposition, Imap_inv, and Imap
    new_decomposition: Decomposition = {(): decomposition[()]}
    new_Imap_inv = []
    w = partition.Imap.shape[0]
    new_Imap = -1 * np.ones(w * w, dtype=np.int32)

    c = 0
    for k in range(partition.D):
        if (not k == u) and (not k == v):
            new_decomposition[(c,)] = decomposition[(k,)]
            new_Imap_inv.append(partition.Imap_inv[k])
            new_Imap[partition.Imap_inv[k]] = c
            c += 1
    # Transform the interaction into a joint main effect for (i,j)
    joint_effect = (
        decomposition[indices_to_fuse] + decomposition[(u,)] + decomposition[(v,)]
    )
    new_group = partition.Imap_inv[u] + partition.Imap_inv[v]
    new_decomposition[(c,)] = joint_effect
    new_Imap_inv.append(new_group)
    new_Imap[new_group] = c
    new_Imap = new_Imap.reshape((w, w))
    new_partition = ImagePartition(new_Imap, new_Imap_inv, partition.D - 1)
    return new_decomposition, new_partition
