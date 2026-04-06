"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Saliency map for torch CNNs

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from torch.autograd import grad
from tqdm import tqdm
import numpy as np
from math import ceil
from sklearn.linear_model import Ridge

from agreed.explainers.images.decomposition import TorchModel
from agreed.explainers.utils import ravel


if TYPE_CHECKING:
    from agreed.explainers.images.grouping import ImagePartition



def setup_attribution_methods(
    x: torch.Tensor, z: torch.Tensor, partition: ImagePartition
):
    """
    This function will flatten the x, z into (N, 3, -1) to make
    it easier to apply masks based on pixel indices
    """
    _, n_channels, h, w = x.shape
    x = x.view(1, n_channels, -1)
    z = z.view(1, n_channels, -1)
    attributions = torch.zeros(1, partition.D)
    return x, z, attributions, n_channels, w


def get_fidelity(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        attributions: torch.Tensor,
        partition: ImagePartition,
        ):

    # Setup
    assert x.shape[0] == 1, "Fidelity only supports single images"
    attributions = attributions[0]    # (D,)
    x, z, _ , n_channels, w = setup_attribution_methods(x, z, partition)
    n_perturbations = 200
    fidelity_data = torch.zeros(n_perturbations, 2)

    # Create the patched inputs
    queries = torch.tile(x[[0]], (n_perturbations+1, 1, 1))
    masks = torch.randint(0, 2, size=(n_perturbations, partition.D))
    for p in range(n_perturbations):
        for j, pixel_group in enumerate(partition.Imap_inv):
            if masks[p][j] == 1:
                queries[p+1, :, pixel_group] = z[0, :, pixel_group]
        # Sum the attribution over the masked patches
        fidelity_data[p, 0] = attributions[masks[p].to(torch.bool)].sum()
    queries = queries.view(n_perturbations+1, n_channels, w, w)
    # Get predictions over all queries
    outputs = model(queries)
    fidelity_data[:, 1] = outputs[0] - outputs[1:]
    fidelity = torch.mean((fidelity_data[:, 0] - fidelity_data[:, 1])**2)
    return fidelity



def get_saliency_from_attributions(attribution, partition):
    h, w = partition.Imap.shape
    saliency = torch.zeros(h * w)
    for j, pixel_group in enumerate(partition.Imap_inv):
        saliency[pixel_group] = attribution[j]
    return saliency.view(h, w)



def IGInputOutput(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        batch_size: int = 50,
        quadrature_steps: int = 100,
        sg_steps = 1,
        device = torch.device("cpu")
        ):
    """
    Compute the Input-Ouput Integrated Gradient over groups of pixels
    """
    # Init tensors
    _, n_channels, w, _ = x.shape
    attribution_individual = torch.zeros(sg_steps, w, w).to(device)
    attributions = torch.zeros(1, partition.D)

    # Apply SG to smooth out the gradients even more
    if sg_steps > 1:
        x = x.tile((sg_steps, 1, 1, 1))
        x += 0.04 * torch.rand(x.shape) - 0.02
    delta = x - z
    delta.to(device)

    for j in tqdm(range(sg_steps)):
        avg_input_grad = torch.zeros(3, w, w)
        t = torch.linspace(0, 1, quadrature_steps).reshape((-1, 1, 1, 1))  # (steps, 1, 1, 1)
        x = x[j].unsqueeze(0)                         # (1, 3, w, w)
        queries = (1 - t) * x + t * z                 # (steps, 3, w, w)
        for i in range(ceil(quadrature_steps / batch_size)):
            X = queries[i*batch_size:(i+1)*batch_size]
            X.requires_grad_()
            output = model(X.to(device), keep_gradients=True)
            avg_input_grad += grad(outputs=output.sum(), inputs=X, create_graph=False)[0].sum(dim=0)
        # Here we already sum the importance over the three RGB channels
        attribution_individual[j] = torch.sum(avg_input_grad / quadrature_steps * delta[j], dim=0 )# (w, w)
    attribution_individual = torch.mean(torch.reshape(attribution_individual, (sg_steps, -1, w*w)), dim=0).cpu()

    # aggregate attributions over patches
    for i, pixel_group in enumerate(partition.Imap_inv):
        attributions[:, i] = attribution_individual[:, pixel_group].sum(dim=1)
    return attributions


def ArchAttributeInputOutput(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        batch_size: int = 50,
        device = torch.device("cpu")
        ):
    """
    Computes the ArchAttribute attribution from the input to the logit
    """
    # Setup
    x, z, attributions, n_channels, w = setup_attribution_methods(x, z, partition)

    # Create the patched inputs
    queries = torch.tile(z[[0]], (partition.D + 1, 1, 1))
    for j, pixel_group in enumerate(partition.Imap_inv):
        queries[j + 1, :, pixel_group] = x[0, :, pixel_group]
    queries = queries.view(partition.D+1, n_channels, w, w)
    # Get predictions over all queries
    outputs = model(queries)
    # Put joint contribution in the attribution array
    attributions[0, :] = outputs[1:] - outputs[0]
    return attributions


def OcclusionInputOutput(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        batch_size: int = 50,
        device = torch.device("cpu")
        ):
    """
    Computes the Occlusion attribution from the input to the logit
    """
    # Setup
    x, z, attributions, n_channels, w = setup_attribution_methods(x, z, partition)

    # Create the patched inputs
    queries = torch.tile(x[[0]], (partition.D + 1, 1, 1))
    for j, pixel_group in enumerate(partition.Imap_inv):
        queries[j + 1, :, pixel_group] = z[0, :, pixel_group]
    queries = queries.view(partition.D+1, n_channels, w, w)
    # Get predictions over all queries
    outputs = model(queries)
    # Put joint contribution in the attribution array
    attributions[0, :] = outputs[0] - outputs[1:]
    return attributions


def RISEInputOutput(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        batch_size: int = 50,
        n_perturbations: int = 1000,
        device = torch.device("cpu")
        ):
    """
    Computes the RISE attribution from the input to the logit
    """

    # Setup
    x, z, attributions, n_channels, w = setup_attribution_methods(x, z, partition)

    # Create the patched inputs
    queries = torch.tile(x[[0]], (n_perturbations, 1, 1))
    masks = torch.randint(0, 2, size=(n_perturbations, partition.D))
    for p in range(n_perturbations):
        for j, pixel_group in enumerate(partition.Imap_inv):
            if masks[p][j] == 0:
                queries[p, :, pixel_group] = z[0, :, pixel_group]
    queries = queries.view(n_perturbations, n_channels, w, w)
    # Get predictions over all queries
    outputs = model(queries)
    # Put joint contribution in the attribution array
    for j in range(partition.D):
        attributions[0, j] -= torch.mean(outputs[masks[:, j]==0])
        attributions[0, j] += torch.mean(outputs[masks[:, j]==1])
    return attributions



def LIMEInputOutput(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        batch_size: int = 50,
        n_perturbations: int = 1000,
        device = torch.device("cpu")
        ):
    """
    Computes the LIME attribution to explain the input-output function
    """

    # Setup
    x, z, attributions, n_channels, w = setup_attribution_methods(x, z, partition)

    # Create the patched inputs
    queries = torch.tile(x[[0]], (n_perturbations, 1, 1))
    masks = torch.randint(0, 2, size=(n_perturbations, partition.D))
    for p in range(n_perturbations):
        for j, pixel_group in enumerate(partition.Imap_inv):
            if masks[p][j] == 0:
                queries[p, :, pixel_group] = z[0, :, pixel_group]
    queries = queries.view(n_perturbations, n_channels, w, w)
    # Get predictions over all queries
    outputs = model(queries)
    # Fit a local linear model (a.k.a LIME)
    weights = Ridge().fit(masks.numpy(), outputs.numpy()).coef_
    # Put joint contribution in the attribution array
    attributions[0, :] = torch.Tensor(weights)
    return attributions



def get_shap_weights(n_permutations, d, reversed):
    if reversed:
        def permutation_generator(M, D):
            assert M%2==0, "Reversed Permutations require an even M"
            for _ in range(M//2):
                perm = np.random.permutation(D)
                yield perm
                yield perm[-1::-1]
    else:
        def permutation_generator(M, D):
            for _ in range(M):
                perm = np.random.permutation(D)
                yield perm
    # Compute the W matrix for SHAP value computations
    all_sets = dict({() : np.zeros(d)})
    # Go though all permutations to store all required sets
    for perm in permutation_generator(n_permutations, d):
        curr_set = []
        prev_tuple = None

        # Add the effect of the null coallition
        all_sets[()][perm[0]] -= 1

        # Going along the permutation
        for i in perm:
            curr_set.append(i)
            curr_set.sort()
            curr_tuple = tuple(curr_set)
            # Check if this curr_tuple has previously been used as key
            if curr_tuple not in all_sets:
                all_sets[curr_tuple] = torch.zeros(d)
            all_sets[curr_tuple][i] += 1
            if prev_tuple is not None:
                all_sets[prev_tuple][i] -= 1
            # Keep track of the previous tuple
            prev_tuple = curr_tuple

    # All sets S on which we will evaluate nu(S)
    W = torch.Tensor(np.column_stack(list(all_sets.values())) / n_permutations)
    interactions = list(all_sets.keys())
    return W, interactions



def SHAPInputOutput(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        batch_size: int = 50,
        n_permutations: int = 100,
        reversed: bool = True,
        device = torch.device("cpu"),
        ):
    # Setup
    x, z, attributions, n_channels, w = setup_attribution_methods(x, z, partition)

    # The SHAP weights used to aggregate marginal contributions
    W, interactions = get_shap_weights(n_permutations, partition.D, reversed)
    S = len(interactions)
    print(f"Considering {S} feature subsets")

    # Setup arrays
    nu = torch.zeros((S, x.shape[0]))
    # Create the patched inputs
    queries = torch.tile(z, (S, 1, 1))
    for s, interaction in enumerate(interactions):
        if len(interaction) == 0:
            continue
        super_pixel_group = ravel([partition.Imap_inv[i] for i in interaction])
        queries[s, :, super_pixel_group] = x[0, :, super_pixel_group]
    queries = queries.view(S, n_channels, w, w)
    # Get predictions over all queries
    nu[:, 0] = model(queries)
    attributions = torch.matmul(W, nu).T # (d, N)
    return attributions



def get_all_attributions(
        model: TorchModel,
        x: torch.Tensor,
        z: torch.Tensor,
        partition: ImagePartition,
        n_perturbations: int = 1000,
        n_permutations: int = 10,
        ) -> dict[str, torch.Tensor]:

    attributions: dict[str, torch.Tensor] = {}

    # Compute individual feature importance
    attributions["Arch"] = ArchAttributeInputOutput(
        model,
        x,
        z,
        partition,
    )
    attributions["Occ"] = OcclusionInputOutput(
        model,
        x,
        z,
        partition,
    )
    attributions["IG"] = IGInputOutput(
        model,
        x,
        z,
        partition,
        quadrature_steps=2000
    )
    attributions["RISE"] = RISEInputOutput(
        model,
        x,
        z,
        partition,
        n_perturbations=n_perturbations
    )
    attributions["LIME"] = LIMEInputOutput(
        model,
        x,
        z,
        partition,
        n_perturbations=n_perturbations
    )
    attributions["SHAP"] = SHAPInputOutput(
        model,
        x,
        z,
        partition,
        n_permutations=n_permutations
    )
    return attributions
