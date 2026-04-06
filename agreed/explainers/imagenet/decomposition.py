"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Some functional decomposition functions specific to how AGREED runs on ImageNet

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from math import sqrt

from agreed.explainers.utils import powerset

if TYPE_CHECKING:
    from agreed.explainers.images.decomposition import Decomposition, ImagePartition
    from agreed.explainers.images.decomposition import TorchModel
    from collections.abc import Sequence



def compute_anchored_decomposition(
    model: TorchModel,
    x: torch.Tensor,
    z: torch.Tensor,
    partition: ImagePartition,
    interactions: Sequence[ tuple[int, ...] ],
    decomposition: Decomposition,
):
    """
    Compute the anchored functional decomposition (stored in 2x2 matrices). This
    will modify the argument `decomposition` in-place by adding new keys related
    to the passed `interactions`.
    """

    # TODO make this work for rectangle images.
    _, n_channels, w_squared = x.shape
    w = int(sqrt(w_squared))

    queries_zref = torch.tile(z, (len(interactions), 1, 1))
    queries_xref = torch.tile(x, (len(interactions), 1, 1))
    for i, u in enumerate(interactions):
        for j in u:
            queries_zref[i, :, partition.Imap_inv[j]] = x[0, :, partition.Imap_inv[j]]
            queries_xref[i, :, partition.Imap_inv[j]] = z[0, :, partition.Imap_inv[j]]
    queries = torch.cat((queries_zref, queries_xref), dim=0).view(
        2 * len(interactions), n_channels, w, w
    )
    # Get predictions over all queries
    outputs = model(queries)

    # Compute the interactions
    for i, u in enumerate(interactions):
        decomposition[u] = torch.zeros(2, 2)
        decomposition[u][0, 1] = outputs[i]
        decomposition[u][1, 0] = outputs[i + len(interactions)]
        for v in powerset(u):
            decomposition[u][0, 1] -= decomposition[v][0, 1]
            decomposition[u][1, 0] -= decomposition[v][1, 0]
