"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

#Baselines for segmenting miniimagenet images

from typing import Iterable
import torch
from torchvision import transforms
import numpy as np
from skimage.segmentation import slic, quickshift

from agreed.plots import scale_to_original
from agreed.explainers.images.decomposition import ImagePartition, get_patch_partition

# # The patch_sizes parameter of each method (larger param implies larger patches)
# if method == "square":
#     # The semi-length of each patch
#     patch_sizes = [14, 32, 48, 64, 80]
# elif method == "quick":
#     # The max-dist hyperparam -> larger means less patches
#     patch_sizes = [5, 10, 100, 500, 1000]
# else:
#     # The number of patches in K-Means
#     patch_sizes = [200, 100, 50, 25]

def get_segmentation_partition(segmented_image: np.ndarray) -> ImagePartition:
    h, w = segmented_image.shape[-2:]
    D = int(segmented_image.max())
    Imap_inv: list[list[int]] = [[] for _ in range(D)]
    curr_idx = 0
    for i in range(h):
        for j in range(w):
            Imap_inv[segmented_image[i, j]-1].append(curr_idx)
            curr_idx += 1
    return ImagePartition(segmented_image, Imap_inv, D)


def SegmentationGrouping(
    foreground: torch.Tensor,
    imsize: tuple[int, int],
    patch_params: tuple[int, ...],
    image_transform: transforms.Compose,
    method: str = "square"
) -> Iterable[ tuple[ ImagePartition, None] ]:
    """ Use simple image segmentation algortihms as a baseline """

    for patch_param in patch_params:
        if method == "square":
            # Get patch
            partition = get_patch_partition(patch_param, imsize)
        elif method == "quick":
            # Get patch
            image = scale_to_original(foreground[0], image_transform)
            segmented_image = quickshift(torch.permute(image, (1, 2, 0)), sigma=1, max_dist=patch_param)
            partition = get_segmentation_partition(segmented_image)
        else:
            # Get patch
            image = scale_to_original(foreground[0], image_transform)
            segmented_image = slic(torch.permute(image, (1, 2, 0)), sigma=1, n_segments=patch_param)
            partition = get_segmentation_partition(segmented_image)

        yield partition, None
