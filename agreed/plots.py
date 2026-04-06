"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import torch
from torchvision import transforms
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib import rc

from agreed.explainers.images.saliency_maps import get_saliency_from_attributions


def hex_to_rbg(hex_color):
    rgb = []
    for i in range(3):
        rgb.append( int(hex_color[1+2*i:1+2*(i+1)], 16) / 255. )
    return tuple(rgb)

def setup_plots(fontsize=20):
    rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':fontsize})
    rc('text', usetex=True)
    my_palette = [hex_to_rbg(hex_color) for hex_color in ["#6d24ff", "#fa824c", "#88d377", "#c41121", "#ffa9e7"]]
    return my_palette



def plot_image(x: torch.Tensor, current_axis: Axes | None = None, image_transforms: transforms.Compose | None = None):
    """
    Plot a given image tensor x. If an image_tansforms is passed, it is assumed that x must be transformed
    back into [0, 1]
    """
    if current_axis is None:
        current_axis = plt.gca()

    n_channels = x.shape[0]
    if image_transforms is not None:
        x = scale_to_original(x, image_transforms)
    # Gray image
    if n_channels == 1:
        current_axis.imshow(x.squeeze(0).numpy(), cmap="gray")
    # RGB Image
    else:
        x = x.numpy()
        x = np.stack((x[0], x[1], x[2]), axis=-1)
        current_axis.imshow(x)
    current_axis.set_xticks([])
    current_axis.set_yticks([])



def scale_to_original(x: torch.Tensor, image_transforms: transforms.Compose):
    """ X must be (C, H, W) """
    x = x.clone()
    for transformer in image_transforms.transforms[-1::-1]:
        if isinstance(transformer, transforms.Normalize):
            assert x.shape[0] == len(transformer.std)
            for c in range(x.shape[0]):
                x[c] = x[c] * transformer.std[c] + transformer.mean[c]
    return x

def scale_from_original(x: torch.Tensor, image_transforms: transforms.Compose):
    """ x must be (C, H, W) """
    x = x.clone()
    for transformer in image_transforms.transforms:
        if isinstance(transformer, transforms.Normalize):
            for c in range(x.shape[0]):
                x[c] = (x[c] - transformer.mean[c]) / transformer.std[c]
    return x



# def superpose_x_and_phi(x, phi, max_attrib, COLOR):
#     # Normalize and reshape to RGB
#     w = x.shape[2]
#     x = x.numpy()
#     phi = phi.abs()
#     x = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
#     X_RGB = np.stack((x, x, x), axis=-1)
#     COLOR = np.array(to_rgb(COLOR))
#     phi = phi.numpy() / max_attrib.item()
#     mixed_image = np.zeros((w, w, 3))
#     for i in range(w):
#         for j in range(w):
#             interpolation = np.abs(phi[i][j])
#             mixed_image[i][j] += (1 - interpolation) * X_RGB[i][j] + interpolation * COLOR
#     return np.clip(mixed_image, 0, 1)



def superpose_gray_x_and_signed_phi(image: torch.Tensor, saliency: torch.Tensor, max_saliency: float):
    h, w = image.shape[-2:]
    image_: np.ndarray = image.numpy()
    # Get a grayscale image with three channels
    if image.shape[0] == 3:
        grayscale_image = 0.299 * image_[0] + 0.587 * image_[1] + 0.114 * image_[2]
    else:
        grayscale_image = image_[0]
    grayscale_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=-1)

    # Hot and cold colors for saliency
    COLOR = np.array([to_rgb("#FF0000"), to_rgb("#0000FF")])
    if np.isclose(max_saliency, 0):
        saliency = saliency.numpy()
    else:
        saliency = saliency.numpy() / max_saliency

    # Mix the grayscale image with the saliency
    mixed_image = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            interpolation = np.abs(saliency[i][j])
            if saliency[i][j] >= 0:
                mixed_image[i][j] += (1 - interpolation) * grayscale_image[i][j] + interpolation * COLOR[0]
            else:
                mixed_image[i][j] += (1 - interpolation) * grayscale_image[i][j] + interpolation * COLOR[1]
    return np.clip(mixed_image, 0, 1)



def plot_saliency_map_and_partition(
        x,
        attribution,
        partition,
        attribution_name,
        axis,
        edge_color = (0.5, 0, 0.5),
        ):
    saliency = get_saliency_from_attributions(attribution, partition)
    mixed_image = superpose_gray_x_and_signed_phi(x, saliency, saliency.abs().max().item())
    mixed_image_with_boundaries = mark_boundaries(resize(mixed_image, (400, 400), order=0),
                                                  label_img=resize(partition.Imap, (400, 400), order=0),
                                                  color=edge_color)
    axis.imshow(mixed_image_with_boundaries)
    axis.set_xticks([])
    axis.set_yticks([])
