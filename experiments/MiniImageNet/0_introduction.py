"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict

from agreed.plots import scale_to_original, plot_image, plot_saliency_map_and_partition
from agreed.dataset.images.factory import ImageDatasetFactory
from agreed.dataset.images.utils import select_correctly_predicted_samples
from agreed.models.factory import ModelFactory

from agreed.config.utils import get_config
from agreed.explainers.images.saliency_maps import get_all_attributions
from agreed.explainers.images.decomposition import (
    get_patch_partition,
    get_baseline,
    TorchModel,
    cosinedisagreement,
)
from agreed.explainers.imagenet.grouping import ImagenetAGREED



@dataclass
class Arguments:
    device: str = "cpu"                # The device on which to run inference
    model: str = "ResNet18"            # The model to explain : ResNet18, VGG16, ConvNext
    batch_size: int = 128              # Batch size for the model
    method: str = "agreed"             # Grouping method : square, quickshift, agreed.
    baseline: str = "zero"             # Baseline: zero, mean, blur.
    init_patch_size: int = 14          # Initial size of the patches to fuse.
    max_patch_size: int | None = None  # Maximum size for a single pixel patch
    label: int = 0                     # Only explain samples from this class
    explain_margin: bool = False       # Explain the logit margin
    show_original_image: bool = False  # Show the original image in color
    verbose: bool = False              # Show the config and AGREED logs


def main(args: Arguments):

    torch.manual_seed(0)

    # Load data and model via configurations
    config = get_config("MiniImageNet", args.model)
    if args.verbose:
        print("Configuration Files")
        print(OmegaConf.to_yaml(config))

    _, test_dataset = ImageDatasetFactory().load_dataset(config.dataset)
    # Keep transforms for later use when we plot [0, 1] images
    image_transform = test_dataset.transforms.transform
    model = ModelFactory().init_model(config.model, config.dataset).eval()
    model = model.to(args.device)

    SAVE_PATH = os.path.join(config.save_path, args.model, "Intro")

    # Only select samples from a given class
    test_dataset = select_correctly_predicted_samples(
        model, args.label, test_dataset, args.device, args.batch_size
    )

    attributions = {}

    max_patch_size = (
        args.max_patch_size
        if args.max_patch_size is not None
        else config.dataset.imsize[0] * config.dataset.imsize[1]
    )

    # Inspect five samples
    for instance in range(5):

        # Get the instance
        x, y = test_dataset[instance]
        x = x.unsqueeze(0)
        image = scale_to_original(x[0], image_transform)
        class_1 = y
        if args.explain_margin:
            _, topk = torch.topk(model(x.to(args.device)), k=2)
            class_0 = int(topk[0][1])
        else:
            class_0 = None

        wrapped_model = TorchModel(
            model,
            args.batch_size,
            args.device,
            class_1=class_1,
            class_0=class_0,
        )

        # The baseline image to "remove" patches
        z = get_baseline(
            args.baseline,
            im_size=config.dataset.imsize,
            n_channels=config.dataset.in_channels,
            x=x,
            model=model,
            device=args.device,
            label=args.label,
        )

        # Setup grouping method
        thresholds = (1, 0.75, 0.5, 0.25)
        n_rows = len(thresholds)
        n_cols = 6 + int(args.show_original_image)
        _, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols, n_rows)))

        init_partition = get_patch_partition(
            args.init_patch_size, config.dataset.imsize
        )

        init_disagreement = None
        # Try various number of max_group_size (each will lead to larger final patches)
        i = 0
        for partition, (_, _) in ImagenetAGREED(
                wrapped_model,
                x,
                z,
                thresholds,
                init_partition,
                max_patch_size=max_patch_size,
                verbose=args.verbose,
        ):

            # attributions = {"Arch": arch.unsqueeze(0), "Occ": occ.unsqueeze(0)}
            # Compute attribution methods
            attributions = get_all_attributions(
                wrapped_model,
                x,
                z,
                partition,
                n_perturbations=200,
                n_permutations=4,
            )

            # Print the objective value
            # disagreement = torch.sqrt(
            #     torch.sum((attributions["Arch"] - attributions["Occ"]) ** 2)
            # )
            disagreement = cosinedisagreement(attributions["Arch"], attributions["Occ"])
            if init_disagreement is None:
                init_disagreement = disagreement
            ax[i][0].set_ylabel(f"L = {disagreement / init_disagreement:.2f}")

            if args.show_original_image:
                plot_image(image, current_axis=ax[i][0])
                # Set titles on first row
                if i == 0:
                    ax[i][0].set_title("Image")

            for j, (method, attribution) in enumerate(attributions.items()):
                current_axis = ax[i][j + int(args.show_original_image)]
                plot_saliency_map_and_partition(
                    image,
                    attribution[0],
                    partition,
                    method,
                    current_axis,
                    edge_color=(0, 0, 0),
                )
                # Set titles on first row
                if i == 0:
                    current_axis.set_title(method)
            i += 1

        plt.tight_layout()
        filename = f"{args.method}_class_{args.label}_instance_{instance}_baseline_{args.baseline}_explain_margin_{args.explain_margin}.png"
        plt.savefig(os.path.join(SAVE_PATH, filename), dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    # Parse config
    from simple_parsing import parse

    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print("Command Line Arguments")
    print(yaml.dump(args_dict, default_flow_style=False))

    # Format options
    torch.set_printoptions(linewidth=2000, precision=2)
    np.set_printoptions(
        edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
    )

    main(args)
