"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Any
import yaml
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict
import pandas as pd

from agreed.config.utils import get_config
from agreed.dataset.images.factory import ImageDatasetFactory
from agreed.models.factory import ModelFactory
from agreed.explainers.images.saliency_maps import get_all_attributions, get_fidelity
from agreed.explainers.images.decomposition import (
    TorchModel,
    get_baseline,
    get_patch_partition,
    get_avg_patch_size,
    l2disagreement,
)
from agreed.explainers.imagenet.grouping import ImagenetAGREED
from agreed.explainers.imagenet.segmentation import SegmentationGrouping


@dataclass
class Arguments:
    model: str = "ResNet18"     # Name of the model: ResNet18, vgg16, or ConvNext
    batch_size: int = 128       # Batch size for the model
    baseline: str = "zero"      # Baseline Image
    method: str = "agreed"      # Method to compute the patches: agreed, square, quickshift, slic
    init_patch_size: int = 14   # The initial size of the patches to fuse
    device: str = "cpu"         # The device on which to run inference
    num_threads: int = -1       # Number of CPU threads on which to run
    verbose: bool = False       # Show the config and AGREED logs


def main(args: Arguments):


    torch.manual_seed(0)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    config = get_config("MiniImageNet", args.model)
    if args.verbose:
        print("Configuration File")
        print(OmegaConf.to_yaml(config))
    _, test_dataset = ImageDatasetFactory().load_dataset(config.dataset)
    image_transform = test_dataset.transforms.transform
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)
    model = ModelFactory().init_model(config.model, config.dataset).eval()
    model = model.to(args.device)

    # Start from previous run if the result file already exists
    folder_path = os.path.join(config.save_path, args.model)
    filename = f"patch_size_vs_disagreement_{args.method}.csv"
    if os.path.exists(os.path.join(folder_path, filename)):
        results = pd.read_csv(os.path.join(folder_path, filename))
        results = results[results["background"] == args.baseline]
        last_instance = len(results["instance"].unique()) - 1
        start_instance = last_instance + 1
    else:
        start_instance = 0


    # Iterate over all test set
    for i, (x, y) in enumerate(test_loader):

        # Only look at first 100 test images
        if i >= 100:
            break

        # Start where the run was interrupted
        if i < start_instance:
            continue

        results_dict: dict[str, Any]  = {
                    "instance": [],
                    "method": [],
                    "background": [],
                    "avg_patch_size": [],
                    "IG-vs-Arch": [],
                    "IG-vs-Occ": [],
                    "IG-vs-LIME": [],
                    "IG-vs-SHAP": [],
                    "INFD-Arch": [],
                    "INFD-Occ": [],
                    "INFD-IG": []
        }

        # Get the predicted label
        y = model(x.to(args.device)).argmax(-1).item()
        print(f"Predicted label {y}")

        z = get_baseline(
            args.baseline,
            config.dataset.imsize,
            config.dataset.in_channels,
            x,
            model,
            args.device,
            y,
        )

        wrapped_model =  TorchModel(
            model,
            batch_size=args.batch_size,
            device=args.device,
            class_1=y,
        )

        sweep_hyperparameters: tuple[int, ...]

        # Setup grouping method
        if args.method == "agreed":
            thresholds = (1, 0.9, 0.75, 0.5, 0.25)
            init_partition = get_patch_partition(args.init_patch_size, config.dataset.imsize)
            iterable = ImagenetAGREED(
                wrapped_model,
                x,
                z,
                thresholds=thresholds,
                partition=init_partition,
                verbose=args.verbose,
            )
        elif args.method in ["square", "quick"]:
            if args.method == "square":
                # The semi-length of each patch
                sweep_hyperparameters = (14, 32, 48, 64, 80)
            elif args.method == "quick":
                # The max-dist hyperparam -> larger means less patches
                sweep_hyperparameters = (10, 50, 100, 500)
            iterable = SegmentationGrouping(
                x, config.dataset.imsize, sweep_hyperparameters, image_transform, method=args.method
            )
        else:
            raise NotImplementedError()

        attributions = {}

        for partition, _ in iterable:
            # Report
            results_dict["instance"] = i
            results_dict["method"] = args.method
            results_dict["background"] = args.baseline
            results_dict["avg_patch_size"] = get_avg_patch_size(partition)

            attributions = get_all_attributions(
                wrapped_model,
                x,
                z,
                partition,
                n_perturbations=200,
                n_permutations=4,
            )

            # Report disagreements
            for method in ["Arch", "Occ", "LIME", "SHAP"]:
                results_dict[f"IG-vs-{method}"] = float( l2disagreement(attributions["IG"], attributions[method]) )

            # Report faithfulness
            for method in ["Arch", "Occ", "IG"]:
                results_dict[f"INFD-{method}"] = float(
                        get_fidelity(
                            wrapped_model,
                            x,
                            z,
                            attributions[method],
                            partition,
                        ))

            results_df = pd.DataFrame([results_dict])
            # Append results if file already exists
            if os.path.exists(os.path.join(folder_path, filename)):
                results_df.to_csv(os.path.join(folder_path, filename), index=False, mode='a', header=False)
            # Create new file if first instance
            else:
                results_df.to_csv(os.path.join(folder_path, filename), index=False)


if __name__ == "__main__":
    # Parse config
    from simple_parsing import parse

    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print(yaml.dump(args_dict, default_flow_style=False))

    # Format options
    torch.set_printoptions(linewidth=2000, precision=2)

    main(args)
