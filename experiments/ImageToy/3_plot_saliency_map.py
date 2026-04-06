""""

Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.

"""

import os
import matplotlib.pyplot as plt
import yaml
from dataclasses import dataclass, asdict
import torch

from agreed.explainers.images.grouping import ImageAGREED
from agreed.explainers.images.decomposition import TorchModel, get_patch_partition, get_baseline
from agreed.explainers.images.saliency_maps import get_all_attributions
from agreed.plots import plot_saliency_map_and_partition

# Local imports
from utils import load_data, ToyModel


@dataclass
class Arguments:
    method: str = "agreed"  # Grouping Method : agreed, pairwise
    device: str = "cpu"  # Device to run the model : cpu, gpu
    W: int = 16  # The size of images WxW
    only_arch_occ: bool = True


def main(args: Arguments):

    # Generate data and model
    X, y, gt_Imap = load_data(W=args.W)
    N = X.shape[0]
    model = ToyModel(input_size=args.W)
    model.to(args.device)
    wrapped_model = TorchModel(model, batch_size=2, device=args.device, class_1=0)
    baseline = get_baseline("zero", im_size=(args.W, args.W), n_channels=1)

    # Run AGREED and report time, number of model calls, and RandIndex
    init_partition = get_patch_partition(patch_size=1, im_size=(args.W, args.W))
    thresholds = (1.0, 0.5, 1e-6)
    for i in range(N):
        if i > 10:
            break
        x = X[[i]]
        Q = torch.concat((x, baseline))

        if args.only_arch_occ:
            num_rows = 2
        else:
            num_rows = 6
        fig, axes = plt.subplots(
            num_rows, len(thresholds), figsize=(2 * len(thresholds), num_rows * 2)
        )
        iteration = 0
        for pred_partition, (arch, occ) in ImageAGREED(
            wrapped_model,
            background=Q,
            init_partition=init_partition,
            thresholds=thresholds,
        ):

            if args.only_arch_occ:
                attributions = {
                    "Arch": arch.unsqueeze(0),
                    "Occlusion": occ.unsqueeze(0),
                }
            else:
                # Compute attribution methods
                attributions = get_all_attributions(
                    model,
                    x,
                    baseline,
                    pred_partition,
                    class_1=0,
                    device=args.device,
                    n_permutations=20,
                )
                # if iteration == 0:
                #     axes[0][5].set_title("SHAP")

            for j, (method, attribution) in enumerate(attributions.items()):
                current_axis = axes[j][iteration]
                plot_saliency_map_and_partition(
                    x[0], attribution[0], pred_partition, method, current_axis
                )
                # Set titles on first row
                if j == 0:
                    current_axis.set_title(
                        r"$\mathcal{L}^{\text{AGREED}}$ = " + str(thresholds[iteration])
                    )

                if iteration == 0:
                    current_axis.set_ylabel(method)
            iteration += 1
        plt.tight_layout()
        plt.savefig(
            os.path.join("results", f"saliency_{i}.pdf"), bbox_inches="tight", dpi=400
        )


if __name__ == "__main__":
    # Parse config
    from simple_parsing import parse

    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print(yaml.dump(args_dict, default_flow_style=False))

    main(args)
