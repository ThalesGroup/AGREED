"""

Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.

"""

import os
import numpy as np
import torch
import yaml
from sklearn.metrics import rand_score
from dataclasses import dataclass, asdict
from time import time

# Src imports
from agreed.explainers.images.grouping import ImageAGREED, ImagePairwise
from agreed.explainers.images.decomposition import (
    TorchModel,
    get_patch_partition,
    get_baseline,
)

# Local imports
from utils import load_data, ToyModel


METHODS = {
    "agreed" : ImageAGREED,
    "pairwise" : ImagePairwise
}

@dataclass
class Arguments:
    method: str = "agreed"    # Grouping method: agreed, pairwise
    W: int = 28               # Size of images WxW
    device: str = "cpu"       # Device to run on: cpu, cuda
    save: bool = False        # Whether or nor to save the results of the run


def main(args: Arguments):

    # Generate data and model
    X, y, gt_Imap = load_data(args.W)
    # N = X.shape[0]
    model = ToyModel(input_size=args.W)
    model.to(args.device)
    wrapped_model = TorchModel(model, batch_size=2, device=args.device, class_1=0)
    baseline = get_baseline("zero", im_size=(args.W, args.W), n_channels=1)

    # Run AGREED and report time, number of model calls, and RandIndex
    init_partition = get_patch_partition(1, (args.W, args.W))
    for i in range(20):
        print(f"### Instance {i}")
        wrapped_model.reset_n_calls()
        Q = torch.concat( (X[[i]], baseline) )
        start = time()
        for pred_partition, (_, _) in METHODS[args.method](
                wrapped_model,
                background=Q,
                init_partition=init_partition,
                thresholds=(1e-6,)
                ):
            end = time()
            # Results
            randindex = rand_score(gt_Imap[i].ravel(), pred_partition.Imap.ravel())
            runtime = end - start
            n_calls = wrapped_model.n_calls
            print(f"RandIndex {randindex: 2f}")
            print(f"Runtime {runtime:.2f} seconds")
            print(f"{n_calls / 1000:.0f} K calls\n")

            if args.save:
                results_file = os.path.join("results", "results.csv")
                # Make the file if it does not exist
                if not os.path.exists(results_file):
                    with open(results_file, 'w') as file:
                        header = "instance,method,W,runtime,n_calls,randindex\n"
                        file.write(header)
                # Append new results to the file
                with open(results_file, 'a') as file:
                    file.write(f"{i:d},{args.method},{args.W},{runtime},{n_calls:d},{randindex}\n")



if __name__ == "__main__":
    # Parse config
    from simple_parsing import parse
    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print(yaml.dump(args_dict, default_flow_style=False))

    # Format options
    torch.set_printoptions(linewidth=2000, precision=2)
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

    main(args)
