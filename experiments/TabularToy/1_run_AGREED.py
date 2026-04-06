"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
from sklearn.metrics import adjusted_rand_score
from joblib import load
from time import time
from dataclasses import dataclass, asdict
import yaml

# Local imports
from utils import generate_data, get_ground_truth_partition
from utils import Model

# Src imports
from agreed.explainers.tabular.grouping import TabularAGREED, TabularPairwise, TabularRecurse, TabularIGreedy

METHODS = {
    "agreed" : TabularAGREED,
    "recurse" : TabularRecurse,
    "pairwise" : TabularPairwise,
    "igreedy" : TabularIGreedy
}

@dataclass
class Arguments:
    seed: int = 0             # The seed for the run
    n_samples: int = 200      # Number of samples used to run AGREED
    d: int = 10               # Number of input features
    rho: float = 0.0          # Feature correlation
    method: str = "agreed"    # The grouping method
    save: bool = False        # Whether to save the results of the run


def main(args: Arguments):

    # Generate data and model
    X = generate_data(args.n_samples, args.d, rho=args.rho, seed=args.seed)
    blocks = load(os.path.join("results", f"blocks_dim_{args.d}_seed_{args.seed}.bz2"))
    model = Model(blocks)
    ground_truth_partition = get_ground_truth_partition(args.d, blocks)

    # Run AGREED and report time, number of model calls, and RandIndex
    start = time()
    estimated_partition = METHODS[args.method](model, X, threshold=1e-10)
    end = time()

    # Results
    randindex = adjusted_rand_score(ground_truth_partition.Imap, estimated_partition.Imap)
    runtime = end - start
    n_calls = model.n_calls
    print(f"RandIndex: {randindex: .2f}")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Called the model {n_calls / 1000:.0f} K times\n\n")

    if args.save:
        results_file = os.path.join("results", f"results_seed_{args.seed}.csv")
        # Make the file if it does not exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as file:
                header = "method,seed,n_samples,d,rho,runtime,n_calls,randindex\n"
                file.write(header)
        # Append new results to the file
        with open(results_file, 'a') as file:
            file.write(f"{args.method},{args.seed:d},{args.n_samples:d},{args.d:d},{args.rho:.2f},{runtime},{n_calls:d},{randindex}\n")



if __name__ == "__main__":
    from simple_parsing import parse
    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print(yaml.dump(args_dict, default_flow_style=False))
    main(args)
