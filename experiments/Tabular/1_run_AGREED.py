"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
import numpy as np
from joblib import load
from dataclasses import dataclass, asdict
import yaml
from time import time

# Local
from utils import custom_train_test_split, CallableModel, get_disagreements_and_fidelity

from agreed.dataset.tabular.factory import DATASET_MAPPING, TASK_MAPPING
from agreed.explainers.tabular.iterable_grouping import GeneratorTabularAGREED
#, GeneratorTabularPairwise, GeneratorTabularRecurse, GeneratorTabularIGreedy

METHODS = {
    "agreed" : GeneratorTabularAGREED,
    # "recurse" : GeneratorTabularRecurse,
    # "pairwise" : GeneratorTabularPairwise,
    # "igreedy" : GeneratorTabularIGreedy
}

@dataclass
class Arguments:
    dataset: str = "marketing"  # The dataset:, marketing, default_credit, spam, nomao
    model: str = "gbt"          # The model : gbt, rf, ebm,
    n_samples: int = 200        # Number of background samples
    n_groups_min: int = 1       # Minimal Number of feature groups
    method: str = "agreed"      # Grouping Method : agreed, pairwise, recurse, igreedy
    n_jobs: int = 8             # Number of jobs on which to run
    seed: int = 0               # Seed used to subsample the background
    save: bool = False          # Save the results of the run in a csv


def main(args: Arguments):

    np.random.seed(args.seed)
    X, y, features = DATASET_MAPPING[args.dataset]()
    task = TASK_MAPPING[args.dataset]
    x_train, x_test, _, _ = custom_train_test_split(X, y.ravel(), task)
    d = x_train.shape[1]

    # Background data to compute explanations (fixed size)
    if x_test.shape[0] > 400:
        large_background = x_test[np.random.choice(x_test.shape[0], 400, replace=False)]
    else:
        large_background = x_train[np.random.choice(x_train.shape[0], 400, replace=False)]

    # Background data to run grouping algorithms (size can vary based on the data and model for faster runtimes)
    small_background = x_test[np.random.choice(x_test.shape[0], args.n_samples, replace=False)]

    # load the model
    model = load(os.path.join("models", f"{args.model}_{args.dataset}.joblib"))
    print(model)

    disagreement, fidelity = get_disagreements_and_fidelity(args.model, model, large_background, features)

    results_file = os.path.join("results", f"results_seed_{args.seed}.csv")
    if args.save:
        # Make the file if it does not exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as file:
                header = "data,model,method,threshold,n_groups,n_samples,n_calls,runtime,"
                header += "l2_pdp_vs_shap,l2_pdp_vs_pfi,l2_shap_vs_pfi,"
                header += "sensitivity_1_pdp,sensitivity_1_shap,sensitivity_1_pfi,"
                header += "INFD_pdp,INFD_shap,INFD_pfi,"
                header += "SWF_pdp,SWF_shap,SWF_pfi\n"
                file.write(header)
        # Append new results to the file
        with open(results_file, 'a') as file:
            file.write(f"{args.dataset},{args.model},{args.method},{100:f},{small_background.shape[1]:d},")
            file.write(f"{args.n_samples:d},{0:d},{0:f},{disagreement['l2_pdp_vs_shap']},{disagreement['l2_pdp_vs_pfi']},")
            file.write(f"{disagreement['l2_shap_vs_pfi']},{fidelity['sensitivity_1_pdp']},{fidelity['sensitivity_1_shap']},")
            file.write(f"{fidelity['sensitivity_1_pfi']},{fidelity['INFD_pdp']},{fidelity['INFD_shap']},")
            file.write(f"{fidelity['INFD_pfi']},{fidelity['SWF_pdp']},{fidelity['SWF_shap']},{fidelity['SWF_pfi']}\n")

    # Run AGREED and report time, number of model calls, and RandIndex
    callable_model = CallableModel(model)
    thresholds = np.logspace(0, -3, 30)
    n_groups_prev = d

    runtime = 0
    start = time()
    for partition in METHODS[args.method](callable_model, small_background, thresholds=thresholds):

        runtime = time() - start + runtime

        ########################### This runtime is not part of the partitioning algorithm ############################
        # Ignore if we have the same number of groups as before
        if partition.D == n_groups_prev:
            # Only recurse is non-sequential and will restart from scratch at each threshold
            if args.method == "recurse":
                callable_model.reset_n_calls()
                runtime = 0
            start = time()
            continue

        # Stop if we have too few groups
        if partition.D < args.n_groups_min:
            break

        n_groups_prev = partition.D
        features_to_group = [group for group in partition.Imap_inv if len(group) > 1]
        grouped_features = features.group(features_to_group)
        grouped_features.summary()

        # Results
        n_calls = callable_model.n_calls
        print(f"{n_calls / 1000:.0f} K calls")
        print(f"runtime: {runtime: .2f} seconds\n")

        disagreement, fidelity = get_disagreements_and_fidelity(args.model, model, large_background, grouped_features)

        if args.save:
            # Append new results to the file
            with open(results_file, 'a') as file:
                file.write(f"{args.dataset},{args.model},{args.method},{100:f},{partition.D:d},")
                file.write(f"{args.n_samples:d},{args.n_samples:d},{runtime:f},{disagreement['l2_pdp_vs_shap']},{disagreement['l2_pdp_vs_pfi']},")
                file.write(f"{disagreement['l2_shap_vs_pfi']},{fidelity['sensitivity_1_pdp']},{fidelity['sensitivity_1_shap']},")
                file.write(f"{fidelity['sensitivity_1_pfi']},{fidelity['INFD_pdp']},{fidelity['INFD_shap']},")
                file.write(f"{fidelity['INFD_pfi']},{fidelity['SWF_pdp']},{fidelity['SWF_shap']},{fidelity['SWF_pfi']}\n")

        if disagreement["l2_pdp_vs_pfi"] < 0.01:
            break

        # Only recurse is non-sequential and will restart from scratch at each threshold
        if args.method == "recurse":
            callable_model.reset_n_calls()
        ###############################################################################################################

        # We restart time timer before continuing the partition algorithm
        start = time()


if __name__ == "__main__":
    from simple_parsing import parse
    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print(yaml.dump(args_dict, default_flow_style=False))
    main(args)
