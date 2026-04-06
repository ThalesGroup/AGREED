"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
import numpy as np
from joblib import dump
from dataclasses import dataclass, asdict
import yaml
from sklearn.model_selection import RandomizedSearchCV

# Local
from utils import get_cross_validator, get_hp_grid, custom_train_test_split, MODEL


from agreed.dataset.tabular.factory import DATASET_MAPPING, TASK_MAPPING


@dataclass
class Arguments:
    model: str = "ebm"         # Type of model: ebm, rf, gbt
    dataset: str = "marketing" # Name of the dataset: marketing, default_credit, spam, nomao
    n_jobs: int = 1            # Number of jobs in which to run
    n_repetitions: int = 20    # Number of times to run cross-validation


def main(args: Arguments):

    X, y, features = DATASET_MAPPING[args.dataset]()
    task = TASK_MAPPING[args.dataset]

    x_train, x_test, y_train, y_test = custom_train_test_split(X, y.ravel(), task)

    # Hyperparameter grid
    hp_grid = get_hp_grid(os.path.join("hparams", f"{args.model}.json"))

    # Cross validator for train/valid splits
    k = 5
    cross_validator = get_cross_validator(k, task, 42, "K-Fold")

    # Load model
    model = MODEL[args.model][task]

    # Cross Validation
    scoring = "neg_root_mean_squared_error" if task == "regression" else "accuracy"
    search = RandomizedSearchCV(
        model,
        hp_grid,
        scoring=scoring,
        cv=cross_validator,
        n_iter=args.n_repetitions,
        verbose=2,
        n_jobs=args.n_jobs
    ).fit(x_train, y_train)
    # Recover the optimal CV model
    model = search.best_estimator_
    res = search.cv_results_
    cv_perf = np.nan_to_num(res['mean_test_score'], nan=-1e10)
    best_idx = np.argmax(cv_perf)
    score = np.array([res[f"split{i}_test_score"][best_idx] for i in range(k)])
    print(score)

    # Save the model for later
    dump(model, os.path.join("models", f"{args.model}_{args.dataset}.joblib"))


if __name__ == "__main__":
    from simple_parsing import parse
    args = parse(config_class=Arguments)
    args_dict = asdict(args)
    print(yaml.dump(args_dict, default_flow_style=False))
    main(args)
