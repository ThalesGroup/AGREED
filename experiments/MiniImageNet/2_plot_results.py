"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from agreed.plots import setup_plots

if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ################################### Setup #################################

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--save_path", type=str, default="results/", help="Where to save the results"
    )
    args, unknown = parser.parse_known_args()
    print(args)

    my_palette = setup_plots()

    # Root folder for results
    df_ = []
    models = ["vgg16", "ResNet18", "ConvNext"]
    filenames = ["square", "quick", "agreed"]
    for model in models:
        folder_path = os.path.join(args.save_path, model)
        for filename in filenames:
            full_filename = os.path.join(
                folder_path, "patch_size_vs_disagreement_" + filename + ".csv"
            )
            results = pd.read_csv(full_filename)
            results["model"] = model
            df_.append(results)
    df = pd.concat(df_, axis=0)
    del df_

    # Consider the mean disagreement
    df["disagreement"] = df[
        ["IG-vs-Arch", "IG-vs-Occ", "IG-vs-LIME", "IG-vs-SHAP"]
    ].mean(axis=1)
    DISAGREEMENT_RANGE = {
        "vgg16": (0, 500),
        "ResNet18": (0, 600),
        "ConvNext": (0, 90),
    }


    # Iterate over all groups
    for (model, baseline), group in df.groupby(["model", "background"]):
        plt.figure()
        for i, (method, group_) in enumerate(group.groupby("method")):
            min_patch_size = group_["avg_patch_size"].min()
            max_patch_size = group_["avg_patch_size"].max()
            patch_size_grid = np.linspace(min_patch_size, max_patch_size, 100)
            disagreements_per_instance = []
            for _, group__ in group_.groupby("instance"):
                disagreements_per_instance.append(
                    np.interp(
                        patch_size_grid,
                        group__["avg_patch_size"],
                        group__["disagreement"],
                    )
                )
            disagreements_per_instance = np.vstack(disagreements_per_instance)
            median = np.quantile(disagreements_per_instance, q=0.5, axis=0)
            lower_bound = np.quantile(disagreements_per_instance, q=0.2, axis=0)
            upper_bound = np.quantile(disagreements_per_instance, q=0.8, axis=0)
            plt.plot(patch_size_grid, median, color=my_palette[i], label=method)
            plt.fill_between(
                patch_size_grid,
                lower_bound,
                upper_bound,
                color=my_palette[i],
                alpha=0.25,
            )
        plt.grid(True)
        if baseline == "zero" and model == "ResNet18":
            plt.legend(framealpha=1)

        plt.ylim(*DISAGREEMENT_RANGE[model])
        plt.xlabel("Average Group Size")
        plt.ylabel("Average Disagreements")

        plt.savefig(
            os.path.join(args.save_path, model, f"disagreement_{baseline}.pdf"),
            bbox_inches="tight",
        )

    # Report faithfulness metrics
    INFD_RANGE = {
        "vgg16": {
            "INFD-Arch": (0, 400),
            "INFD-Occ": (0, 400),
            "INFD-IG": (0, 400),
        },
        "ResNet18": {
            "INFD-Arch": (0, 400),
            "INFD-Occ": (0, 400),
            "INFD-IG": (0, 400),
        },
        "ConvNext": {
            "INFD-Arch": (0, 40),
            "INFD-Occ": (0, 10),
            "INFD-IG": (0, 20),
        }
    }
    for faithfulness in ["INFD-Arch", "INFD-Occ", "INFD-IG"]:
        for (model, baseline), group in df.groupby(["model", "background"]):
            plt.figure()
            for i, (method, group_) in enumerate(group.groupby("method")):
                min_patch_size = group_["avg_patch_size"].min()
                max_patch_size = group_["avg_patch_size"].max()
                patch_size_grid = np.linspace(min_patch_size, max_patch_size, 100)
                faithfulness_per_instance_ = []
                for _, group__ in group_.groupby("instance"):
                    faithfulness_per_instance_.append(
                        np.interp(
                            patch_size_grid,
                            group__["avg_patch_size"],
                            group__[faithfulness],
                        )
                    )
                faithfulness_per_instance = np.vstack(faithfulness_per_instance_)

                # Scale down this value to improve the yticks (same scale as others)
                if faithfulness == "INFD-Arch" and model == "ConvNext":
                    faithfulness_per_instance = faithfulness_per_instance / 100

                median = np.quantile(faithfulness_per_instance, q=0.5, axis=0)
                lower_bound = np.quantile(faithfulness_per_instance, q=0.3, axis=0)
                upper_bound = np.quantile(faithfulness_per_instance, q=0.7, axis=0)
                plt.plot(patch_size_grid, median, color=my_palette[i], label=method)
                plt.fill_between(
                    patch_size_grid,
                    lower_bound,
                    upper_bound,
                    color=my_palette[i],
                    alpha=0.25,
                )
            plt.grid(True)
            plt.ylim(*INFD_RANGE[model][faithfulness])
            plt.xlabel("Average Group Size")
            plt.ylabel(faithfulness)

            plt.savefig(
                os.path.join(args.save_path, model, f"{faithfulness}_{baseline}.pdf"),
                bbox_inches="tight",
            )
