"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from agreed.plots import setup_plots


if __name__ == "__main__":

    my_palette = setup_plots()

    # Load the results
    df_ = []
    temp = None
    for seed in range(5):
        temp = pd.read_csv(os.path.join("results", f"results_seed_{seed}.csv"))
        df_.append(temp)
    df = pd.concat(df_, ignore_index=True)
    del temp

    d_range = np.sort(np.unique(df["d"]))
    N_range = np.sort(np.unique(df["n_samples"]))

    ##################################################################################
    # Plot the randindex as boxplot for each method aggregated over all experiments
    ##################################################################################

    plt.figure()
    sns.boxplot(x="method", y="randindex", data=df, width=0.45, palette=my_palette[0:1])
    plt.ylabel("Adjusted Rand Index")
    plt.xlabel("Grouping Method")
    plt.ylim(-1, 1.1)
    plt.savefig(os.path.join("results", "rand_index.pdf"), bbox_inches='tight')

    # # Plot the legend separately
    # plt.figure(figsize=(5, 0.6))
    # partitions = ["Random", "CART", "GADGET-PDP", "CoE", "PDP-PFI"]
    # for p in range(len(partitions)):
    #     plt.scatter(0, 0, label=partitions[p])
    # plt.legend(loc='center', ncol=5, prop={"size": 10}, framealpha=1)
    # plt.axis('off')
    # plt.savefig(os.path.join("Images", f"results_legend.pdf"), bbox_inches='tight', pad_inches=0)

    ##################################################################################
    # Plot the number of model inference as a function of N for each method
    ##################################################################################

    df_temp = df.copy()
    df_temp["n_calls"] = df_temp["n_calls"] / df_temp["n_samples"] / 1000
    plt.figure()
    sns.lineplot(x="n_samples", y="n_calls", hue="method", style="method",
                 data=df_temp, markers=True, dashes=False, palette=my_palette)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}K"))
    plt.xticks(N_range)
    plt.xlim(N_range[0] - 1, N_range[-1] + 1)
    plt.ylim(0, 250)
    plt.ylabel(r"(Number of Inference) / $N$")
    plt.xlabel(r"$N$")
    plt.legend([], framealpha=0)
    plt.savefig(os.path.join("results", "calls_vs_N.pdf"), bbox_inches='tight')


    ##################################################################################
    # Plot the number of model inference as a function of d for each method
    ##################################################################################

    df_temp = df.copy()
    df_temp["n_calls"] = df_temp["n_calls"] / df_temp["d"] / 1000
    plt.figure()
    sns.lineplot(x="d", y="n_calls", hue="method", style="method",
                 data=df_temp, markers=True, dashes=False, palette=my_palette)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}K"))
    plt.xticks(d_range)
    plt.xlim(d_range[0] - 1, d_range[-1] + 1)
    plt.ylim(0, 150)
    plt.ylabel(r"(Number of Inference) / $d$")
    plt.xlabel(r"$d$")
    plt.savefig(os.path.join("results", "calls_vs_d.pdf"), bbox_inches='tight')


    ##################################################################################
    # Plot the runtime as a function of N for each method
    ##################################################################################

    plt.figure()
    sns.lineplot(x="n_samples", y="runtime", hue="method", style="method",
                 data=df, markers=True, dashes=False, palette=my_palette, errorbar=("ci", 95))
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    plt.xticks(N_range)
    plt.xlim(N_range[0] - 1, N_range[-1] + 1)
    # plt.ylim(0, 250)
    plt.ylabel("Runtime (s)")
    plt.xlabel(r"$N$")
    plt.legend([], framealpha=0)
    plt.savefig(os.path.join("results", "time_vs_N.pdf"), bbox_inches='tight')


    ##################################################################################
    # Plot the runtime as a function of d for each method
    ##################################################################################

    plt.figure()
    sns.lineplot(x="d", y="runtime", hue="method", style="method",
                 data=df, markers=True, dashes=False, palette=my_palette, errorbar=("ci", 95))
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    plt.xticks(d_range)
    plt.xlim(d_range[0] - 1, d_range[-1] + 1)
    # plt.ylim(0, 150)
    plt.ylabel("Runtime (s)")
    plt.xlabel(r"$d$")
    plt.savefig(os.path.join("results", "time_vs_d.pdf"), bbox_inches='tight')
