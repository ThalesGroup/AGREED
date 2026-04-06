""""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from agreed.plots import setup_plots


if __name__ == "__main__":

    my_palette = setup_plots()

    # Load the results
    df = pd.read_csv(os.path.join("results", "results.csv"))

    ##################################################################################
    # Plot the randindex as boxplot for each method aggregated over all experiments
    ##################################################################################

    plt.figure()
    sns.boxplot(x="method", y="randindex", data=df, width=0.45, palette=my_palette[0:1])
    plt.ylabel("Adjusted Rand Index")
    plt.xlabel("Grouping Method")
    plt.ylim(0.95, 1.05)
    plt.savefig(os.path.join("results", "rand_index.pdf"), bbox_inches='tight')

    ##################################################################################
    # Plot the number of model inference as a function of W for each method
    ##################################################################################

    df_temp = df.copy()
    df_temp["n_calls"] = df_temp["n_calls"]
    plt.figure()
    sns.lineplot(x="W", y="n_calls", hue="method", style="method",
                 data=df_temp, markers=True, dashes=False, palette=my_palette)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}K"))
    plt.xticks([16, 20, 24, 28, 32])
    # plt.xlim(N_range[0] - 1, N_range[-1] + 1)
    # plt.ylim(0, 250)
    plt.ylabel("Number of Inference")
    plt.xlabel(r"$W$")
    plt.legend(framealpha=1)
    plt.savefig(os.path.join("results", "calls_vs_W.pdf"), bbox_inches='tight')
