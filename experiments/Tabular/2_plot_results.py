"""
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
    df_ = []
    temp = None
    for seed in range(1):
        temp = pd.read_csv(os.path.join("results", f"results_seed_{seed}.csv"))
        df_.append(temp)
    df = pd.concat(df_, ignore_index=True)
    del temp, df_
    # # Turn cosine similarity into cosine distance
    # df["cosine_g"] = 0.5 * (1 - df["cosine_g"])
    # df["cosine_l"] = 0.5 * (1 - df["cosine_l"])

    # # Plot the legend separately
    # plt.figure(figsize=(5, 0.6))
    # partitions = ["Random", "CART", "GADGET-PDP", "CoE", "PDP-PFI"]
    # for p in range(len(partitions)):
    #     plt.scatter(0, 0, label=partitions[p])
    # plt.legend(loc='center', ncol=5, prop={"size": 10}, framealpha=1)
    # plt.axis('off')
    # plt.savefig(os.path.join("Images", f"results_legend.pdf"), bbox_inches='tight', pad_inches=0)

    ##################################################################################
    # For each dataset and model plot the l2 disagreement vs number of groups
    ##################################################################################
    plot_legend = True
    max_disagreement = {"marketing": 0.15, "default_credit": 0.05, "spam": 0.03, "nomao": 0.05}
    hue_order = ["l2_pdp_vs_shap", "l2_pdp_vs_pfi", "l2_shap_vs_pfi"]
    for (dataset, model), group in df.groupby(["data", "model"]):

        plt.figure()
        plt.plot(group["n_groups"], group["l2_pdp_vs_shap"], marker="o", c=my_palette[0], label="pdp_vs_shap")
        plt.plot(group["n_groups"], group["l2_pdp_vs_pfi"], marker="o", c=my_palette[1], label="pdp_vs_pfi")
        plt.plot(group["n_groups"], group["l2_shap_vs_pfi"], marker="o", c=my_palette[2], label="shap_vs_pfi")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
        # plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
        # plt.ylim(0, group["l2_pdp_vs_pfi"].max())
        # plt.title(f"{model} on {dataset}")
        plt.legend(framealpha=1)
        plt.ylabel(r"$L_2$ Disagreement")
        plt.xlabel("Number of Feature Groups")
        plt.grid(True)
        plt.legend(framealpha=1)
        plt.savefig(os.path.join("results", f"disagreements_{model}_{dataset}.pdf"), bbox_inches='tight')

    ##################################################################################
    # For each dataset and model plot the Sensitivity-n fidelity vs number of groups
    ##################################################################################
    plot_legend = True
    hue_order = ["l2_pdp_vs_shap", "l2_pdp_vs_pfi", "l2_shap_vs_pfi"]
    for (dataset, model), group in df.groupby(["data", "model"]):

        plt.figure()
        plt.plot(group["n_groups"], group["sensitivity_1_pdp"], marker="o", c=my_palette[0], label="pdp")
        plt.plot(group["n_groups"], group["sensitivity_1_shap"], marker="o", c=my_palette[1], label="shap")
        plt.plot(group["n_groups"], group["sensitivity_1_pfi"], marker="o", c=my_palette[2], label="pfi")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
        # plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
        # plt.ylim(0, group["sensitivity_1_pdp"].max())
        # plt.title(f"{model} on {dataset}")
        plt.legend(framealpha=1)
        plt.ylabel(r"Sensitivity-1 unfaithfulness")
        plt.xlabel("Number of Feature Groups")
        plt.grid(True)
        plt.savefig(os.path.join("results", f"sensitivity_1_{model}_{dataset}.pdf"), bbox_inches='tight')

    ##################################################################################
    # For each dataset and model plot the SWF fidelity vs number of groups
    ##################################################################################
    plot_legend = True
    for (dataset, model), group in df.groupby(["data", "model"]):

        plt.figure()
        plt.plot(group["n_groups"], group["SWF_pdp"], marker="o", c=my_palette[0], label="pdp")
        plt.plot(group["n_groups"], group["SWF_shap"], marker="o", c=my_palette[1], label="shap")
        plt.plot(group["n_groups"], group["SWF_pfi"], marker="o", c=my_palette[2], label="pfi")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
        # plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
        # plt.ylim(0, group["sensitivity_1_pdp"].max())
        # plt.title(f"{model} on {dataset}")
        plt.legend(framealpha=1)
        plt.ylabel("SWF unfaithfulness")
        plt.xlabel("Number of Feature Groups")
        plt.grid(True)
        plt.savefig(os.path.join("results", f"SWF_{model}_{dataset}.pdf"), bbox_inches='tight')

    ##################################################################################
    # For each dataset and model plot the INFD fidelity vs number of groups
    ##################################################################################
    plot_legend = True
    for (dataset, model), group in df.groupby(["data", "model"]):

        plt.figure()
        plt.plot(group["n_groups"], group["INFD_pdp"], marker="o", c=my_palette[0], label="pdp")
        plt.plot(group["n_groups"], group["INFD_shap"], marker="o", c=my_palette[1], label="shap")
        plt.plot(group["n_groups"], group["INFD_pfi"], marker="o", c=my_palette[2], label="pfi")
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
        # plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
        # plt.ylim(0, group["sensitivity_1_pdp"].max())
        # plt.title(f"{model} on {dataset}")
        plt.legend(framealpha=1)
        plt.ylabel("INFD unfaithfulness")
        plt.xlabel("Number of Feature Groups")
        plt.grid(True)
        plt.savefig(os.path.join("results", f"INFD_{model}_{dataset}.pdf"), bbox_inches='tight')

    # ##################################################################################
    # # For each dataset and model plot the l2 disagreement vs number of groups
    # ##################################################################################
    # plot_legend = True
    # max_disagreement = {"marketing": 0.15, "default_credit": 0.05, "spam": 0.03, "nomao": 0.05}
    # hue_order = ["agreed", "recurse", "igreedy", "pairwise", "baseline"]
    # for (dataset, model), group in df.groupby(["data", "model"]):
    #     if dataset == "nomao":
    #         continue

    #     plt.figure()
    #     sns.lineplot(x="n_groups", y="l2_l", hue="method", hue_order=hue_order,
    #                  style="method", data=group, markers=True, dashes=False, palette=my_palette)
    #     plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    #     plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
    #     plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
    #     plt.ylim(0, max(max_disagreement[dataset], group["l2_l"].max()))
    #     # plt.title(f"{model} on {dataset}")
    #     plt.ylabel(r"$L_2$ Disagreement")
    #     plt.xlabel("Number of Feature Groups")
    #     # if plot_legend:
    #     #     plot_legend = False
    #     # else:
    #     #     plt.legend('', frameon=False)
    #     # plt.show()
    #     plt.savefig(os.path.join("results", f"l2_disagreements_{model}_{dataset}.pdf"), bbox_inches='tight')


    # ##################################################################################
    # # For each dataset and model plot the cosine disagreement vs number of groups
    # ##################################################################################
    # plot_legend = True
    # max_disagreement = {"marketing": 0.15, "default_credit": 0.05, "spam": 0.03, "nomao": 0.05}
    # hue_order = ["agreed", "recurse", "igreedy", "pairwise", "baseline"]
    # for (dataset, model), group in df.groupby(["data", "model"]):
    #     if dataset == "nomao":
    #         continue

    #     plt.figure()
    #     sns.lineplot(x="n_groups", y="cosine_l", hue="method", hue_order=hue_order,
    #                  style="method", data=group, markers=True, dashes=False, palette=my_palette)
    #     plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    #     plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
    #     plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
    #     plt.ylim(0, max(max_disagreement[dataset], group["cosine_l"].max()))
    #     # plt.title(f"{model} on {dataset}")
    #     plt.ylabel("Cosine Disagreement")
    #     plt.xlabel("Number of Feature Groups")
    #     # if plot_legend:
    #     #     plot_legend = False
    #     # else:
    #     #     plt.legend('', frameon=False)
    #     # plt.show()
    #     plt.savefig(os.path.join("results", f"cosine_disagreements_{model}_{dataset}.pdf"), bbox_inches='tight')

    # ##################################################################################
    # # For each dataset and model plot the fidelity vs number of groups
    # ##################################################################################
    # plot_legend = True
    # max_disagreement = {"marketing": 0.15, "default_credit": 0.05, "spam": 0.03, "nomao": 0.05}
    # hue_order = ["agreed", "recurse", "igreedy", "pairwise", "baseline"]
    # for (dataset, model), group in df.groupby(["data", "model"]):
    #     if dataset == "nomao":
    #         continue

    #     plt.figure()
    #     sns.lineplot(x="n_groups", y="fidelity", hue="method", hue_order=hue_order,
    #                  style="method", data=group, markers=True, dashes=False, palette=my_palette)
    #     plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    #     plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
    #     plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
    #     plt.ylim(0, max(max_disagreement[dataset], group["fidelity"].max()))
    #     # plt.title(f"{model} on {dataset}")
    #     plt.ylabel("Explanation Fidelity")
    #     plt.xlabel("Number of Feature Groups")
    #     # if plot_legend:
    #     #     plot_legend = False
    #     # else:
    #     #     plt.legend('', frameon=False)
    #     # plt.show()
    #     plt.savefig(os.path.join("results", f"fidelity_disagreements_{model}_{dataset}.pdf"), bbox_inches='tight')


    # ##################################################################################
    # # For NOMAO and GBT plot the disagreement vs number of groups
    # ##################################################################################
    # hue_order = ["agreed", "baseline"]
    # group = df[df["data"] == "nomao"]

    # plt.figure()
    # sns.lineplot(x="n_groups", y="disagreement_local", hue="method", hue_order=hue_order,
    #                 style="method", data=group, markers=True, dashes=False, palette=my_palette)
    # plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    # plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1), fontsize=10)
    # plt.xlim(group["n_groups"].min(), group["n_groups"].max() + 1)
    # plt.ylim(0, max(max_disagreement[dataset], group["disagreement_local"].max()))
    # plt.ylabel("Cosine Disagreement")
    # plt.xlabel("Number of Feature Groups")
    # plt.savefig(os.path.join("results", f"disagreements_gbt_nomao.pdf"), bbox_inches='tight')



    # # ##################################################################################
    # # Report the number of function calls for each dataset and method
    # # ##################################################################################

    # df_ = df[~(df["method"]=="baseline")]
    # df_ = df_[df_["n_calls"] > 0]

    # plt.figure()
    # order = ["agreed", "recurse", "igreedy", "pairwise"]
    # sns.boxplot(x="method", y="n_calls", hue="data", data=df_, order=order, palette=my_palette)
    # # plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}K"))
    # # plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1))
    # # plt.xlim(N_range[0] - 1, N_range[-1] + 1)
    # # plt.ylim(0.75, 1.01)
    # # plt.xscale("log")
    # plt.ylabel("Number of Model Inferences")
    # plt.xlabel("Grouping Method")
    # plt.legend(framealpha=1, fontsize=10)
    # plt.savefig(os.path.join("results", "scalability.pdf"), bbox_inches='tight')



    # ##################################################################################
    # Report the runtime for each dataset and method
    # ##################################################################################

    df_copy = df[~(df["method"]=="baseline")]
    df_copy = df_copy[df_copy["runtime"] > 0]

    plt.figure()
    order = ["agreed", "recurse", "igreedy", "pairwise"]
    sns.boxplot(x="method", y="runtime", hue="data", data=df_copy, order=order, palette=my_palette)
    # plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}K"))
    # plt.xticks(range(group["n_groups"].min(), group["n_groups"].max()+1))
    # plt.xlim(N_range[0] - 1, N_range[-1] + 1)
    # plt.ylim(0.75, 1.01)
    # plt.xscale("log")
    plt.ylabel("Runtime (s)")
    plt.xlabel("Grouping Method")
    plt.legend(framealpha=1, fontsize=10)
    plt.savefig(os.path.join("results", "runtimes.pdf"), bbox_inches='tight')
