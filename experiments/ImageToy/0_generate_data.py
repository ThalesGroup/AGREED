"""

Copyright © 2026 Thales. All rights reserved.


Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.

"""
import os
from joblib import dump
import matplotlib.pyplot as plt

from utils import generate_data


def main():

    data_folder = os.path.join("results")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    for W in [16, 20, 24, 28, 32]:
        X, y, partition = generate_data(N=1000, W=W)
        dump([X, y, partition], os.path.join(data_folder, f"dataset_{W}.bz2"))

        if W == 28:
            # Plot one image
            plt.figure()
            plt.imshow(X[0][0], cmap="grey")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join("results", "example.png"), dpi=220, bbox_inches="tight", pad_inches=0)



if __name__ == "__main__":
    main()
