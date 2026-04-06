"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

import os
from joblib import dump

from utils import generate_blocks


def main():

    data_folder = os.path.join("results")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    for d in [10, 25, 50, 75, 100]:
        for i in range(5):
            blocks = generate_blocks(d, seed=i)
            dump(blocks, os.path.join(data_folder, f"blocks_dim_{d}_seed_{i}.bz2"))


if __name__ == "__main__":
    main()
