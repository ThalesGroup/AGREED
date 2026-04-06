""""

Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.

"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from joblib import load


def generate_data(N: int, W: int):
    np.random.seed(42)
    X = np.zeros((N, 1, W, W))
    y = np.zeros(N)
    partition = np.zeros((N, W, W))
    # Generate a random pair of points in each image
    corners = np.random.randint(low=4, high=W-4, size=(N, 4))
    for i in range(N):
        points = corners[i]
        min_y, max_y = np.sort(points[:2])
        min_x, max_x = np.sort(points[2:])
        h = max(max_y - min_y + 1, 4)
        w = max(max_x - min_x + 1, 4)
        if h > w:
            y[i] = 1
        elif h == w:
            w += 1
        # Uniform noise at random positions
        # X[i] += (np.random.uniform(size=(W, W)) > 0.9) * np.random.uniform(high=0.25, size=(W, W))
        # X[i, :, min_y, min_x] = 0.75
        # X[i, :, min_y+h-1, min_x] = 0.75
        # X[i, :, min_y, min_x+w-1] = 0.75
        # X[i, :, min_y+h-1, min_x+w-1] = 0.75
        X[i, :, min_y, min_x+1:min_x+w-1] = 1
        X[i, :, min_y+h-1, min_x+1:min_x+w-1] = 1
        X[i, :, min_y+1:min_y+h-1, min_x] = 1
        X[i, :, min_y+1:min_y+h-1, min_x+w-1] = 1

        # Set the ground-truth partition
        partition[i, min_y, min_x+1:min_x+w-1] = -1
        partition[i, min_y+h-1, min_x+1:min_x+w-1] = -2
        partition[i, min_y+1:min_y+h-1, min_x] = -3
        partition[i, min_y+1:min_y+h-1, min_x+w-1] = -4
        counter = 0
        for j in range(W*W):
            if partition[i, j//W, j%W] >= 0:
                partition[i, j//W, j%W] = counter
                counter += 1
    return X, y, partition + 4


def load_data(W: int):
    data = load(os.path.join("results", f"dataset_{W}.bz2"))
    X = torch.tensor(data[0], dtype=torch.float32)
    y = torch.tensor(data[1], dtype=torch.float32)
    partition = torch.tensor(data[2], dtype=torch.float32)
    return X, y, partition



class ToyModel(nn.Module):
    def __init__(self, input_size: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=input_size-1)
        self.linear = nn.Linear(in_features=4, out_features=1, bias=False)
        weights = torch.zeros(4, 1, 2, 2)
        weights[0, 0, :, :] = 10 * torch.Tensor([[-1, -1], [1, 1]])
        weights[1, 0, :, :] = 10 * torch.Tensor([[1, -1], [1, -1]])
        weights[2, 0, :, :] = 10 * torch.Tensor([[1, 1], [-1, -1]])
        weights[3, 0, :, :] = 10 * torch.Tensor([[-1, 1], [-1, 1]])
        self.conv.weight = nn.Parameter(weights)
        self.conv.bias = nn.Parameter(-19 * torch.ones(4))
        self.linear.weight = nn.Parameter(torch.Tensor([-20.0, 20.0, -20.0, 20.0]))

    def forward(self, x: torch.Tensor):
        # x = x.reshape(-1, 1, self.input_size, self.input_size)
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        return self.linear(x).view(-1, 1)
