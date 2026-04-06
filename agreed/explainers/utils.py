"""
Copyright © 2026 Thales. All rights reserved.

Licensed under the Thales Non Production License Agreement; you may not

use this file except in compliance with the License. See LICENSE File provided with this software.
"""

from itertools import chain, combinations
from typing import List, Tuple, Dict, Any
import torch
import numpy as np

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


def ravel(tuples: List[Tuple[Any, ...]] | List[List[Any]]) -> List[Any]:
    """
    turn a tuple of tuples ((a, b), (c,)) into a single tuple (a, b, c)
    turn a list of list [[a, b], [c]] into a single list [a, b, c]
    """
    out = []
    for seq_tuple in tuples:
        for element in seq_tuple:
            out.append(element)
    return out


def disagreement_for_instance(
        attr_1: np.ndarray | torch.Tensor,
        attr_2: np.ndarray | torch.Tensor,
        metric: str = "l2"
        ) -> float:
    """
    Compute the disagreement between two attributions.
    A value of zero implies perfect agreement.

    Parameters
    ---------
    attr_1:
        A (d,) array of attributions
    attr_2:
        A (d,) array of attributions
    metric:
        The type of metric (l2, cosine)
    """
    assert isinstance(attr_1, (torch.Tensor, np.ndarray))
    assert isinstance(attr_2, (torch.Tensor, np.ndarray))
    if isinstance(attr_1, torch.Tensor):
        attr_1 = attr_1.numpy()
    if isinstance(attr_2, torch.Tensor):
        attr_2 = attr_2.numpy()
    if metric == "l2":
        return float(np.sum((attr_1 - attr_2)**2))
    elif metric == "cosine":
        norm_1 = np.linalg.norm(attr_1)
        norm_2 = np.linalg.norm(attr_2)
        product = np.sum(attr_1 * attr_2)
        return 1 - 0.5 * float(product / (norm_1 * norm_2))
    else:
        raise Exception("Metrics must be l2 or cosine")

def disagreement_for_data(
        attr_1: np.ndarray | torch.Tensor,
        attr_2: np.ndarray | torch.Tensor,
        metric: str = "l2"
        ) -> float:
    """
    Compute the disagreement between two attributions.
    A value of zero implies perfect agreement.

    Parameters
    ---------
    attr_1:
        A (N, d) array of attributions
    attr_2:
        A (N, d,) array of attributions
    metric:
        The type of metric (l2, cosine)
    """
    assert isinstance(attr_1, (torch.Tensor, np.ndarray))
    assert isinstance(attr_2, (torch.Tensor, np.ndarray))
    if isinstance(attr_1, torch.Tensor):
        attr_1 = attr_1.numpy()
    if isinstance(attr_2, torch.Tensor):
        attr_2 = attr_2.numpy()
    disagreement_score = 0.0
    N = attr_1.shape[0]
    for i in range(N):
        disagreement_score += disagreement_for_instance(attr_1[i], attr_2[i], metric=metric)
    return disagreement_score / N
