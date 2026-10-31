"""
Competition scoring: per puzzle s / sqrt(n) - 1, summed; lower is better.

Missing or invalid puzzles are scored as the trivial grid packing.
"""
 
import math

import numpy as np

from sis.geometry import container_side
from sis.validate import is_valid

N_MAX = 1000


def puzzle_score(n: int, params: np.ndarray) -> float:
    return container_side(params) / math.sqrt(n) - 1

def grid_score(n: int) -> float:
    """The score of squares packed as a grid. Default score."""
    return math.ceil(math.sqrt(n)) / math.sqrt(n) - 1

def scores(solutions: dict[int, np.ndarray], n_max: int = N_MAX) -> dict[int, float]:
    """Score every puzzle 1..n_max, falling back to the grid score."""
    out = {}
    for n in range(1, n_max + 1):
        params = solutions.get(n)
        if params is not None and is_valid(n, params):
            out[n] = puzzle_score(n, params)
        else:
            out[n] = grid_score(n)
    return out

def total(solutions: dict[int, np.ndarray], n_max: int = N_MAX) -> float:
    return sum(scores(solutions, n_max).values())
