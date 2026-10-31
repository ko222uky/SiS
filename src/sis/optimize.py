"""
Contains functions for applying simulated annealing to the square packing problem.
"""
import dataclasses
from typing import Callable
import numpy as np
from scipy.spatial.distance import pdist
from sis.geometry import overlaps
from sis.score import puzzle_score
# Minimum separation allowing full rotation of a unit square
# is 2 * 1/sqrt(2) = sqrt(2)
_MIN_SEP = np.sqrt(2)


class AnnealResults():
    def __init__(self, 
        solution: np.ndarray, 
        starting_solution: np.ndarray,
        temperatures: list, 
        scores: list,
        probabilities: list):
        self.starting_solution = starting_solution
        self.solution = solution
        self.temperatures = temperatures
        self.scores = scores
        self.probabilities = probabilities

# this might be redundant in light of the perturbation, but the idea is that this is a more
# extreme form of separating out squares.
def separate(params: np.ndarray):
    """
    Separates squares by scaling the current layout.
    This drastically alters the current solution.
    """
    sq = np.asarray(params, dtype=float).copy()
    xy = sq[:, :2] # get our xy of our squares,
    d = pdist(xy).min() # compute pairwise distances,
                        # and min() gets the least,
                        # so we get the distance between a square and its closest neighbor
    sq[:, :2] *= _MIN_SEP / d
    return sq

def init_random(n, seed=None):
    """
    Random initialization of a solution.
    Estimates a box with side L = sqrt(2*n) (and thus area 2n).
    Square centers are drawn within the uniform range of this container side L.
    Returns squares.
    """
    rng = np.random.default_rng(seed)
    L = np.sqrt(2 * n)                    # box with about the right area
    xy = rng.uniform(0, L, size=(n, 2))
    angle = rng.uniform(0, 90, size=n)
    return np.column_stack([xy, angle])

def perturb(sq, factor=1.3):
    """
    Perturbs the current solution by nudging by a small scaling factor away from the centroid.
    Factor can scale.
    """
    sq = sq.copy()
    # get centroid
    c = sq[:, :2].mean(axis=0)
    # nudge things away from centraid by a scaled factor.
    sq[:, :2] = c + (sq[:, :2] - c) * factor
    return sq


def compact(sq: np.ndarray, overlaps: Callable, steps=5000, step=0.02, seed=None):
    """
    Compacts by randomly selecting a square and moving it towards the centroid of all squares.
    This is somewhat of a blind algorithm since each square moves by a step size.

    Edited to add a pivot. If we collide with a square, try a 90 degree turn, randomly.
    If THAT results in a collision, then we revert.
    """
    rng = np.random.default_rng(seed)
    sq = sq.copy()
    n = len(sq)
    for _ in range(steps):
        #if _ % 1000 == 0:
        #    print(f"Compaction step: {_}")
        i = rng.integers(n)
        orig_sq = sq[i].copy()      # the square PRIOR to any changes. Save the state to allow reverting.
        c = sq[:, :2].mean(axis=0)  # the centroid (mean of all centers) of our squares
        direction = c - sq[i, :2]
        sq[i, :2] += step * (direction)      # move current square toward the center

        sq[i, 2] += rng.normal(0, 2)             # small random rotation

        def collides(i):
            return any(overlaps(sq[i], sq[j]) for j in range(n) if j != i)

        # check if we have ANY overap between squares i, j. If we do, undo the move we just performed.
        if collides(i):
            # try a sideway pivot to escape getting stuck in traffic
            # build a 90 degree rotation matrix, but have a random choice that it's left or right rotation.
            side = np.array([-direction[1], direction[0]]) * rng.choice([-1, 1])
            sq[i, :2] = orig_sq[:2] + step * side

            if collides(i):
                sq[i] = orig_sq
    return sq

def anneal(n: int, sq: np.ndarray, overlaps: Callable = overlaps, rounds=100, T=0.1, cooling=0.97, perturbance=1.3, seed=None):
    """
    Simulated Annealing algorithm for packing squares in squares.
    Given an initial packing of squares, the basic idea is to:
        (1) Separate the initial packing then compact it
        (2) For N rounds, we perturb the current solution the compact it
            to get a candidate solution.
        (3) Compare the candidate solution to current solution.
        (4) Accept candidate as current if:
            a. The candidate solution is better
            b. The worst solution wins the gamble of acceptance probability.
        (5) Reduce the temperature by scaling with the cooling factor

    The acceptance probability is exp(-d/T), meaning that:
        i. High temperatures = Higher Probability
        ii. Worse solutions (higher d) = Lower Probability

    """
    rng = np.random.default_rng(seed)
    cur = compact(separate(sq), overlaps)
    best = cur # we keep track of the best solution obtained so far.
    temperatures = []
    probabilities = []
    candidate_scores = []

    for _ in range(rounds):
        
        print(f"Annealing round {_}, T={T}")
        cand = compact(perturb(cur, factor=perturbance), overlaps)

        # compare score difference. If candidate is lower, then
        # d will be negative
        d = puzzle_score(n, cand) - puzzle_score(n, cur)

        # of candidate is better OR 
        # the acceptance probability rolls in favor of the solution regardless,
        # then set the current solution as the candidate
        if d < 0 or rng.random() < np.exp(-d / T):
            cur = cand

        if puzzle_score(n, cur) < puzzle_score(n, best):
            best = cur

        temperatures.append(T)
        probabilities.append(np.exp(-d / T))
        candidate_scores.append(puzzle_score(n, cand))

        results = AnnealResults(
            solution=best,
            starting_solution=sq,
            temperatures=temperatures,
            probabilities=probabilities,
            scores=candidate_scores
        )

        T *= cooling
    return results

    