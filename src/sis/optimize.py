"""
Contains functions for applying simulated annealing to the square packing problem.
"""
import math
from typing import Callable
import numpy as np
import shapely
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from sis.geometry import overlaps, grid, polygons, corners, container_side
from sis.score import puzzle_score
from sis.validate import is_valid
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

def block_start(n, b=None, seed=None):
    """
    A (b x b) aligned grid in the corner, plus the remaining squares placed
    randomly (without overlap) in an L-shaped strip around it.
   
    The idea is to keep a fixed grid core and move surrounding blocks, assuming that
    the grid packing is already close to optimal. Thus, we expect improvements in surrounding blocks.
    """
    rng = np.random.default_rng(seed)
    k = int(np.ceil(np.sqrt(n)))
    b = k - 1 if b is None else b
    block = grid(b * b)                         # centers at 0.5 ... b-0.5, angle 0

    free = []
    hi = b + 2                                  # loose outer box for the strip
    while len(free) < n - b * b:
        x, y = rng.uniform(0.5, hi, 2)
        if x < b + 0.8 and y < b + 0.8:         # inside (or touching) the block: skip
            continue
        s = np.array([x, y, rng.uniform(0, 90)])
        if not any(overlaps(s, t) for t in [*block, *free]):
            free.append(s)

    sq = np.vstack([block, free])
    fixed = np.arange(n) < b * b                # first b*b rows are the block
    return sq, fixed

def perturb(sq, factor=1.3, fixed=None):
    """
    Perturbs the current solution by nudging by a small scaling factor away from the centroid.
    Factor can scale.
    """
    move = slice(None) if fixed is None else ~fixed
    
    sq = sq.copy()

    # get centroid
    c = sq[:, :2].mean(axis=0)
    # nudge things away from centraid by a scaled factor.
    # but if the square is fixed, it is NOT perturbed.
    sq[move, :2] = c + (sq[move, :2] - c) * factor
    return sq


def compact(
        sq: np.ndarray,
        overlaps: Callable = overlaps, 
        steps=5000, 
        step=0.02, 
        seed=None,
        fixed=None,
        rotation='random',
        p45=0.1):
    """
    Compacts by randomly selecting a square and moving it towards the centroid of all squares.
    This is somewhat of a blind algorithm since each square moves by a step size.

    Edited to add a pivot. If we collide with a square, try a 90 degree turn, randomly.
    If THAT results in a collision, then we revert.

    Option to not move fixed squares in cases where grid layout is already optimal.

    Args
    ====

        rotation : 'random' gives rotations with mean 0 and std dev 45 degrees. '45' gives a small percent chance to rotate a square by 45 degrees.

    """
    rng = np.random.default_rng(seed)
    sq = sq.copy()
    n = len(sq)

    # 'fixed' specifies which squares we can move;
    # otherwise, we ignore it all and every square can move
    if fixed is None:
        # makes all squares movable if fixed is None
        fixed = np.zeros(n, dtype=bool)
    movable = np.flatnonzero(~fixed)  
    if movable.size == 0:
        # shouldn't expect this, but add this case to guard
        return sq 

    for _ in range(steps):
        #if _ % 1000 == 0:
        #    print(f"Compaction step: {_}")
        i = rng.choice(movable)     # pick a random index from the movable squares
        orig_sq = sq[i].copy()      # the square PRIOR to any changes. Save the state to allow reverting.
        c = sq[:, :2].mean(axis=0)  # the centroid (mean of all centers) of our squares
        direction = c - sq[i, :2]
        sq[i, :2] += step * (direction)      # move current square toward the center

        if rotation == 'random':
            sq[i, 2] += rng.normal(0, 12.5)         # small random rotation

        elif rotation == '45':
            # with probability p45, turn the square 45 degrees.
            # a square looks the same every 90 degrees, so +45 and -45 give the same result.
            if rng.random() < p45:
                sq[i, 2] += 45

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

def anneal(
        n: int, 
        sq: np.ndarray, 
        overlaps: Callable = overlaps, 
        rounds=100, 
        T=0.1, 
        T_method='inv_log',
        cooling=0.97, 
        perturbance=1.3,
        fixed=None,
        seed=None,
        **compact_options):
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

    `T_method` includes 'mult' for multiplicative reduction with `cooling`, 'inv_log' for T = 

    """
    T0 = T
    rng = np.random.default_rng(seed)
    if fixed is None:
        cur = compact(separate(sq), overlaps, **compact_options)
    else:
        cur = compact(sq, overlaps, fixed=fixed, **compact_options)
    best = cur # we keep track of the best solution obtained so far.
    temperatures = []
    probabilities = []
    candidate_scores = []

    for j in range(rounds):
        
        print(f"Annealing round {j}, T={T}")
        cand = compact(perturb(cur, factor=perturbance, fixed=fixed), fixed=fixed, **compact_options)

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
        if T_method == 'mult':
            T *= cooling
        elif T_method == 'inv_log':
            T = (T0 * cooling) / math.log(j+2)
    return results


#######################################################
# IMPLEMENTING A CLASSICAL METHOD: GRADIENT DESCENT
#######################################################
# TODO: define a loss function that we can use for a gradient descent method.
# In this method, loss should consider (1) the size of container L,
# (2) the overlap of squares with each other, and maybe (3) squares the exceed the container L's bounds.
# So, loss = L + overlap of squares i,j + overlap of square i beyond L 
#
# Thus, we have three penalties to add that gives us a loss.
#
# The main variables are x, y, and theta.
# To simplify things, we let shapely compute the overlap,
# so we don't need to manually write the formula...

def loss(sq: np.ndarray, L, lam=10.0):
    """
    Returns the loss based on current square configurations.
    """

    sq = np.asarray(sq, dtype=float)

    # exact overlap area over all pairs i < j, in one vectorized call
    # convert squares to shapely polygons.
    polys = polygons(sq)
    i, j = np.triu_indices(len(sq), k=1) # succinct way to get (i, j) for i < j; it's the upper triangle

    # FIRST PENALTY
    # vectorized shapely intersections between (i, j); get area then sum.
    overlap = shapely.area(shapely.intersection(polys[i], polys[j])).sum()

    # SECOND PENALTY
    # Compute square corners; we know that all corners MUST be inside L.
    # every corner must sit inside [0, L] x [0, L]
    c = corners(sq)

    # note: this is computed element-wise; so compute this for x term,
    # then for y term, and sum them both to get the total penality for a given square,
    # then sum all squares to get total pentalty for all squares.
    outside = (np.maximum(0, -c)**2 + np.maximum(0, c - L)**2).sum()

    # THIRD PENALTY: container size L
    return L + lam * (overlap + outside)


def flat_loss(p, n, lam=10.0):
    """
    Wraps `loss()` and allows it to be called if `p` is a 1D array. 

    E.g:

        sq = [x1, y1, angle1, ..., xn, yn, anglen, L]

    The lambda coefficient tells us how much violations of boundaries 
    (square overlaps, boundary overlaps) should contribute to loss,
    compared to the box size L, which we also want to reduce.

    """
    # so sq[-1], the last element, holds L0 (our initial L) or the current L in the descent.
    # and all the squares are everything else, hence :-1.
    # This single 1-D array is reshaped to n squares, each representing a triplet
    # of x, y, angle.
    return loss(p[:-1].reshape(n, 3), p[-1], lam)

def lbfgs(sq: np.ndarray, n: int, lam):
    # placeholder for increasing lambda over time.
    L0 = container_side(sq)
    sq0 = np.append(sq.ravel(), L0) # this attaches L0 to the very end
    losses = [flat_loss(sq0, n, lam)]
    def callback(intermediate_result):
        losses.append(intermediate_result.fun) 
    res = minimize(flat_loss, sq0, args=(n, lam), method="L-BFGS-B", callback=callback)
    sq_opt = res.x[:-1].reshape(n, 3)       
    L_opt = res.x[-1]
    return sq, sq_opt, L_opt, losses

def seq_lbfgs(sq: np.ndarray, n: int, lam_start, lam_mult=10.0, lam_stages=4):
    """Repeated L-BFGS, raising lam by lam_mult each stage and warm-starting from the last result."""
    p = np.append(sq.ravel(), container_side(sq))   # [x1, y1, a1, ..., L]
    losses = []
    def callback(intermediate_result):
        losses.append(intermediate_result.fun)

    lam = lam_start
    for _ in range(lam_stages):
        losses.append(flat_loss(p, n, lam))          # loss at the start of this stage
        res = minimize(flat_loss, p, args=(n, lam), method="L-BFGS-B", callback=callback)
        p = res.x                                     # warm start the next stage
        if is_valid(n, p[:-1].reshape(n, 3)):
            break
        lam *= lam_mult

    return sq, p[:-1].reshape(n, 3), p[-1], losses


    

    