"""Strict validity checks for a single puzzle's packing."""

import numpy as np

from sis.geometry import overlapping_pairs

# maximum intersection area treated as "touching". 
NONZERO_OVERLAP_TOL = 1e-12


def problems(n: int, params: np.ndarray, tol: float = NONZERO_OVERLAP_TOL) -> list[str]:
    """
    Return a list of human-readable problems, i.e., issues with our solution.
    An empty list means no problems and hence valid."""
    params = np.asarray(params, dtype=float)

    # n should equal param length
    if params.shape != (n, 3):
        return [f"expected shape ({n}, 3), got {params.shape}"]

    # if we have any infinite values, that's a problem.
    if not np.isfinite(params).all():
        return ["non-finite values"]

    # overlapping pairs are problems:
    # finds overlapping pairs 
    pairs = overlapping_pairs(params, tol)

    # if we DO have overlapping pairs, then report what they are.
    if len(pairs):
        shown = ", ".join(f"{i}-{j}" for i, j in pairs[:5])
        more = f" (+{len(pairs) - 5} more)" if len(pairs) > 5 else ""
        return [f"{len(pairs)} overlapping pairs: {shown}{more}"]
    return []


def is_valid(n: int, params: np.ndarray, tol: float = NONZERO_OVERLAP_TOL) -> bool:
    """
    Wraps `problems()` to act as a boolean check.
    """
    # We only have a valid solution if problems() returns EMPTY.
    # Because if we have a problem then... we have a problem.
    # so is_valid() returns false if we find problems with our solution.
    return not problems(n, params, tol)


# TODO: cost function. If there's overlap, perhaps make that cost infinite.
