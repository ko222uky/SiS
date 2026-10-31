"""Unit-square geometry.

A packing for puzzle n is an (n, 3) float array of rows [x, y, angle]:
the square's center and its rotation in degrees.
"""

import numpy as np
import shapely
from shapely import Polygon

# stored angles are in degrees, which is assumed below
ANGLE_UNITS = "degrees"

# corner coordinates of the unit square, centered at the origin (0, 0).
# This is used to compute the unit square corner coordinates, given the
# centroid x,y and the rotation (in degrees). See corners().
_UNIT = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

def corners(params: np.ndarray) -> np.ndarray:
    """Return the (n, 4, 2) corner coordinates of each square.
    Recall each square is an [x, y, angle]
    """
    params = np.asarray(params, dtype=float)
    t = np.radians(params[:, 2]) # this is the angle column, in degrees.
    c, s = np.cos(t), np.sin(t)  # cos and sin of each square's angle

    # the basic idea is to take our unit square's coordinates from _UNIT,
    # which is a square centered at the origin, and then perform two operations to it:
    # (1) rotational transformation, which we can do with a matrix multiplication.
    #
    #   A rotational matrix uses cos and sin, and is defined as:
    #               R = [[c, -s],
    #                   [s,  c]]
    #   but here we have the transpose of R.
    #   because the mult we do is _UNIT @ Rt; we are working with row vectors, which is how
    #   our unit square _UNIT stores its corners.
    #
    # (2) translation, which we do by adding each square's center coordinates,
    #     (params[:, None, :2]) to the rotated corners.
    #     NOTE: since this confuses me time to time: params[:, None, :2] 
    #           is selecting the x, y coordinates of the square center. 
    #           since [:, :2] takes the first two rows of our [x, y, angle].
    #           but adding None adds a third axis, so that we have a a shape of (n, 1, 2).
    #    the rotated corners has shape (n, 4, 2), so we can broadcast the addition after
    #    adding our new third axis.
    rot_t = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)

    return _UNIT @ rot_t + params[:, None, :2] 

# I don't think I will actually need this function,
# but I"ll keep it here as an example of how
# shapely provides this functionality.
def gap(s1, s2):
    """Computes the shortest distance between two unit squares."""
    return Polygon(corners(s1)).distance(Polygon(corners(s2)))


def overlaps(s1, s2, tol: float = 1e-12) -> bool:
    """
    Check if a pair of squares overlaps.
    True if two unit squares [x, y, angle] overlap by more than tol area.
    Primarily used in the compaction algorithm.
    """
    # cheap circle checks first, using the distance between the two centers.
    # each square fits inside a circle of radius sqrt(2)/2 (through its corners),
    # and contains a circle of radius 1/2 (touching its sides). So:
    #   d >= sqrt(2) -> the outer circles don't touch, so the squares can't overlap.
    #   d <  1       -> the inner circles overlap, so the squares must overlap.
    d = np.hypot(s1[0] - s2[0], s1[1] - s2[1])
    if d >= np.sqrt(2):
        return False
    if d < 1:
        return True

    # otherwise do the exact polygon check. corners() wants an (n, 3) array,
    # so wrap each single [x, y, angle] row with [None].
    p1 = Polygon(corners(np.asarray(s1, dtype=float)[None])[0])
    p2 = Polygon(corners(np.asarray(s2, dtype=float)[None])[0])
    return p1.intersection(p2).area > tol



# important function for computing the cost. Given our set of squares, this finds the 
# containing square L's side length. The cost is s/sqrt(n) - 1. 
# NOTE: THIS DOES NOT CHECK VALIDITY OF CURRENT SOLUTION!
def container_side(params: np.ndarray) -> float:
    """
    Returns length in units of the smallest axis-aligned square containing every square.
    """
    # get corner coords of every square in current solution, and flatten it into list of points.
    # this is because we don't need the squares anymore; the container ONLY cares about the
    # points in space (so we can divorce the corner coordinate from the square!)
    # so points is a list of (x,y) of length 2n.
    points = corners(params).reshape(-1, 2)

    # ptp = "peak to peak" which takes a range like max - min
    # see def: "Range of values (maximum - minimum) along an axis."
    # computed down each x and y column separately, so we get a range of
    #
    #       width = max x - min x, height = max y - min y,
    #  
    # which is like a width and height.
    # Since our container L must be a SQUARE (and not a rectangle), then
    # we need to take whichever is the larger of max x - min x, max y - min y.
    # Thus, we get our square this way.
    return float(np.ptp(points, axis=0).max())

def polygons(params: np.ndarray) -> np.ndarray:
    """Return the squares as a numpy array of shapely Polygons."""
    return shapely.polygons(corners(params))

# this is the main validation checks. Need to define a tolerance.
# otherwise, it sticks with 0.0.
# Returns pairs of squares that overlap. ALso useful when visualizing the bad squares.
def overlapping_pairs(params: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Return (k, 2) index pairs i < j whose intersection area exceeds tol."""
    polys = polygons(params)
    i, j = shapely.STRtree(polys).query(polys, predicate="intersects")
    keep = i < j
    i, j = i[keep], j[keep]
    area = shapely.area(shapely.intersection(polys[i], polys[j]))
    bad = area > tol
    return np.column_stack([i[bad], j[bad]])


def grid(n: int) -> np.ndarray:
    """Trivial packing: row-major fill of a ceil(sqrt(n)) x ceil(sqrt(n)) grid."""
    k = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n)
    return np.column_stack([idx % k + 0.5, idx // k + 0.5, np.zeros(n)])
