"""
Matplotlib plots for annealing runs and square packings.
Every function takes an optional `ax` (or returns a new figure) so the plots
can be dropped into notebooks or combined into larger figures.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch, Rectangle
from matplotlib.ticker import StrMethodFormatter

from sis.geometry import container_side, corners, overlapping_pairs
from sis.optimize import AnnealResults
from sis.score import puzzle_score

# predefine colors here, to make changing things easy.
# this almost looks like the defaults in Excel, which work
_INK = "#0b0b0b"
_INK_2 = "#52514e"      # secondary text, axes
_GRID = "#e4e3df"
_BLUE = "#2a78d6"       # primary series
_ORANGE = "#eb6834"     # second series
_RED = "#e34948"        # reserved for "something is wrong" (overlaps); i dont expect this to happen since we validate solutions


# define and set a consistent style to apply across the board
def _style(ax, xlabel=None, ylabel=None):
    """Recessive axes: light grid behind the data, no top/right spines."""
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=_INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK_2)

#############################################
#               MAIN METRICS
# individual plots to capture metrics / 
# parameters during the simulated annealing
# rounds. To be used in the main plotting function.
#############################################
def plot_temperatures(results: AnnealResults, ax=None):
    """Temperature per round. Log scale, since geometric cooling is a straight line there."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    rounds = np.arange(len(results.temperatures))
    ax.plot(rounds, results.temperatures, color=_BLUE, linewidth=2)
    # ax.set_yscale("log") # changed my mind; I don't like the log scale visual
    ax.set_title("Temperature", loc="left", color=_INK)
    _style(ax, "round", "T")
    return ax


def plot_scores(results: AnnealResults, ax=None):
    """Candidate score per round and the best score found so far (line)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    scores = np.asarray(results.scores, dtype=float)
    rounds = np.arange(len(scores))

    # running minimum = best candidate seen up to each round. (anneal() also compares
    # the starting solution, so its final best can be slightly lower than this line.)
    best = np.minimum.accumulate(scores)

    ax.scatter(rounds, scores, s=14, color=_BLUE, alpha=0.6, linewidths=0, label="candidate")
    ax.step(rounds, best, where="post", color=_ORANGE, linewidth=2, label="best so far")
    ax.annotate(f"{best[-1]:.4f}", (rounds[-1], best[-1]), xytext=(0, 5),
                textcoords="offset points", ha="right", color=_INK, fontsize=9)
    ax.set_title("Score (lower is better)", loc="left", color=_INK)
    ax.legend(frameon=False, labelcolor=_INK_2)
    _style(ax, "round", "score")
    return ax


def plot_probabilities(results: AnnealResults, ax=None):
    """
    Acceptance probability exp(-d/T) per round.
    Improvements (d < 0) give values above 1, so we clip them.
    Recall that improvements are always accepted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    p = np.clip(np.asarray(results.probabilities, dtype=float), 0, 1) # Clip probabilities when candidate is an improvement.
    rounds = np.arange(len(p))
    ax.plot(rounds, p, color=_BLUE, linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Acceptance probability", loc="left", color=_INK)
    _style(ax, "round", "P(accept)")
    return ax


# Plots a square packing solution. ALso used in the main plotting function
# for AnnealResults.
def plot_solution(params: np.ndarray, ax=None, show_overlaps: bool = True, title: str | None = None):
    """
    This is basically plotting the polygons.
    See the following:
        * https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html
    
        * https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

    """
    params = np.asarray(params, dtype=float)
    n = len(params)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6.4), layout="constrained")

    # use our geometry functions to get spatial params for our square corner points
    # and our bounding box L
    pts = corners(params)                  # (n, 4, 2) for n squares, each with four corners, with each corner an x,y tuple
    L = container_side(params)          
    x0, y0 = pts.reshape(-1, 2).min(axis=0)  # bottom-left corner of the container

    bad = set()
    if show_overlaps:
        bad = set(overlapping_pairs(params, 1e-12).ravel().tolist())

    # for each squares' corners, we can build a polygon patch! Wow!
    for i, quad in enumerate(pts):
        color = _RED if i in bad else _BLUE

        # no need to specify rotation, just the four points.
        ax.add_patch(PolygonPatch(quad, closed=True, facecolor=color, alpha=0.25,
                                  edgecolor=color, linewidth=1))

    # the container: anchored at the min x, min y of all corners, side L.
    # no need for rotation, since it has a fixed default rotation.
    ax.add_patch(Rectangle((x0, y0), L, L, fill=False, edgecolor=_INK_2,
                           linewidth=1.5, linestyle="--"))

    # dimension line for L, just under the box.
    off = 0.06 * L
    ax.annotate("", xy=(x0, y0 - off), xytext=(x0 + L, y0 - off),
                arrowprops=dict(arrowstyle="<->", color=_INK_2, linewidth=1))
    ax.text(x0 + L / 2, y0 - off, f"L = {L:.4f}", ha="center", va="top",
            color=_INK, fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

    pad = 0.15 * L
    ax.set_xlim(x0 - pad, x0 + L + pad)
    ax.set_ylim(y0 - pad, y0 + L + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    # title, with the score as a subtitle underneath.
    ax.set_title(f"{title}: {n} squares" if title else f"{n} squares", color=_INK, pad=20)
    subtitle = f"score = {puzzle_score(n, params):.5f}"
    if bad:
        subtitle += f"   ({len(bad)} squares overlapping)"
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            color=_RED if bad else _INK_2, fontsize=10)
    return ax


################################################################################
# THIS IS THE MAIN PLOTTING FUNCTION. 
# It's simply and captures an entire annealing run.
# It stacks the plots into a subplot mosaic.
################################################################################
def plot_annealing(results: AnnealResults):
    """
    The starting (top) and best (bottom) solutions on the left; temperature, scores
    and acceptance probability stacked on a shared round axis on the right.
    """
    fig, axes = plt.subplot_mosaic(
        # this is getting funky, so
        # solutions go to the left, with the metrics to the right.
        # 6 rows so the left column splits in halves (start/end)
        # while the right column splits in thirds (one per metric).
          # the starting, initial solution paired against metrics
          # with ending solution below.
          # the dimensions are basically determined by 2*3 = 6 units in this mosaic
        [["start", "T"],
         ["start", "T"],
         ["start", "score"],

         ["end", "score"],
         ["end", "p"],
         ["end", "p"]],
        figsize=(13, 13), width_ratios=[1, 1.2], layout="constrained",
    )
    # share the round axis for temperature, probability, and scores
    axes["score"].sharex(axes["T"])
    axes["p"].sharex(axes["T"])

    # here, we call each plotting function and it takes up the mosaic defined by the axes aliases
    plot_solution(results.starting_solution, axes["start"], title="Start")
    plot_solution(results.solution, axes["end"], title="Best")
    plot_temperatures(results, axes["T"])
    plot_scores(results, axes["score"])
    plot_probabilities(results, axes["p"])
    for key in ("T", "score"):
        axes[key].set_xlabel("")
        axes[key].tick_params(labelbottom=False)
    return fig


#############################################
#               L-BFGS PLOTS
# same idea as the annealing plots, but for a
# gradient run: the loss is recorded per iteration.
#############################################

# helper function for visualizing loss, using log scaling due to big drop
def plot_losses(losses, ax=None):
    """Loss per L-BFGS iteration (iteration 0 is the starting solution)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    losses = np.asarray(losses, dtype=float)
    its = np.arange(len(losses))
    ax.plot(its, losses, color=_BLUE, linewidth=2)
    ax.annotate(f"{losses[-1]:.4f}", (its[-1], losses[-1]), xytext=(0, 5),
                textcoords="offset points", ha="right", color=_INK, fontsize=9)
    # the loss usually drops A LOT (overlaps get pushed apart = big loss change),
    # so a log scale keeps run readable.
    if losses.min() > 0:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))   # 6 instead of 6x10^0
        ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:g}"))
        ax.tick_params(which="minor", colors=_INK_2, labelsize=8)
    ax.set_title("Loss (lower is better)", loc="left", color=_INK)
    _style(ax, "iteration", "loss")
    return ax


def plot_descent(start: np.ndarray, end: np.ndarray, losses):
    """Starting (left) and final (right) solutions, with the loss curve underneath."""
    fig, axes = plt.subplot_mosaic(
        [["start", "end"],
         ["loss", "loss"]],
        figsize=(13, 11), height_ratios=[2, 1], layout="constrained",
    )
    plot_solution(start, axes["start"], title="Start")
    plot_solution(end, axes["end"], title="L-BFGS")
    plot_losses(losses, axes["loss"])
    return fig


