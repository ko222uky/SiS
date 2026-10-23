"""Example usage of the sis package.

Run with:  uv run python main.py
"""

import math
import tempfile
from pathlib import Path

import numpy as np

from sis import submission
from sis.geometry import container_side, corners, grid
from sis.score import grid_score, puzzle_score, total
from sis.store import SolutionStore
from sis.validate import is_valid, problems


def main():
    # 1. A packing is an (n, 3) array of [x, y, angle_degrees] per square.
    n = 5
    baseline = grid(n)
    print(f"grid packing for n={n}:\n{baseline}")
    print(f"container side: {container_side(baseline)}, score: {puzzle_score(n, baseline):.6f}")

    # 2. A better packing: four corner squares plus one tilted 45 degrees in
    #    the middle (the known optimum for n=5, side 2 + sqrt(2)/2).
    s = 2 + math.sqrt(2) / 2
    better = np.array([
        [0.5, 0.5, 0],
        [s - 0.5, 0.5, 0],
        [0.5, s - 0.5, 0],
        [s - 0.5, s - 0.5, 0],
        [s / 2, s / 2, 45],
    ])
    print(f"\ntilted packing for n={n}: valid={is_valid(n, better)}, "
          f"side={container_side(better):.6f}, score={puzzle_score(n, better):.6f}")
    print(f"corners of the tilted square:\n{corners(better)[4].round(4)}")

    # 3. Validation explains what is wrong with a bad packing.
    clash = better.copy()
    clash[4, :2] -= 0.3
    print(f"\nshifted tilted square into a corner square: {problems(n, clash)}")

    # 4. The store keeps only valid, strictly better packings per n.
    #    A throwaway directory here; the real store defaults to ./solutions.
    with tempfile.TemporaryDirectory() as tmp:
        store = SolutionStore(Path(tmp) / "solutions")
        print(f"\noffer grid:    accepted={store.offer(n, baseline)}  (no better than fallback)")
        print(f"offer clash:   accepted={store.offer(n, clash)}  (invalid)")
        print(f"offer tilted:  accepted={store.offer(n, better)}")
        print(f"stored score for n={n}: {store.score(n):.6f} (grid: {grid_score(n):.6f})")

        # 5. Export a full submission (missing puzzles filled with the grid)
        #    and score it the way Kaggle would.
        out = Path(tmp) / "submission.csv"
        store.export(out, n_max=10)
        sols = submission.read(out)
        grid_total = sum(grid_score(k) for k in range(1, 11))
        print(f"\nexported puzzles 1..10 -> total {total(sols, n_max=10):.6f} "
              f"(grid total {grid_total:.6f})")


if __name__ == "__main__":
    main()
