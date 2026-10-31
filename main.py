"""

Example usage of the sis package:
    * Container size
    * Scoring
    * Solution format
    * Solution validation
    * Solution scoring
    * Storing solutions in Kaggle submission format
        - Store and export solutions
        - Store and export scores

Run with:
```
uv run python main.py
```

TODO: Add an optimization algorithm.

    - Complete search, simulated annealing.

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
    # packing is an (n, 3) array of [x, y, angle_degrees] per square.
    # note: the grid score IS the fallback score
    n = 5
    baseline = grid(n)
    print(f"grid packing for n={n}:\n{baseline}")
    print(f"container side: {container_side(baseline)}, score: {puzzle_score(n, baseline):.6f}")

    # better packing: four corner squares plus one tilted 45 degrees in
    # the middle (the known optimum for n=5, side 2 + sqrt(2)/2).
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

    # validation explains what is wrong with a bad packing.
    clash = better.copy()
    clash[4, :2] -= 0.3
    print(f"\nshifted tilted square into a corner square: {problems(n, clash)}")

    # the store keeps only valid, strictly better packings per n.
    # a throwaway directory here; the real store defaults to ./solutions.
    with tempfile.TemporaryDirectory() as tmp:
        store = SolutionStore(Path(tmp) / "solutions")
        print("Example of offering the three solutions we made: ")
        print(f"\noffer grid:    accepted={store.offer(n, baseline)}  (no better than fallback)")
        print(f"offer clash:   accepted={store.offer(n, clash)}  (invalid)")
        print(f"offer tilted:  accepted={store.offer(n, better)}")
        print(f"stored score for n={n}: {store.score(n):.6f} (grid: {grid_score(n):.6f})")

        #    export a full submission (missing puzzles filled with the grid)
        #    and score it the way Kaggle would.
        out = "submission.csv" # <-- this stores the solutions for the problem n
        store.export(out, n_max=10)
        sols = submission.read(out)
        grid_total = sum(grid_score(k) for k in range(1, 11))
        print(f"\nexported puzzles 1..10 -> total {total(sols, n_max=10):.6f} "
              f"(grid total {grid_total:.6f})")

        #    export the score per puzzle n alongside the grid baseline.
        scores_out = "scores.csv" # <-- this stores the obtained scores per problem n
        store.summary(n_max=10).to_csv(scores_out, index=False)
        print(f"per-puzzle scores -> {scores_out}")


if __name__ == "__main__":
    main()
