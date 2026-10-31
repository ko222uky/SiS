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

Also demonstrates the simulated annealing algorithm based on simple
single-square manipulations after exploding out the solution.

"""

import math
import tempfile
from pathlib import Path

import numpy as np

import sis.visualize as sv
from sis import submission
from sis.geometry import container_side, corners, grid
from sis.score import grid_score, puzzle_score, total
from sis.store import SolutionStore
from sis.validate import is_valid, problems
from sis.optimize import separate, anneal, init_random


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


    # Take the clash and separate it:
    print('=' * 50)
    print("SEPARATING THE CLASH")
    sep_clash = separate(clash)
    print(sep_clash)
    print(f"Separated clash packing for n={n}: valid={is_valid(n, sep_clash)}, "
          f"side={container_side(sep_clash):.6f}, score={puzzle_score(n, sep_clash):.6f}")
    print(f"separated squares in clash: {problems(n, sep_clash)}")
    print('=' * 50)

    print('#' * 50)
    print("\n\nDemonstration of the simulated annealing algorithm")

    # PARAMS:
    T = 1.0
    cooling = 0.95
    perturbance = 1.5
    rounds=300

    init = init_random(1000)

    results = anneal(
        n, 
        init, 
        T = T, 
        cooling=cooling, 
        perturbance=perturbance, 
        rounds=rounds, 
        seed=None
    )
    
    fig = sv.plot_annealing(results)
    fig.savefig("demo_anneal.png")
    candidate_best = results.solution
    print(candidate_best)
    print(f"Solution from annealing: packing for n={n}: valid={is_valid(n, candidate_best)}, "
          f"side={container_side(candidate_best):.6f}, score={puzzle_score(n, candidate_best):.6f}")
    print(f"Any problems? We'll see them here (though we shouldnt!!): {problems(n, candidate_best)}")
    print('#' * 50)


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
