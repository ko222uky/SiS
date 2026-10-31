"""
Module to maintain a store of the best-obtained square packing per puzzle, 
one CSV file per problem size n.

A potential improvement can be stored via a call with `offer`; only valid, strictly better packings are kept, so
improvements from any run accumulate.

Format is based on a submission of the store, wherein the submission is a merge of the store.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sis import submission
from sis.score import N_MAX, grid_score, puzzle_score
from sis.validate import is_valid

DEFAULT_ROOT = Path("solutions")

# Ignore "improvements" below float noise.
MIN_GAIN = 1e-12

class SolutionStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, n: int) -> Path:
        return self.root / f"n{n:04d}.csv"

    def get(self, n: int) -> np.ndarray | None:
        path = self._path(n)
        if not path.exists():
            return None
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)

    def score(self, n: int) -> float:
        params = self.get(n)
        return grid_score(n) if params is None else puzzle_score(n, params)

    def offer(self, n: int, params: np.ndarray) -> bool:
        """Save params if valid and better than the stored packing."""
        params = np.asarray(params, dtype=float)
        if not is_valid(n, params):
            return False
        if puzzle_score(n, params) >= self.score(n) - MIN_GAIN:
            return False

        # safe saving.
        tmp = self._path(n).with_suffix(".tmp") # saves a temp file
        np.savetxt(tmp, params, delimiter=",", fmt="%.17g", header="x,y,angle", comments="")
        tmp.replace(self._path(n))              # replace with temp file
        return True

    def all(self, n_max: int = N_MAX) -> dict[int, np.ndarray]:
        return {n: p for n in range(1, n_max + 1) if (p := self.get(n)) is not None}

    def summary(self, n_max: int = N_MAX) -> pd.DataFrame:
        rows = [(n, self.score(n), grid_score(n)) for n in range(1, n_max + 1)]
        df = pd.DataFrame(rows, columns=["n", "score", "grid_score"])
        df["gain"] = df["grid_score"] - df["score"]
        return df

    def export(self, path: str | Path, n_max: int = N_MAX) -> None:
        submission.write(self.all(n_max), path, n_max)
