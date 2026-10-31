"""Read and write the submission CSV: id,n,square_id,x,y,angle."""

from pathlib import Path

import numpy as np
import pandas as pd

from sis.geometry import grid
from sis.score import N_MAX

COLUMNS = ["id", "n", "square_id", "x", "y", "angle"]


def read(path: str | Path) -> dict[int, np.ndarray]:
    df = pd.read_csv(path).sort_values(["n", "square_id"])
    return {
        int(n): g[["x", "y", "angle"]].to_numpy(dtype=float)
        for n, g in df.groupby("n", sort=True)
    }

def write(solutions: dict[int, np.ndarray], path: str | Path, n_max: int = N_MAX) -> None:
    frames = []
    for n in range(1, n_max + 1):
        params = solutions.get(n)
        params = grid(n) if params is None else np.asarray(params, dtype=float)
        frames.append(pd.DataFrame({
            "n": n,
            "square_id": np.arange(n),
            "x": params[:, 0],
            "y": params[:, 1],
            "angle": params[:, 2],
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "id", np.arange(len(df)))
    df.to_csv(path, index=False, float_format=None)
