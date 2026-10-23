import math

import numpy as np
import pytest

from sis import submission
from sis.geometry import container_side, grid
from sis.score import grid_score, puzzle_score, total
from sis.store import SolutionStore
from sis.validate import is_valid, problems


@pytest.mark.parametrize("n", [1, 2, 5, 9, 10, 50])
def test_grid_is_valid_and_scores_as_grid(n):
    params = grid(n)
    assert is_valid(n, params)
    assert puzzle_score(n, params) == pytest.approx(grid_score(n))


def test_perfect_square_scores_zero():
    assert puzzle_score(16, grid(16)) == pytest.approx(0)


def test_rotated_square_side():
    assert container_side(np.array([[0, 0, 45.0]])) == pytest.approx(math.sqrt(2))


def test_overlap_detected():
    params = np.array([[0.5, 0.5, 0], [1.2, 0.5, 0]])
    assert not is_valid(2, params)
    assert "overlapping" in problems(2, params)[0]


def test_touching_allowed_including_rotated():
    # Two 30-degree squares sharing a full edge.
    t = math.radians(30)
    params = np.array([[0, 0, 30], [math.cos(t), math.sin(t), 30]])
    assert is_valid(2, params)


def test_wrong_count_invalid():
    assert not is_valid(3, grid(2))


def test_invalid_scored_as_grid():
    bad = {2: np.array([[0.5, 0.5, 0], [0.6, 0.5, 0]])}
    assert total(bad, n_max=2) == pytest.approx(grid_score(1) + grid_score(2))


def test_submission_roundtrip(tmp_path):
    sols = {3: np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [0.1234567890123, 1.5, 12.5]])}
    path = tmp_path / "sub.csv"
    submission.write(sols, path, n_max=4)
    back = submission.read(path)
    assert sorted(back) == [1, 2, 3, 4]
    np.testing.assert_array_equal(back[3], sols[3])
    np.testing.assert_array_equal(back[4], grid(4))


def test_store_keeps_only_valid_improvements(tmp_path):
    store = SolutionStore(tmp_path)
    assert not store.offer(2, grid(2))  # equal to grid fallback: not better
    assert not store.offer(2, np.array([[0.5, 0.5, 0], [0.6, 0.5, 0]]))  # overlapping
    # A worse-but-valid packing is rejected; then nothing is stored.
    assert not store.offer(2, np.array([[0.5, 0.5, 0], [2.5, 0.5, 0]]))
    assert store.get(2) is None

    # n=5: grid side 3; the classic 2+sqrt(2)/2 packing (side ~2.707) is better.
    h = math.sqrt(2) / 2
    s = 2 + h
    better = np.array([
        [0.5, 0.5, 0], [s - 0.5, 0.5, 0], [0.5, s - 0.5, 0], [s - 0.5, s - 0.5, 0],
        [s / 2, s / 2, 45],
    ])
    assert store.offer(5, better)
    np.testing.assert_allclose(store.get(5), better)
    assert store.score(5) == pytest.approx(s / math.sqrt(5) - 1)
