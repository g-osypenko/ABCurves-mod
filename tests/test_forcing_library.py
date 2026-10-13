"""Pure-numpy checks for the baked ProDMP forcing library (no renderer needed)."""
from __future__ import annotations

import numpy as np
import pytest

from abcurves.forcing_library import ForcingLibrary


def _lib(n_basis=4, k=5, seed=0) -> ForcingLibrary:
    rng = np.random.default_rng(seed)
    return ForcingLibrary(rng.normal(size=(k, n_basis, 2)) * 1000.0, n_basis)


def test_orient_round_trips_to_canonical() -> None:
    lib = _lib()
    shape = lib.pick(np.random.default_rng(1))
    for goal in ([120.0, -40.0], [-30.0, 90.0], [10.0, 0.0]):
        g = np.asarray(goal, dtype=np.float64)
        oriented = lib.orient(shape, g)
        dist = float(np.hypot(*g))
        c, s = g[0] / dist, g[1] / dist
        rot = np.array([[c, -s], [s, c]])
        recovered = (oriented / dist) @ rot  # inverse of orient
        assert np.allclose(recovered, shape, atol=1e-9)


def test_orient_scales_with_distance_and_zero_goal() -> None:
    lib = _lib()
    shape = lib.pick(np.random.default_rng(2))
    a = lib.orient(shape, [100.0, 0.0])
    b = lib.orient(shape, [200.0, 0.0])
    assert np.allclose(b, 2.0 * a, atol=1e-9)  # linear in goal magnitude
    assert np.allclose(lib.orient(shape, [0.0, 0.0]), 0.0)  # degenerate goal


def test_rejects_malformed_shapes() -> None:
    with pytest.raises(ValueError):
        ForcingLibrary(np.zeros((3, 5, 2)), n_basis=4)  # n_basis mismatch
    with pytest.raises(ValueError):
        ForcingLibrary(np.zeros((0, 4, 2)), n_basis=4)  # empty
