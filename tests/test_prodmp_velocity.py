"""Analytic C^1 velocity (``ProDMP.velocity_at``) for receding-horizon splicing.

``velocity_at`` must be the exact time-derivative of ``evaluate`` so a replanned
segment can inherit the previous plan's boundary velocity analytically instead of
from a noisy external finite difference.
"""
from __future__ import annotations

import numpy as np
import pytest

from abcurves.prodmp import ProDMP, ProDMPConfig


@pytest.fixture(scope="module")
def dmp() -> ProDMP:
    return ProDMP(ProDMPConfig())


def test_velocity_matches_finite_difference(dmp: ProDMP) -> None:
    """Analytic velocity equals a central finite difference of ``evaluate``."""
    rng = np.random.default_rng(0)
    nw = dmp.n_weights
    max_err = 0.0
    for _ in range(200):
        tau = float(rng.uniform(50, 1000))
        t_b = float(rng.choice([0.0, rng.uniform(0.0, tau * 0.3)]))
        w = rng.normal(size=(nw, 2)) * rng.uniform(0.1, 5)
        ydot_b = rng.normal(size=2) * rng.uniform(0, 3)
        y_b = rng.normal(size=2) * rng.uniform(0, 10)
        t = np.sort(rng.uniform(max(t_b, 1.0), tau, size=5))
        h = 1e-4 * tau

        v = dmp.velocity_at(t, w, tau=tau, ydot_b=ydot_b, y_b=y_b, t_b=t_b)
        yp = dmp.evaluate(t + h, w, tau=tau, ydot_b=ydot_b, y_b=y_b, t_b=t_b)
        ym = dmp.evaluate(t - h, w, tau=tau, ydot_b=ydot_b, y_b=y_b, t_b=t_b)
        fd = (yp - ym) / (2 * h)
        max_err = max(max_err, float(np.max(np.abs(v - fd))))

    assert max_err < 1e-3, f"analytic velocity drifts from finite diff: {max_err:.3e}"


def test_velocity_at_boundary_equals_ydot_b(dmp: ProDMP) -> None:
    """At the seam (t -> t_b) the analytic velocity is exactly the boundary velocity."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=(dmp.n_weights, 2))
    ydot_b = np.array([1.7, -0.9])
    v0 = dmp.velocity_at(
        np.array([1e-6]), w, tau=300.0, ydot_b=ydot_b, y_b=np.zeros(2), t_b=0.0
    )
    assert np.allclose(v0[0], ydot_b, atol=1e-4)


def test_velocity_at_matches_velocity_basis(dmp: ProDMP) -> None:
    """With a zero boundary and t_b=0, velocity_at reduces to velocity_basis @ w."""
    rng = np.random.default_rng(2)
    d = 120
    w = rng.normal(size=(dmp.n_weights, 2))
    vb = dmp.velocity_basis(d) @ w
    t = np.arange(1, d + 1, dtype=np.float64)
    va = dmp.velocity_at(t, w, tau=float(d), ydot_b=np.zeros(2), y_b=np.zeros(2), t_b=0.0)
    assert np.allclose(vb, va, atol=1e-12)
