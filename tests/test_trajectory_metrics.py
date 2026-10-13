"""Checks for the objective trajectory metrics (pure numpy/scipy, no renderer)."""
from __future__ import annotations

import numpy as np
import pytest

from abcurves import trajectory_metrics as tm


def _two_thirds_ellipse(a=320.0, b=170.0, beta=1.0 / 3.0, fs=1000.0, dur_s=3.0):
    """Sample an ellipse traversed so that speed = C * kappa^(-beta).

    By construction this obeys the two-thirds power law with the given beta, so a
    correct metric must recover it.
    """
    th = np.linspace(0.0, 2 * np.pi, 20000)
    x, y = a * np.cos(th), b * np.sin(th)
    dx, dy = -a * np.sin(th), b * np.cos(th)
    ddx, ddy = -a * np.cos(th), -b * np.sin(th)
    sp_th = np.hypot(dx, dy)  # |dp/dtheta|
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(sp_th**3, 1e-12)
    # dt/dtheta so that v = |dp/dtheta|*dtheta/dt = C*kappa^-beta
    dt_dth = sp_th * kappa**beta
    t = np.concatenate([[0.0], np.cumsum(0.5 * (dt_dth[1:] + dt_dth[:-1]) * np.diff(th))])
    t = t / t[-1] * dur_s  # normalize to the requested duration
    grid = np.arange(0.0, dur_s, 1.0 / fs)
    th_t = np.interp(grid, t, th)
    return np.stack([np.interp(th_t, th, x), np.interp(th_t, th, y)], axis=1)


def test_recovers_two_thirds_exponent() -> None:
    path = _two_thirds_ellipse(beta=1.0 / 3.0)
    beta, r2 = tm.two_thirds_power_law(path)
    assert abs(beta - 1.0 / 3.0) < 0.07, f"beta={beta}"
    assert r2 > 0.9, f"r2={r2}"


def test_distinguishes_a_non_human_exponent() -> None:
    # A curve built with beta=0 (constant speed) must NOT look like beta~1/3.
    path = _two_thirds_ellipse(beta=0.0)
    beta, _ = tm.two_thirds_power_law(path)
    assert beta < 0.15, f"beta={beta}"


def test_straight_constant_velocity_is_smooth_and_untremored() -> None:
    t = np.arange(0, 1500)
    path = np.stack([0.5 * t, np.zeros_like(t)], axis=1).astype(float)  # 500 counts/s
    assert tm.tremor_band_fraction(path) < 0.02  # no oscillation in a straight glide
    assert np.isnan(tm.two_thirds_power_law(path)[0])  # no curvature to fit
    assert np.isfinite(tm.log_dimensionless_jerk(path))  # defined (absolute value is convention-bound)
    assert np.isfinite(tm.sparc(path))


def test_evaluate_report_shape_and_target_errors() -> None:
    path = _two_thirds_ellipse()
    rep = tm.evaluate(path)
    assert rep.samples == len(path)
    assert np.isnan(rep.rms_error) and np.isnan(rep.final_error)  # no target given
    target = path + np.array([5.0, -3.0])  # constant 5.83-count offset
    rep2 = tm.evaluate(path, target)
    assert abs(rep2.rms_error - np.hypot(5.0, 3.0)) < 1e-6
    assert rep2.peak_speed > 0


def test_guards_on_tiny_input() -> None:
    tiny = np.zeros((4, 2))
    assert np.isnan(tm.two_thirds_power_law(tiny)[0])
    assert np.isnan(tm.log_dimensionless_jerk(tiny))
    assert np.isnan(tm.sparc(tiny))
    assert np.isnan(tm.tremor_band_fraction(tiny))
