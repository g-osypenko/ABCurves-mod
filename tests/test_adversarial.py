"""Adversarial proof that the old ContinuousPlanner symptoms are gone.

Each test drives the closed-loop controller through an input that specifically
triggered one of the documented pathologies, and asserts the failure mode is
absent with quantitative bounds (thresholds chosen with margin from measured
healthy behaviour):

* #3 finite-difference noise amplification  -> target observation jitter
* #1 31 Hz head-flutter / ringing           -> sharp target reversal
* #2 Poisson brake freeze + ballistic saccade -> pause in deadzone then far jump
* zero-start singularity                     -> first move from v = 0

These run the real GRU renderer, so they skip where it is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from abcurves.closed_loop import ClosedLoopConfig, ClosedLoopProDMPController
from abcurves.portable_renderer import RendererRuntimeError

try:
    from scipy.ndimage import gaussian_filter1d as _gf
except Exception:  # pragma: no cover
    _gf = None


def _make(gain: float = 0.0, seed: int = 7) -> ClosedLoopProDMPController:
    try:
        return ClosedLoopProDMPController(config=ClosedLoopConfig(event_seed=seed, forcing_gain=gain))
    except RendererRuntimeError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"native Renderer unavailable: {exc}")


def _drive(ctl, target_fn, ms, update_ms=8):
    cur = np.zeros(2)
    ctl.reset()
    rep, path = [], []
    t = 0
    while t < ms:
        tgt = np.asarray(target_fn(t), dtype=np.float64)
        ctl.on_target_update((tgt[0] - cur[0], tgt[1] - cur[1]), update_ms)
        for _ in range(update_ms):
            if t >= ms:
                break
            r = ctl.step().astype(np.float64)
            cur += r
            rep.append(r)
            path.append(cur.copy())
            t += 1
    return np.asarray(rep), np.asarray(path)


def _speed(rep):  # counts/s
    return np.hypot(rep[:, 0], rep[:, 1]) * 1000.0


def _smooth(x, sigma=8.0):
    return _gf(x, sigma) if _gf is not None else x


def test_symptom3_target_jitter_is_not_amplified():
    """Analytic C^1 boundary velocity must not turn target jitter into speed spikes."""
    clean = _speed(_drive(_make(), lambda t: (150.0, 0.0), 900)[0])
    rng = np.random.default_rng(0)
    jit = _speed(_drive(_make(), lambda t: (150.0 + rng.normal(0, 1.5),
                                            0.0 + rng.normal(0, 1.5)), 900)[0])
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    jit_rms = float(np.sqrt(np.mean(jit**2)))
    # A naive dx/dt on sigma=1.5 @ 8 ms would inject ~188 counts/s/axis of phantom speed.
    assert jit.max() <= 1.3 * clean.max()
    assert jit_rms <= 1.3 * clean_rms
    assert (jit_rms - clean_rms) < 100.0  # nowhere near the finite-difference phantom


def test_symptom1_sharp_reversal_has_no_ringing_or_overshoot():
    """A hard +X -> -X reversal must settle cleanly, not flutter/ring."""
    _, path = _drive(_make(), lambda t: (120.0, 0.0) if t < 400 else (-120.0, 0.0), 1400)
    x = path[:, 0]
    overshoot = max(0.0, -120.0 - x.min())
    vx = np.gradient(_smooth(x))
    settle = vx[460:]
    sign_reversals = int(np.sum(settle[1:] * settle[:-1] < 0))
    assert overshoot < 20.0, f"overshoot {overshoot:.1f}"
    assert sign_reversals <= 3, f"ringing reversals {sign_reversals}"


def test_symptom1_reversal_stays_clean_with_b2prime():
    """The B2' forcing must not reintroduce reversal ringing (looser bounds)."""
    lib_ctl = _make(gain=0.45)
    if lib_ctl._forcing_library is None:
        # no library present; B2' reduces to B1, already covered
        return
    _, path = _drive(lib_ctl, lambda t: (120.0, 0.0) if t < 400 else (-120.0, 0.0), 1400)
    x = path[:, 0]
    overshoot = max(0.0, -120.0 - x.min())
    vx = np.gradient(_smooth(x))
    sign_reversals = int(np.sum(vx[460:][1:] * vx[460:][:-1] < 0))
    assert overshoot < 45.0
    assert sign_reversals <= 8


def test_symptom2_pause_then_reacquire_no_freeze_no_saccade():
    """Settle in the deadzone, then jump far: react promptly, no ballistic overshoot."""
    rep, path = _drive(_make(), lambda t: (0.0, 0.0) if t < 500 else (220.0, -90.0), 1600)
    after = np.hypot(rep[500:, 0], rep[500:, 1])
    moved = int(np.argmax(after > 0.8)) if np.any(after > 0.8) else 10_000
    peak = _speed(rep)[500:].max()
    tgt = np.array([220.0, -90.0])
    along = path @ (tgt / np.hypot(*tgt))
    overshoot = max(0.0, float(along.max()) - float(np.hypot(*tgt)))
    assert moved < 30, f"re-acquire latency {moved} ms (old FSM froze 100-200 ms)"
    assert peak < 9000.0, f"ballistic saccade peak {peak:.0f} counts/s"
    assert overshoot < 30.0, f"overshoot {overshoot:.1f}"


def test_zero_start_has_no_jerk_spike():
    """Starting from rest (v=0) must be smooth, not a startup jerk singularity."""
    _, path = _drive(_make(), lambda t: (150.0, 50.0), 700)
    sp = _smooth(path, 6.0) if _gf is not None else path
    vel = np.gradient(sp, axis=0)
    jerk = np.gradient(np.gradient(vel, axis=0), axis=0)
    jm = np.hypot(jerk[:, 0], jerk[:, 1])
    active_peak = float(jm[:300].max())  # the reach itself; the long tail is ~idle
    assert np.all(np.isfinite(jm))
    # A v=0 singularity would spike at the very opening and dominate the run.
    assert jm[:5].max() < 0.8 * active_peak   # opening is gentler than the move's peak
    assert jm[:30].max() < jm.max()           # the worst jerk is not at the start
