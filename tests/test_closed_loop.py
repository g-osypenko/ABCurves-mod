"""End-to-end checks for the closed-loop Receding-Horizon ProDMP controller.

These exercise the real persistent GRU renderer, so they skip where the native
Renderer library is unavailable (e.g. a non-Windows checkout without a built
``runtime/c`` library).
"""
from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from abcurves.closed_loop import ClosedLoopConfig, ClosedLoopProDMPController
from abcurves.forcing_library import ForcingLibrary
from abcurves.portable_renderer import RendererRuntimeError

_LIB_PATH = Path(__file__).resolve().parents[1] / "models" / "continuous" / "forcing_lib.npz"


def _lateral(path, target):
    target = np.asarray(target, dtype=np.float64)
    length = np.hypot(*target)
    direction = target / length
    normal = np.array([-direction[1], direction[0]])
    return float(np.max(np.abs(path @ normal)))


@pytest.fixture(scope="module")
def controller() -> ClosedLoopProDMPController:
    try:
        return ClosedLoopProDMPController(config=ClosedLoopConfig(event_seed=7))
    except RendererRuntimeError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"native Renderer unavailable: {exc}")


def _track(ctl: ClosedLoopProDMPController, target, ms=800, update_every=15):
    cursor = np.zeros(2)
    ctl.reset()
    reports = []
    path = []
    t = 0
    while t < ms:
        offset = np.asarray(target, dtype=np.float64) - cursor
        ctl.on_target_update((offset[0], offset[1]), update_every)
        for _ in range(update_every):
            r = ctl.step().astype(np.float64)
            cursor += r
            reports.append(r)
            path.append(cursor.copy())
            t += 1
            if t >= ms:
                break
    return cursor, np.asarray(reports), np.asarray(path)


def test_converges_to_moving_target(controller: ClosedLoopProDMPController) -> None:
    target = np.array([120.0, -40.0])
    cursor, reports, _ = _track(controller, target, ms=900)
    residual = float(np.hypot(*(target - cursor)))
    assert residual < 5.0, f"did not converge: residual {residual:.2f} counts"
    assert np.all(np.isfinite(reports))


def test_output_is_smooth_no_jerk_spikes(controller: ClosedLoopProDMPController) -> None:
    _, reports, _ = _track(controller, np.array([200.0, 80.0]), ms=900)
    # No single-millisecond report should be a wild ballistic spike.
    speed = np.hypot(reports[:, 0], reports[:, 1])
    assert np.max(speed) < 25.0, f"per-ms spike too large: {np.max(speed):.1f}"
    # Second difference (jerk proxy) must stay bounded -- no infinite-jerk seams.
    jerk = np.diff(reports, n=2, axis=0)
    assert np.max(np.hypot(jerk[:, 0], jerk[:, 1])) < 40.0


def test_idle_step_keeps_running_when_buffer_empty(
    controller: ClosedLoopProDMPController,
) -> None:
    controller.reset()
    committed = controller.on_target_update((50.0, 0.0), 15)
    # Drain the committed slice, then keep stepping with an empty buffer.
    for _ in range(committed):
        controller.step()
    assert controller.pending_ticks == 0
    # Idle steps must still produce finite reports (renderer coasts + tremor).
    idle = np.asarray([controller.step() for _ in range(50)], dtype=np.float64)
    assert np.all(np.isfinite(idle))


def test_b2prime_library_adds_curvature_and_still_converges() -> None:
    """With the baked forcing library, motion curves more yet still converges."""
    if not _LIB_PATH.is_file():
        pytest.skip("baked forcing_lib.npz not present")
    try:
        plain = ClosedLoopProDMPController(config=ClosedLoopConfig(event_seed=3))
    except RendererRuntimeError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"native Renderer unavailable: {exc}")
    lib = ForcingLibrary.load(_LIB_PATH)
    target = np.array([160.0, -50.0])

    # A single sticky shape may happen to be nearly straight; the macro-curvature
    # shows up as a population effect, so average over several sticky draws.
    plain_lat, curved_lat = [], []
    for seed in range(12):
        p = ClosedLoopProDMPController(config=ClosedLoopConfig(event_seed=seed))
        c = ClosedLoopProDMPController(
            config=ClosedLoopConfig(event_seed=seed, forcing_gain=0.45), forcing_library=lib
        )
        cur_p, _, path_p = _track(p, target, ms=1100)
        cur_c, rep_c, path_c = _track(c, target, ms=1100)
        assert float(np.hypot(*(target - cur_c))) < 8.0  # still converges
        assert np.max(np.hypot(rep_c[:, 0], rep_c[:, 1])) < 25.0  # still smooth
        plain_lat.append(_lateral(path_p, target))
        curved_lat.append(_lateral(path_c, target))

    # Meaningfully more curved on average (relative margin is robust to which
    # sticky shapes happen to be sampled across seeds).
    assert np.mean(curved_lat) > 1.3 * np.mean(plain_lat)


def test_forcing_gain_is_safely_capped() -> None:
    """An over-strong forcing request must be clamped, not diverge (P2 safety)."""
    if not _LIB_PATH.is_file():
        pytest.skip("baked forcing_lib.npz not present")
    lib = ForcingLibrary.load(_LIB_PATH)
    target = np.array([120.0, -40.0])
    for seed in range(4):
        try:
            ctl = ClosedLoopProDMPController(
                config=ClosedLoopConfig(event_seed=seed, forcing_gain=3.0),  # absurd request
                forcing_library=lib,
            )
        except RendererRuntimeError as exc:  # pragma: no cover
            pytest.skip(f"native Renderer unavailable: {exc}")
        cur, _, _ = _track(ctl, target, ms=1200)
        assert float(np.hypot(*(target - cur))) < 8.0  # ceiling kept it convergent


def test_pursuit_backoff_reduces_gain_on_fast_pursuit() -> None:
    """Adaptive gain: full curvature on a reach, backed off during fast pursuit (P2)."""
    if not _LIB_PATH.is_file():
        pytest.skip("baked forcing_lib.npz not present")
    lib = ForcingLibrary.load(_LIB_PATH)
    tau = 250
    z = np.zeros(2)
    try:
        on = ClosedLoopProDMPController(
            config=ClosedLoopConfig(forcing_gain=0.45, forcing_adaptive=True), forcing_library=lib
        )
        off = ClosedLoopProDMPController(
            config=ClosedLoopConfig(forcing_gain=0.45, forcing_adaptive=False), forcing_library=lib
        )
    except RendererRuntimeError as exc:  # pragma: no cover
        pytest.skip(f"native Renderer unavailable: {exc}")
    # adaptive on: far goal keeps full curvature, small error backs it off
    reach = on._effective_gain(z, np.array([200.0, 0.0]), tau)
    pursuit = on._effective_gain(z, np.array([15.0, 0.0]), tau)
    assert abs(reach - 0.45) < 1e-9
    assert pursuit < 0.5 * reach
    # adaptive off (default): gain is distance-independent (texture preserved)
    assert abs(off._effective_gain(z, np.array([200.0, 0.0]), tau)
               - off._effective_gain(z, np.array([15.0, 0.0]), tau)) < 1e-9


def test_deadzone_scale_is_monotone_and_damps_inside(
    controller: ClosedLoopProDMPController,
) -> None:
    cfg = controller.config
    inside = controller._deadzone_scale(0.0)
    edge = controller._deadzone_scale(cfg.r_inner)
    outside = controller._deadzone_scale(cfg.r_inner + 20.0)
    assert inside < edge < outside  # smooth, monotone (no wall slam)
    assert inside < 0.5 <= outside
    assert outside > 0.99  # far outside, error passes through essentially intact


def test_motor_noise_is_signal_dependent_and_direction_preserving(
    controller: ClosedLoopProDMPController,
) -> None:
    cfg = controller.config
    assert cfg.motor_noise_rel > 0.0  # on by default (P3 texture)
    # zero command -> zero noise: no idle jitter from the motor-noise term
    assert controller._add_motor_noise((0.0, 0.0)) == (0.0, 0.0)
    # pure-x command -> noise stays on x (no per-axis sign flicker on y)
    rng_state = controller._rng.bit_generator.state
    out = np.array([controller._add_motor_noise((1.5, 0.0)) for _ in range(500)])
    controller._rng.bit_generator.state = rng_state
    assert np.all(out[:, 1] == 0.0)
    assert np.all(out[:, 0] > 0.0)  # never reverses the commanded direction
    rel_std = float(np.std(out[:, 0] / 1.5))
    assert 0.5 * cfg.motor_noise_rel < rel_std < 1.5 * cfg.motor_noise_rel


def test_initial_velocity_handoff_is_c1(controller: ClosedLoopProDMPController) -> None:
    v0 = np.array([2.0, -1.0])
    controller.reset(initial_velocity=v0)
    controller.on_target_update((150.0, -60.0), 8)
    # the first plan starts from the handed-off velocity, not from rest
    _, vel = controller.get_analytical_state_at(0.0)
    np.testing.assert_allclose(vel, v0, atol=1e-9)
    controller.reset()
    np.testing.assert_allclose(controller.last_analytical_vel, 0.0)


def test_fitts_horizon_shrinks_with_error() -> None:
    try:
        ctl = ClosedLoopProDMPController(
            config=ClosedLoopConfig(horizon_fitts_a=100.0, horizon_fitts_b=60.0)
        )
    except RendererRuntimeError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"native Renderer unavailable: {exc}")
    taus = [ctl._horizon(r) for r in (0.0, 16.0, 64.0, 256.0)]
    assert taus[0] == pytest.approx(100.0)
    assert taus[1] == pytest.approx(160.0)
    assert all(a < b for a, b in zip(taus, taus[1:]))
    # off by default -> constant horizon
    assert ClosedLoopConfig().horizon_fitts_b == 0.0


def _ff_controller(**cfg) -> ClosedLoopProDMPController:
    try:
        return ClosedLoopProDMPController(config=ClosedLoopConfig(**cfg))
    except RendererRuntimeError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"native Renderer unavailable: {exc}")


def test_alpha_beta_tracks_constant_target_velocity() -> None:
    ctl = _ff_controller(target_ff="alpha_beta")
    v_true = np.array([1.5, -0.5])  # counts/ms
    vel = np.zeros(2)
    for k in range(80):  # cursor stays at 0 -> offset == absolute target
        vel = ctl._update_target_velocity(v_true * 8.0 * k, 8.0)
    np.testing.assert_allclose(vel, v_true, atol=0.02)


def test_target_jump_is_gated_not_a_phantom_velocity() -> None:
    for mode in ("alpha_beta", "raw"):
        ctl = _ff_controller(target_ff=mode)
        for _ in range(10):
            ctl._update_target_velocity(np.array([0.0, 0.0]), 8.0)
        vel = ctl._update_target_velocity(np.array([220.0, -90.0]), 8.0)  # a step jump
        np.testing.assert_allclose(vel, 0.0)


def test_static_target_gets_no_feedforward_lead() -> None:
    ctl = _ff_controller()
    assert ctl.config.target_ff == "alpha_beta"  # on by default (P4)
    for _ in range(20):
        vel = ctl._update_target_velocity(np.array([150.0, 40.0]), 8.0)
    np.testing.assert_allclose(vel, 0.0, atol=1e-9)


def test_commit_covers_the_update_interval() -> None:
    ctl = _ff_controller(commit_ms=12)
    assert ctl.on_target_update((150.0, 0.0), 8) == 12
    # slower updates must not leave the renderer idling until the next replan
    assert ctl.on_target_update((150.0, 0.0), 16) >= 16
    assert ctl.on_target_update((150.0, 0.0), 40) >= 40
