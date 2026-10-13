"""Closed-loop Receding-Horizon ProDMP controller for dynamic target tracking.

This is the trajectory engine described in ``rules.md`` / ``REFACTORING_DOCS``.
It replaces the unstable ``_continuous`` FSM with a transparent Receding Horizon
Control (RHC) loop:

* a macro-plan is a ProDMP curve homing to the (damped) target error with
  ``goal_mode='target'`` -- mathematically smooth, minimum-jerk, convergent;
* the boundary velocity of every replanned segment is the **analytic** derivative
  of the previous plan (``ProDMP.velocity_at``), giving strict :math:`C^1`
  continuity at the seam instead of a noisy external ``dx/dt``;
* the committed macro deltas are streamed into a persistent 1 kHz GRU renderer
  (``PortableRendererStream``) whose hidden state, fractional accumulator and W5
  state survive every replan, hold and restart -- that is what carries the
  8-12 Hz physiological tremor and log-normal micro-texture.

Forcing-weight source (the macro acceleration shape) is decision **B2'**
(see ``REFACTORING_DOCS/DESIGN_NOTE.md``): an offline-baked library sampled here.
Until that library is wired in, ``forcing_library=None`` gives **B1** -- pure
boundary+goal homing, which is the robust fallback.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .model_store import default_model_dir
from .portable_renderer import PortableRendererModel
from .prodmp import ProDMP, ProDMPConfig

CONTEXT_TICKS = 256


@dataclass
class ClosedLoopConfig:
    """Tunables for :class:`ClosedLoopProDMPController`.

    Lengths are in milliseconds (one 1 kHz tick each); offsets/radii are in the
    renderer's canonical x-right/y-up native mouse counts.
    """

    horizon_ms: int = 320
    """Full ProDMP homing horizon ``tau``; only ``commit_ms`` of it is applied.

    320 (was 250): the longer horizon removes the reach "crawl" tail (Texture19
    0.35 -> 0.28); the pursuit lag it would cost is cancelled by ``target_ff``."""
    commit_ms: int = 12
    """Minimum slice of each plan committed before the next replan (receding).
    The commit is always extended to cover the update interval (``dt + 2``), so
    slower target updates never leave the renderer idling between replans."""
    min_horizon_ms: int = 60
    horizon_fitts_b: float = 0.0
    """Time-to-go horizon (Hoff & Arbib): when > 0, each replan uses
    ``tau = horizon_fitts_a + horizon_fitts_b * log2(1 + r / horizon_fitts_w)``
    (Fitts-law remaining movement time from the current error ``r``) instead of
    the constant ``horizon_ms``.  The horizon shrinks as the error shrinks, so a
    reach arrives in finite time instead of crawling in on an exponential tail;
    it is stateless, so a moving or jumping target needs no special handling."""
    horizon_fitts_a: float = 100.0
    horizon_fitts_w: float = 16.0
    r_inner: float = 2.0
    """Inner deadzone radius: inside it the error is damped toward zero."""
    deadzone_lambda: float = 1.0
    """Logistic steepness of the deadzone edge (prevents 'wall slamming')."""
    deadzone_eps: float = 0.0
    rest_noise_ms: int = 12
    """Synthetic resting micro-tremor primed on reset (avoids a hard v=0 start)."""
    rest_noise_sigma: float = 0.1
    forcing_gain: float = 0.45
    """Scales the baked macro curvature (B2'); 0 -> pure B1 homing.

    Only used when a ``forcing_library`` is supplied.  ~0.4-0.5 adds visible,
    human-like curvature while keeping exact convergence; the safety cap below
    makes any value safe (an over-strong request is clamped, not catastrophic)."""
    forcing_gain_max: float = 0.6
    """Safety ceiling on the *effective* forcing gain.  The forcing feeds its
    velocity forward into the next plan's boundary condition, so too large a gain
    is a positive-feedback instability (divergence); clamping the effective gain
    to a measured-stable ceiling makes any requested ``forcing_gain`` safe (P2)."""
    forcing_adaptive: bool = False
    """Enable the distance-based curvature back-off below.  Default off: measured
    on reaches it trades away texture human-likeness (which wants curvature through
    the whole move) for slightly tighter continuous-pursuit tracking (not a
    pathology).  The lever stays available for P4 tuning / specific pursuit use."""
    forcing_reach_lo: float = 30.0
    """Adaptive gain (distance-based): below this goal distance (counts) the
    curvature is at its floor; this is the small-persistent-error regime of fast
    pursuit and of the final approach, where tracking accuracy beats bend."""
    forcing_reach_hi: float = 90.0
    """At/above this goal distance the forcing is full -- the ballistic phase of a
    reach, where human movement carries its curvature."""
    forcing_pursuit_floor: float = 0.2
    """Minimum fraction of forcing kept in the small-error regime (never zero, so
    motion keeps some human character even when settling)."""
    motor_noise_rel: float = 0.25
    """Signal-dependent motor noise (Harris & Wolpert): multiplicative noise on
    the commanded macro speed *along* the movement direction (std = this fraction
    of the speed).  Zero at rest -> no idle jitter; direction-preserving -> no
    per-axis sign flicker; zero-mean drift is absorbed by the next replan."""
    motor_noise_perp: float = 0.0
    """Perpendicular motor-noise std as a fraction of the speed (small: humans
    are noisier along the movement than across it)."""
    motor_noise_tau_ms: float = 0.0
    """Correlation time of the motor noise (first-order, OU); 0 -> white."""
    target_ff: str = "alpha_beta"
    """Target-velocity feedforward (predictive pursuit): ``"off"``, ``"raw"``
    (plain ``dx/dt`` of the target -- amplifies observation jitter, pathology #3)
    or ``"alpha_beta"`` (fixed-gain steady-state Kalman tracker, ~10 flops per
    update).  The target's absolute position is the controller's own exactly-known
    cursor (sum of its reports) plus the observed error; the plan then homes to
    where the target *will be* ``ff_lead_ms`` ahead, which removes the steady
    pursuit lag of pure homing.  Measured (P4): mean pursuit RMS 37.5 -> ~14 at
    unchanged jitter robustness; ``"raw"`` is ~10% tighter but fails the
    target-jitter gate, so it is opt-in only."""
    ff_lead_frac: float = 0.3
    """Prediction lead as a fraction of the homing horizon ``tau`` (the receding
    homing lag scales with ``tau``; 0.3 is the measured optimum at 8 and 16 ms
    update cadence)."""
    ff_alpha: float = 0.5
    ff_beta: float = 0.2
    ff_gate: float = 40.0
    """Innovation gate (counts): a larger surprise is a target *jump*, not motion,
    so the estimator re-initialises (velocity 0) instead of inventing a huge
    phantom velocity that would overshoot the new target."""
    ff_vmax: float = 10.0
    """Clamp on the estimated target speed (counts/ms)."""
    event_seed: int = 7
    prodmp: ProDMPConfig = field(default_factory=ProDMPConfig)


@dataclass
class _Plan:
    """The current macro-plan, stored so its analytic state can be queried."""

    w_g: np.ndarray  # (n_weights, 2) with the goal row already set
    ydot_b: np.ndarray  # (2,) boundary velocity, counts/ms
    tau: float  # horizon length, ms
    y_b: np.ndarray  # (2,) boundary position (0 in the plan-relative frame)


class ClosedLoopProDMPController:
    """Receding-horizon trajectory generator with a persistent 1 kHz renderer.

    Typical use::

        ctl = ClosedLoopProDMPController()
        ctl.on_target_update((dx, dy), dt_ms)   # whenever the target moves
        report = ctl.step()                     # once per millisecond

    ``on_target_update`` takes the *current* offset from the cursor to the target
    (closed-loop error) and the time since the previous update; ``step`` returns
    one integer ``(dx, dy)`` hardware report.
    """

    def __init__(
        self,
        *,
        config: Optional[ClosedLoopConfig] = None,
        renderer_context: Optional[np.ndarray] = None,
        model_dir: Optional[str | Path] = None,
        renderer_model: Optional[PortableRendererModel] = None,
        forcing_library=None,
    ) -> None:
        self.config = config or ClosedLoopConfig()
        self.prodmp = ProDMP(self.config.prodmp)
        root = default_model_dir() if model_dir is None else Path(model_dir)
        self.renderer_model = renderer_model or PortableRendererModel(
            root / "renderer_global_h80.bin"
        )
        if renderer_context is None:
            # A quiet device before the first movement (cold start).
            renderer_context = np.zeros((CONTEXT_TICKS, 2), dtype=np.int16)
        self._context = self.renderer_model.prepare_context(renderer_context)
        self._forcing_library = forcing_library  # B2' hook; None -> B1 zero forcing
        self._rng = np.random.default_rng(self.config.event_seed)

        self._stream = None
        self._buffer: deque = deque()
        self._plan: Optional[_Plan] = None
        self._time_since_plan = 0
        self.last_analytical_vel = np.zeros(2)
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, *, event_seed: Optional[int] = None, initial_velocity=None) -> None:
        """Begin a fresh renderer stream and clear all plan state.

        Primes a few ms of resting micro-tremor so that the first movement does
        not start from a mathematically hard standstill (``v = 0``).
        ``initial_velocity`` (counts/ms) hands off a movement already in progress:
        it becomes the first plan's boundary velocity, so taking over an in-flight
        cursor is :math:`C^1` instead of an abrupt stop-and-restart.
        """
        seed = self.config.event_seed if event_seed is None else int(event_seed)
        self._stream = self._context.begin_stream(event_seed=int(seed))
        self._buffer = deque()
        self._plan = None
        self._time_since_plan = 0
        self._shape = None
        self._noise_state = np.zeros(2)
        self._cursor = np.zeros(2)  # sum of emitted reports (exact, integer)
        self._tgt_pos: Optional[np.ndarray] = None
        self._tgt_vel = np.zeros(2)
        self.last_analytical_vel = (
            np.zeros(2)
            if initial_velocity is None
            else np.asarray(initial_velocity, dtype=np.float64).reshape(2).copy()
        )
        self._prime_rest_noise()

    def _prime_rest_noise(self) -> None:
        cfg = self.config
        if cfg.rest_noise_ms <= 0:
            return
        noise = self._rng.normal(scale=cfg.rest_noise_sigma, size=(cfg.rest_noise_ms, 2))
        for dx, dy in noise:
            self._stream.step((float(dx), float(dy)))

    # ------------------------------------------------------------------
    # Analytic state (C^1 splicing support)
    # ------------------------------------------------------------------
    def get_analytical_state_at(self, delta_t: float) -> tuple[np.ndarray, np.ndarray]:
        """Analytic ``(position, velocity)`` of the current plan at ``delta_t`` ms.

        Velocity is the exact ProDMP derivative, not a finite difference -- this
        is the boundary velocity that keeps replanned segments :math:`C^1`.
        """
        if self._plan is None:
            return np.zeros(2), self.last_analytical_vel.copy()
        p = self._plan
        t = np.array([float(np.clip(delta_t, 0.0, p.tau))], dtype=np.float64)
        pos = self.prodmp.evaluate(t, p.w_g, tau=p.tau, ydot_b=p.ydot_b, y_b=p.y_b)[0]
        vel = self.prodmp.velocity_at(t, p.w_g, tau=p.tau, ydot_b=p.ydot_b, y_b=p.y_b)[0]
        return pos, vel

    # ------------------------------------------------------------------
    # Target updates (macro replanning)
    # ------------------------------------------------------------------
    def _deadzone_scale(self, r: float) -> float:
        """Logistic ``omega(r)``: ~0 inside the deadzone, ->1 well outside it."""
        cfg = self.config
        z = cfg.deadzone_lambda * (r - cfg.r_inner - cfg.deadzone_eps)
        # clip keeps exp finite for large |z|
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))

    def _update_target_velocity(self, target: np.ndarray, dt: float) -> np.ndarray:
        """Estimate the target's velocity (counts/ms) from one observation."""
        cfg = self.config
        z = self._cursor + target  # absolute target position
        dt = max(float(dt), 1e-3)
        if self._tgt_pos is None:
            self._tgt_pos = z.copy()
            self._tgt_vel = np.zeros(2)
            return self._tgt_vel
        pred = self._tgt_pos + self._tgt_vel * dt
        innov = z - pred
        if float(np.hypot(innov[0], innov[1])) > cfg.ff_gate:
            # a jump, not motion: re-acquire without a phantom velocity
            self._tgt_pos = z.copy()
            self._tgt_vel = np.zeros(2)
            return self._tgt_vel
        if cfg.target_ff == "raw":
            vel = (z - self._tgt_pos) / dt
            self._tgt_pos = z.copy()
        elif cfg.target_ff == "alpha_beta":
            self._tgt_pos = pred + cfg.ff_alpha * innov
            vel = self._tgt_vel + (cfg.ff_beta / dt) * innov
        else:
            raise ValueError(f"unknown target_ff {cfg.target_ff!r}")
        speed = float(np.hypot(vel[0], vel[1]))
        if speed > cfg.ff_vmax:
            vel = vel * (cfg.ff_vmax / speed)
        self._tgt_vel = vel
        return vel

    def _horizon(self, r: float) -> float:
        """Homing horizon ``tau`` (ms): constant, or Fitts time-to-go of ``r``."""
        cfg = self.config
        if cfg.horizon_fitts_b <= 0.0:
            return float(cfg.horizon_ms)
        return cfg.horizon_fitts_a + cfg.horizon_fitts_b * float(
            np.log2(1.0 + r / max(cfg.horizon_fitts_w, 1e-6))
        )

    def _effective_gain(self, ydot_b: np.ndarray, goal: np.ndarray, tau: int) -> float:
        """Adaptive, safety-ceilinged forcing gain for this update."""
        cfg = self.config
        gain = cfg.forcing_gain
        # (a) optional distance-based back-off: full curvature in the ballistic
        #     phase (goal far), floored for a small persistent error (fast pursuit
        #     / settling).  Off by default -- see the config note.
        if cfg.forcing_adaptive:
            dist = float(np.hypot(goal[0], goal[1]))
            span = max(cfg.forcing_reach_hi - cfg.forcing_reach_lo, 1e-6)
            ramp = (dist - cfg.forcing_reach_lo) / span
            factor = cfg.forcing_pursuit_floor + (1.0 - cfg.forcing_pursuit_floor) * ramp
            gain *= float(np.clip(factor, cfg.forcing_pursuit_floor, 1.0))
        # (b) hard safety ceiling (always on) -> loop stays stable for any gain.
        return min(gain, cfg.forcing_gain_max)

    def _forcing_rows(self, goal: np.ndarray, ydot_b: np.ndarray, tau: int) -> np.ndarray:
        """Forcing rows ``w[:-1]`` for this plan (macro acceleration shape).

        B1 (no library): zeros -> pure minimum-jerk boundary+goal homing.
        B2' (library): a sticky canonical shape, oriented/scaled to ``goal`` and
        scaled by an adaptive, safety-ceilinged gain.  The shape is chosen once per
        movement (sticky) and only re-oriented per replan -- never resampled per
        tick -- so the curvature stays coherent instead of flickering.
        """
        n_basis = self.prodmp.n_basis
        if self._forcing_library is None or self.config.forcing_gain == 0.0:
            return np.zeros((n_basis, 2), dtype=np.float64)
        if self._shape is None:
            self._shape = self._forcing_library.pick(self._rng)
        gain = self._effective_gain(ydot_b, goal, tau)
        return self._forcing_library.orient(self._shape, goal) * gain

    def on_target_update(
        self, target_offset, dt: float, *, apply_deadzone: bool = True
    ) -> int:
        """Replan toward ``target_offset`` (current cursor->target error).

        ``dt`` is milliseconds since the previous update; it advances the time
        used to read the previous plan's analytic boundary velocity.  Returns the
        number of committed macro ticks pushed to the renderer buffer.
        """
        cfg = self.config
        target = np.asarray(target_offset, dtype=np.float64).reshape(2)

        # (1) C^1 boundary velocity: analytic derivative of the previous plan.
        if self._plan is not None:
            _, ydot_b = self.get_analytical_state_at(self._time_since_plan)
        else:
            ydot_b = self.last_analytical_vel.copy()

        tau = int(max(cfg.min_horizon_ms, self._horizon(float(np.hypot(*target)))))

        # (1b) Predictive pursuit: aim where the target will be (feedforward).
        if cfg.target_ff != "off":
            lead = cfg.ff_lead_frac * tau
            target = target + self._update_target_velocity(target, dt) * lead

        # (2) Deadzone damping of the error vector (smooth, no wall slam).
        r = float(np.hypot(target[0], target[1]))
        goal = target.copy()
        if apply_deadzone:
            goal *= self._deadzone_scale(r)

        # (3) Macro-plan: ProDMP homing to goal over a receding horizon.
        w_eff = np.zeros((self.prodmp.n_weights, 2), dtype=np.float64)
        w_eff[:-1] = self._forcing_rows(goal, ydot_b, tau)  # macro curvature (B2') or 0 (B1)
        w_eff[-1] = goal  # goal row -> homes to target (goal_mode='target' equivalent)
        deltas = self.prodmp.generate_deltas(
            w_eff, ydot_b, tau, y_b=np.zeros(2), goal_mode="learned"
        )

        # (4) Commit only the near slice; the next update replans (receding).
        commit = int(min(max(cfg.commit_ms, int(np.ceil(dt)) + 2), len(deltas)))
        self._buffer = deque(deltas[:commit])
        self._plan = _Plan(w_g=w_eff, ydot_b=ydot_b, tau=float(tau), y_b=np.zeros(2))
        self._time_since_plan = 0
        self.last_analytical_vel = ydot_b
        return commit

    def _add_motor_noise(self, delta: tuple[float, float]) -> tuple[float, float]:
        """Signal-dependent motor noise on one committed macro delta."""
        cfg = self.config
        if cfg.motor_noise_rel <= 0.0 and cfg.motor_noise_perp <= 0.0:
            return delta
        white = self._rng.standard_normal(2)
        if cfg.motor_noise_tau_ms > 0.0:
            # unit-variance OU: n <- a n + sqrt(1 - a^2) w
            a = float(np.exp(-1.0 / cfg.motor_noise_tau_ms))
            self._noise_state = a * self._noise_state + np.sqrt(1.0 - a * a) * white
        else:
            self._noise_state = white
        n_par, n_perp = self._noise_state
        dx, dy = delta
        # (dx, dy) * (1 + rel n_par)  +  (-dy, dx) * perp n_perp
        return (
            dx * (1.0 + cfg.motor_noise_rel * n_par) - dy * cfg.motor_noise_perp * n_perp,
            dy * (1.0 + cfg.motor_noise_rel * n_par) + dx * cfg.motor_noise_perp * n_perp,
        )

    # ------------------------------------------------------------------
    # High-frequency generation (1 kHz)
    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        """Advance one millisecond; return one integer ``(dx, dy)`` report.

        Drains one committed macro delta, or runs idle ``(0, 0)`` when the buffer
        is empty (target in the deadzone, or a pause between updates).  Even when
        idle the renderer keeps emitting: the Sigma-Delta accumulator bleeds off
        residual momentum into a smooth log-normal tail and the GRU keeps laying
        down micro-tremor.
        """
        if self._stream is None:
            raise RuntimeError("controller stream not started; call reset()")
        if self._buffer:
            dx, dy = self._buffer.popleft()
            delta = (float(dx), float(dy))
            delta = self._add_motor_noise(delta)
        else:
            delta = (0.0, 0.0)
        report = self._stream.step(delta)
        self._cursor += report
        self._time_since_plan += 1
        return report

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def pending_ticks(self) -> int:
        """Committed macro deltas not yet consumed by :meth:`step`."""
        return len(self._buffer)

    @property
    def ticks(self) -> int:
        """Total 1 kHz ticks emitted by the current renderer stream."""
        return 0 if self._stream is None else self._stream.ticks
