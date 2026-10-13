"""Objective scoring for generated trajectories: tracking + human-likeness.

Given a 1 kHz cursor path (and optionally the target it was tracking), this
computes a compact, reusable report used both by the headless evaluator
(``tools/evaluate_trajectory.py``) and the live harness.  Two families:

* **Tracking** -- did it get there?  RMS / final / mean error to the target.
* **Human-likeness** -- does the motion obey human motor invariants?
    - Two-Thirds Power Law: fit ``log v = log alpha - beta * log kappa``;
      a human hand gives ``beta ~ 1/3`` with a high R^2.
    - Smoothness: log dimensionless jerk (LDLJ) and spectral arc length
      (SPARC) -- both more-negative = less smooth; human reaches sit in a
      characteristic band, robotic/ataxic motion is far more negative.
    - Physiological tremor: fraction of speed-spectrum power in 8-12 Hz.

All metrics are guarded against stationary / degenerate input (they return
``nan`` rather than raising), so they are safe to call on any window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - scipy is a core dep, but stay importable
    gaussian_filter1d = None


FS_HZ = 1000.0  # one report per millisecond


def _smooth(path: np.ndarray, sigma_ms: float) -> np.ndarray:
    if sigma_ms <= 0 or gaussian_filter1d is None or len(path) < 3:
        return path.astype(np.float64)
    return np.stack(
        [gaussian_filter1d(path[:, d].astype(np.float64), sigma_ms, mode="nearest")
         for d in range(path.shape[1])],
        axis=1,
    )


def two_thirds_power_law(path_xy: np.ndarray, *, smooth_ms: float = 8.0,
                         v_min: float = 20.0) -> tuple[float, float]:
    """Fit ``log v = c - beta log kappa``; return ``(beta, r2)``.

    Human curved motion gives ``beta`` near ``1/3`` with a high R^2.  Returns
    ``(nan, nan)`` if there is too little curved, moving motion to fit.
    """
    p = _smooth(np.asarray(path_xy, dtype=np.float64), smooth_ms)
    if len(p) < 16:
        return float("nan"), float("nan")
    dt = 1.0 / FS_HZ
    vx, vy = np.gradient(p[:, 0], dt), np.gradient(p[:, 1], dt)
    ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
    speed = np.hypot(vx, vy)
    cross = np.abs(vx * ay - vy * ax)
    kappa = cross / np.maximum(speed**3, 1e-9)
    moving = (speed > v_min) & (kappa > 1e-6) & np.isfinite(kappa)
    if int(np.count_nonzero(moving)) < 32:
        return float("nan"), float("nan")
    lv = np.log(speed[moving])
    lk = np.log(kappa[moving])
    beta, _ = np.polyfit(lk, lv, 1)
    pred = np.polyval([beta, _], lk)
    ss_res = float(np.sum((lv - pred) ** 2))
    ss_tot = float(np.sum((lv - lv.mean()) ** 2)) or 1e-12
    return float(-beta), float(1.0 - ss_res / ss_tot)


def log_dimensionless_jerk(path_xy: np.ndarray, *, smooth_ms: float = 6.0) -> float:
    """LDLJ smoothness (more negative = less smooth); ``nan`` if ~stationary."""
    p = _smooth(np.asarray(path_xy, dtype=np.float64), smooth_ms)
    if len(p) < 8:
        return float("nan")
    dt = 1.0 / FS_HZ
    vx, vy = np.gradient(p[:, 0], dt), np.gradient(p[:, 1], dt)
    jx, jy = np.gradient(np.gradient(vx, dt), dt), np.gradient(np.gradient(vy, dt), dt)
    speed = np.hypot(vx, vy)
    peak = float(np.max(speed))
    duration = len(p) * dt
    if peak < 1e-6 or duration <= 0:
        return float("nan")
    integral = float(np.sum((jx**2 + jy**2)) * dt)
    dlj = (duration**3 / peak**2) * integral
    return float(-np.log(max(dlj, 1e-12)))


def sparc(path_xy: np.ndarray, *, fc: float = 20.0, amp_th: float = 0.05) -> float:
    """Spectral arc length of the speed profile (more negative = less smooth)."""
    p = np.asarray(path_xy, dtype=np.float64)
    if len(p) < 16:
        return float("nan")
    dt = 1.0 / FS_HZ
    speed = np.hypot(np.gradient(p[:, 0], dt), np.gradient(p[:, 1], dt))
    if np.max(speed) < 1e-6:
        return float("nan")
    n = int(2 ** np.ceil(np.log2(len(speed) * 4)))
    mag = np.abs(np.fft.rfft(speed, n))
    freq = np.fft.rfftfreq(n, dt)
    if mag[0] <= 0:
        return float("nan")
    mag = mag / mag[0]
    band = freq <= fc
    f, m = freq[band], mag[band]
    keep = m >= amp_th
    if int(np.count_nonzero(keep)) < 3:
        return float("nan")
    f, m = f[keep], m[keep]
    df = (f[-1] - f[0]) or 1e-9
    return float(-np.sum(np.sqrt((np.diff(f) / df) ** 2 + np.diff(m) ** 2)))


def tremor_band_fraction(path_xy: np.ndarray, *, lo: float = 8.0, hi: float = 12.0) -> float:
    """Fraction of speed-spectrum power in the ``lo..hi`` Hz band (tremor)."""
    p = np.asarray(path_xy, dtype=np.float64)
    if len(p) < 64:
        return float("nan")
    dt = 1.0 / FS_HZ
    speed = np.hypot(np.gradient(p[:, 0], dt), np.gradient(p[:, 1], dt))
    speed = speed - speed.mean()
    power = np.abs(np.fft.rfft(speed)) ** 2
    freq = np.fft.rfftfreq(len(speed), dt)
    total = float(np.sum(power[freq > 0])) or 1e-12
    band = float(np.sum(power[(freq >= lo) & (freq <= hi)]))
    return float(band / total)


@dataclass
class TrajectoryReport:
    samples: int
    mean_speed: float          # counts/s
    peak_speed: float          # counts/s
    rms_error: float           # counts (nan if no target)
    final_error: float         # counts (nan if no target)
    two_thirds_beta: float     # ~0.33 for human curved motion
    two_thirds_r2: float       # fit quality of the power law
    ldlj: float                # smoothness (more negative = less smooth)
    sparc: float               # smoothness (more negative = less smooth)
    tremor_8_12_frac: float    # physiological tremor power fraction

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(cursor_xy, target_xy=None) -> TrajectoryReport:
    """Score a 1 kHz cursor path ``[N,2]``; ``target_xy`` ``[N,2]`` is optional."""
    p = np.asarray(cursor_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("cursor_xy must be [N, 2]")
    dt = 1.0 / FS_HZ
    speed = np.hypot(np.gradient(p[:, 0], dt), np.gradient(p[:, 1], dt)) if len(p) > 2 else np.zeros(1)
    rms = final = float("nan")
    if target_xy is not None:
        t = np.asarray(target_xy, dtype=np.float64)
        m = min(len(t), len(p))
        err = np.hypot(p[:m, 0] - t[:m, 0], p[:m, 1] - t[:m, 1])
        rms = float(np.sqrt(np.mean(err**2)))
        final = float(err[-1])
    beta, r2 = two_thirds_power_law(p)
    return TrajectoryReport(
        samples=len(p),
        mean_speed=float(np.mean(speed)),
        peak_speed=float(np.max(speed)),
        rms_error=rms,
        final_error=final,
        two_thirds_beta=beta,
        two_thirds_r2=r2,
        ldlj=log_dimensionless_jerk(p),
        sparc=sparc(p),
        tremor_8_12_frac=tremor_band_fraction(p),
    )
