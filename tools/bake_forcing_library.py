"""Offline bake of the ProDMP forcing-weight library (decision B2').

Runs the continuous ``ContinuousPlanner`` (its own torch-free native inference)
over many randomized goal-directed moves, slices the generated macro trajectory
into horizon-length windows, and fits ``abcurves.prodmp.ProDMP`` forcing weights
to each window with ``goal_mode='target'``.  Each fit is canonicalized (rotated
so the net move points along +x, scaled to unit distance) and stored, giving a
compact, direction/scale-agnostic library of human-like macro curvatures.

This is a one-time tool.  The live controller never runs the motor; it only
loads the resulting ``.npz`` via :class:`abcurves.forcing_library.ForcingLibrary`.

Usage::

    python tools/bake_forcing_library.py --runs 120 --out models/continuous/forcing_lib.npz

Requires numpy/scipy/numba (the continuous native backend).  On Python < 3.11 a
``hashlib.file_digest`` shim is installed automatically (the continuous asset
loader needs it); the library itself targets 3.11+.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

# --- Python < 3.11 shim: the continuous asset loader uses hashlib.file_digest
if not hasattr(hashlib, "file_digest"):  # pragma: no cover - version dependent
    def _file_digest(fileobj, digest, /, *, _bufsize=1 << 18):
        h = hashlib.new(digest) if isinstance(digest, str) else digest()
        while True:
            chunk = fileobj.read(_bufsize)
            if not chunk:
                break
            h.update(chunk)
        return h

    hashlib.file_digest = _file_digest  # type: ignore[attr-defined]

import abcurves  # noqa: E402  (after the shim)
from abcurves.prodmp import ProDMP, ProDMPConfig  # noqa: E402


def _random_targets(rng: np.random.Generator, n: int, reach: float):
    """A sequence of waypoints that forces direction changes (hence curvature)."""
    pts = []
    pos = np.zeros(2)
    for _ in range(n):
        step = rng.normal(size=2)
        step = step / (np.hypot(*step) + 1e-9) * rng.uniform(0.3, 1.0) * reach
        pos = pos + step
        pts.append(pos.copy())
    return pts


def harvest(runs: int, reach: float, seed: int) -> np.ndarray:
    """Return generated trajectories, one [T, 2] array per run."""
    rng = np.random.default_rng(seed)
    trajs = []
    for r in range(runs):
        planner = abcurves.load(seed=int(rng.integers(0, 2**31)))
        t_us = 0
        xy_all = [np.zeros((1, 2))]
        for wp in _random_targets(rng, n=rng.integers(3, 7), reach=reach):
            planner.update_target((float(wp[0]), float(wp[1])), timestamp_us=t_us)
            dt_us = int(rng.integers(300_000, 700_000))
            t_us += dt_us
            out = planner.advance(t_us)
            xy_all.append(np.asarray(out["xy"]))
        traj = np.concatenate(xy_all, axis=0)
        if len(traj) > 10 and np.all(np.isfinite(traj)):
            trajs.append(traj)
    return trajs


def fit_shapes(
    trajs, dmp: ProDMP, window_ms: int, stride_ms: int, min_dist: float, max_resid: float
) -> list[np.ndarray]:
    """Fit canonical forcing shapes to horizon windows of each trajectory."""
    shapes = []
    for traj in trajs:
        for start in range(0, len(traj) - window_ms, stride_ms):
            win = traj[start : start + window_ms]
            rel = win - win[0]
            goal = rel[-1]
            dist = float(np.hypot(goal[0], goal[1]))
            if dist < min_dist:
                continue
            # boundary velocity = measured start slope (counts/ms), lightly smoothed
            ydot_b = np.mean(np.diff(rel[:4], axis=0), axis=0)
            d = len(rel)
            w_g = dmp.fit_weights(rel, ydot_b, d, y_b=np.zeros(2), goal=goal, goal_mode="target")
            recon = dmp.generate(w_g, ydot_b, d, y_b=np.zeros(2), goal=goal, goal_mode="target")
            resid = float(np.sqrt(np.mean(np.sum((recon - rel) ** 2, axis=1)))) / dist
            if not np.isfinite(resid) or resid > max_resid:
                continue
            forcing = w_g[:-1]  # drop goal row
            c, s = goal[0] / dist, goal[1] / dist
            rot = np.array([[c, -s], [s, c]], dtype=np.float64)  # +x -> goal dir
            canonical = (forcing / dist) @ rot  # inverse-orient into +x unit frame
            if np.all(np.isfinite(canonical)):
                shapes.append(canonical)
    return shapes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/continuous/forcing_lib.npz")
    ap.add_argument("--runs", type=int, default=120)
    ap.add_argument("--reach", type=float, default=120.0, help="waypoint step scale (counts)")
    ap.add_argument("--window-ms", type=int, default=250)
    ap.add_argument("--stride-ms", type=int, default=40)
    ap.add_argument("--min-dist", type=float, default=15.0, help="skip near-stationary windows")
    ap.add_argument("--max-resid", type=float, default=0.15, help="keep only well-represented fits")
    ap.add_argument("--max-shapes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()

    dmp = ProDMP(ProDMPConfig())
    print(f"harvesting {args.runs} runs from the continuous motor ...")
    trajs = harvest(args.runs, args.reach, args.seed)
    print(f"  got {len(trajs)} trajectories "
          f"({sum(len(t) for t in trajs)} ms of motion)")
    shapes = fit_shapes(
        trajs, dmp, args.window_ms, args.stride_ms, args.min_dist, args.max_resid
    )
    print(f"fit {len(shapes)} forcing shapes (n_basis={dmp.n_basis})")
    if not shapes:
        raise SystemExit("no shapes harvested; loosen --min-dist / --max-resid")
    shapes = np.asarray(shapes, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    if len(shapes) > args.max_shapes:
        keep = rng.choice(len(shapes), args.max_shapes, replace=False)
        shapes = shapes[keep]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, shapes=shapes.astype(np.float32), n_basis=np.int64(dmp.n_basis))
    mag = np.linalg.norm(shapes.reshape(len(shapes), -1), axis=1)
    print(f"saved {len(shapes)} shapes -> {out}")
    print(f"  shape magnitude: mean={mag.mean():.3f} p95={np.percentile(mag,95):.3f}")


if __name__ == "__main__":
    main()
