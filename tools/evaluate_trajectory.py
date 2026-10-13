"""Headless, reproducible scoring of the closed-loop controller.

The program drives the target along scripted scenarios (the user no longer has
to move the mouse), runs the 1 kHz generator closed-loop, and scores the result
with :mod:`abcurves.trajectory_metrics` -- both tracking accuracy and human-motor
invariants.  It reports B1 (pure homing) next to B2' (baked macro curvature) so
the effect of the forcing library is objective, not a vibe.

Usage::

    python tools/evaluate_trajectory.py
    python tools/evaluate_trajectory.py --scenario circle --seconds 6 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from abcurves.closed_loop import ClosedLoopConfig, ClosedLoopProDMPController
from abcurves.forcing_library import ForcingLibrary
from abcurves.trajectory_metrics import evaluate

LIB_PATH = Path("models/continuous/forcing_lib.npz")


# --- scenarios: t_ms -> target (x, y) in native counts ----------------------
def make_scenario(name: str, reach: float, period_ms: float, rng: np.random.Generator):
    if name == "step":
        pts = np.array([[reach, 0], [reach, reach], [-reach, reach * 0.5],
                        [-reach * 0.6, -reach], [0, 0]], dtype=float)
        return lambda t: pts[int(t // period_ms) % len(pts)]
    if name == "circle":
        w = 2 * np.pi / period_ms
        return lambda t: np.array([reach * np.cos(w * t), reach * np.sin(w * t)])
    if name == "figure8":
        w = 2 * np.pi / period_ms
        return lambda t: np.array([reach * np.sin(w * t), reach * np.sin(2 * w * t) * 0.5])
    if name == "ramp":
        return lambda t: np.array([reach * (t / period_ms - 1.0), reach * 0.3 * np.sin(t / period_ms)])
    if name == "random":
        knots = rng.normal(size=(64, 2)) * reach
        def f(t):
            k = t / period_ms
            i = int(k) % len(knots)
            a = k - int(k)
            return (1 - a) * knots[i] + a * knots[(i + 1) % len(knots)]
        return f
    raise ValueError(f"unknown scenario '{name}'")


SCENARIOS = ("step", "circle", "figure8", "ramp", "random")


def run(ctl: ClosedLoopProDMPController, target_fn, seconds: float, update_ms: int):
    ms = int(seconds * 1000)
    cursor = np.zeros(2)
    ctl.reset()
    cur_path, tgt_path = [], []
    t = 0
    while t < ms:
        tgt = np.asarray(target_fn(float(t)), dtype=np.float64)
        ctl.on_target_update((tgt[0] - cursor[0], tgt[1] - cursor[1]), update_ms)
        for _ in range(update_ms):
            if t >= ms:
                break
            tgt = np.asarray(target_fn(float(t)), dtype=np.float64)
            cursor = cursor + ctl.step().astype(np.float64)
            cur_path.append(cursor.copy())
            tgt_path.append(tgt.copy())
            t += 1
    return np.asarray(cur_path), np.asarray(tgt_path)


def _row(label: str, rep) -> str:
    return (f"{label:16s} rms={rep.rms_error:7.2f} final={rep.final_error:6.2f} "
            f"| 2/3 beta={rep.two_thirds_beta:5.2f} (R2={rep.two_thirds_r2:4.2f}) "
            f"| LDLJ={rep.ldlj:7.2f} SPARC={rep.sparc:7.2f} "
            f"| tremor8-12={rep.tremor_8_12_frac:5.2f} vpk={rep.peak_speed:6.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=SCENARIOS + ("all",), default="all")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--reach", type=float, default=200.0, help="scenario size (counts)")
    ap.add_argument("--period-ms", type=float, default=2000.0, help="scenario period")
    ap.add_argument("--update-ms", type=int, default=16, help="target update cadence")
    ap.add_argument("--gain", type=float, default=0.45, help="B2' curvature gain")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None, help="write the full report to this path")
    args = ap.parse_args()

    library = ForcingLibrary.load(LIB_PATH) if LIB_PATH.is_file() else None
    modes = [("B1 (homing)", None, 0.0)]
    if library is not None:
        modes.append((f"B2' (gain {args.gain})", library, args.gain))
    else:
        print("note: no forcing_lib.npz; scoring B1 only")

    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    report: dict = {}
    print(f"scenarios: {', '.join(names)}  |  {args.seconds}s @ {args.update_ms}ms updates\n")
    for name in names:
        print(f"[{name}]")
        report[name] = {}
        for label, lib, gain in modes:
            rng = np.random.default_rng(args.seed)
            target_fn = make_scenario(name, args.reach, args.period_ms, rng)
            ctl = ClosedLoopProDMPController(
                config=ClosedLoopConfig(event_seed=args.seed, forcing_gain=gain),
                forcing_library=lib,
            )
            cur, tgt = run(ctl, target_fn, args.seconds, args.update_ms)
            rep = evaluate(cur, tgt)
            print("  " + _row(label, rep))
            report[name][label] = rep.as_dict()
        print()

    print("Reading the numbers: lower rms/final = better tracking; 2/3 beta near 0.33 with "
          "high R2 = human curvature;\nLDLJ/SPARC closer to 0 = smoother; tremor8-12 > 0 = "
          "physiological micro-tremor present.")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
