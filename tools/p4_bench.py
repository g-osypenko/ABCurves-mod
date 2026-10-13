"""Operating-point bench (P4): every axis that matters, for each config variant.

Tuning one number against one metric is how a fix for texture quietly costs
tracking accuracy (or brings back a pathology).  This bench scores each variant
on all of them at once:

* **texture** -- Texture19 distance to human reaches on held-out rows (rows not
  used for P3 tuning), per-row seeds + velocity handoff, as ``texture_eval``;
* **pursuit** -- RMS tracking error on the moving-target scenarios of
  ``evaluate_trajectory`` (B2' curvature on), plus the step scenario;
* **adversarial** -- the P1 symptom checks as numbers: target-jitter
  amplification, hard-reversal overshoot/ringing, pause->re-acquire latency and
  overshoot;
* **cost** -- mean wall time of one ``on_target_update`` (the replan).

Usage::

    python tools/p4_bench.py
    python tools/p4_bench.py --variants '[{"name":"h320","cfg":{"horizon_ms":320}}]'
    python tools/p4_bench.py --variants variants.json --rows 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from abcurves.closed_loop import ClosedLoopConfig, ClosedLoopProDMPController
from abcurves.forcing_library import ForcingLibrary
from abcurves.model_store import default_model_dir
from abcurves.portable_renderer import PortableRendererModel
from abcurves.texture import texture_features
from abcurves.trajectory_metrics import evaluate
from evaluate_trajectory import make_scenario
from texture_eval import standardized_panel_w1

LIB_PATH = Path("models/continuous/forcing_lib.npz")
DATA_PATH = Path("examples/aim_test.npz")
PURSUIT = ("circle", "figure8", "ramp", "random")


# --- shared closed-loop driver ---------------------------------------------------
def drive(ctl, target_fn, ms, update_ms, *, cursor0=None, timings=None):
    """Run closed-loop; return (reports, cursor path, target path)."""
    cursor = np.zeros(2) if cursor0 is None else np.asarray(cursor0, float).copy()
    reps, cur_path, tgt_path = [], [], []
    t = 0
    while t < ms:
        tgt = np.asarray(target_fn(float(t)), dtype=np.float64)
        t0 = time.perf_counter()
        ctl.on_target_update((tgt[0] - cursor[0], tgt[1] - cursor[1]), update_ms)
        if timings is not None:
            timings.append(time.perf_counter() - t0)
        for _ in range(update_ms):
            if t >= ms:
                break
            r = ctl.step().astype(np.float64)
            cursor += r
            reps.append(r)
            cur_path.append(cursor.copy())
            tgt_path.append(np.asarray(target_fn(float(t)), dtype=np.float64))
            t += 1
    return np.asarray(reps), np.asarray(cur_path), np.asarray(tgt_path)


# --- texture on held-out human rows ---------------------------------------------
def holdout_rows(n_rows: int, seed: int = 11):
    data = np.load(DATA_PATH)
    n = len(data["future_mask"])
    tuned = np.random.default_rng(7).choice(n, 128, replace=False)  # P3 tuning rows
    rest = np.setdiff1d(np.arange(n), tuned)
    sel = np.random.default_rng(seed).choice(rest, min(n_rows, len(rest)), replace=False)
    keys = ("future_raw_dxdy", "future_mask", "prefix_raw_dxdy", "prefix_mask",
            "target_rel_x_at_B", "target_rel_y_at_B")
    return {k: data[k][sel] for k in keys}


def texture_distance(cfg, rows, model, library, update_ms=8, seed_base=1000) -> float:
    hd = rows["future_raw_dxdy"].astype(np.float64)
    hm = rows["future_mask"].astype(np.float64)
    hmax = hd.shape[1]
    bufs, masks = [], []
    for i in range(len(hm)):
        pre = rows["prefix_raw_dxdy"][i][rows["prefix_mask"][i] > 0.5]
        ctx = np.zeros((256, 2), dtype=np.int16)
        if len(pre):
            p = np.rint(pre[-256:]).astype(np.int16)
            ctx[-len(p):] = p
        ctl = ClosedLoopProDMPController(config=replace(cfg, event_seed=seed_base + i),
                                         renderer_model=model, renderer_context=ctx,
                                         forcing_library=library)
        ctl.reset(initial_velocity=pre[-16:].mean(axis=0) if len(pre) else None)
        target = (float(rows["target_rel_x_at_B"][i]), float(rows["target_rel_y_at_B"][i]))
        dur = int(min(max(int(hm[i].sum()), 32), hmax))
        reps, _, _ = drive(ctl, lambda t: target, dur, update_ms)
        b = np.zeros((hmax, 2)); m = np.zeros(hmax)
        k = min(len(reps), hmax); b[:k] = reps[:k]; m[:k] = 1.0
        bufs.append(b); masks.append(m)
    human = texture_features(hd, hm)
    gen = texture_features(np.asarray(bufs), np.asarray(masks))
    return standardized_panel_w1(gen, human, scale_reference=human)


# --- pursuit accuracy -------------------------------------------------------------
def pursuit(cfg, model, library, seconds=6.0, update_ms=16, timings=None) -> dict:
    out = {}
    for name in PURSUIT + ("step",):
        fn = make_scenario(name, 200.0, 2000.0, np.random.default_rng(7))
        ctl = ClosedLoopProDMPController(config=cfg, renderer_model=model, forcing_library=library)
        _, cur, tgt = drive(ctl, fn, int(seconds * 1000), update_ms, timings=timings)
        out[name] = float(evaluate(cur, tgt).rms_error)
    out["pursuit_mean"] = float(np.mean([out[n] for n in PURSUIT]))
    return out


# --- adversarial symptoms (P1) as numbers -----------------------------------------
def adversarial(cfg, model) -> dict:
    def ctl():
        return ClosedLoopProDMPController(config=cfg, renderer_model=model)

    def speed(r):
        return np.hypot(r[:, 0], r[:, 1]) * 1000.0

    clean = speed(drive(ctl(), lambda t: (150.0, 0.0), 900, 8)[0])
    rng = np.random.default_rng(0)
    jit = speed(drive(ctl(), lambda t: (150.0 + rng.normal(0, 1.5), rng.normal(0, 1.5)),
                      900, 8)[0])
    _, path, _ = drive(ctl(), lambda t: (120.0, 0.0) if t < 400 else (-120.0, 0.0), 1400, 8)
    x = path[:, 0]
    vx = np.gradient(gaussian_filter1d(x, 8.0))[460:]
    tgt = np.array([220.0, -90.0])
    rep, path2, _ = drive(ctl(), lambda t: (0.0, 0.0) if t < 500 else tuple(tgt), 1600, 8)
    after = np.hypot(rep[500:, 0], rep[500:, 1])
    along = path2 @ (tgt / np.hypot(*tgt))
    return {
        "jitter_peak_ratio": float(jit.max() / clean.max()),
        "jitter_rms_ratio": float(np.sqrt(np.mean(jit**2)) / np.sqrt(np.mean(clean**2))),
        "reversal_overshoot": float(max(0.0, -120.0 - x.min())),
        "reversal_ringing": int(np.sum(vx[1:] * vx[:-1] < 0)),
        "reacquire_ms": int(np.argmax(after > 0.8)) if np.any(after > 0.8) else 10_000,
        "reacquire_overshoot": float(max(0.0, along.max() - np.hypot(*tgt))),
    }


DEFAULT_VARIANTS = [
    {"name": "current defaults"},
    {"name": "h320", "cfg": {"horizon_ms": 320}},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default=None, help="JSON list (or .json file) of "
                    '{"name": ..., "cfg": {ClosedLoopConfig overrides}}')
    ap.add_argument("--rows", type=int, default=192, help="held-out human rows for texture")
    ap.add_argument("--no-texture", action="store_true", help="skip the (slow) texture axis")
    ap.add_argument("--json", default=None, help="write all results to this path")
    args = ap.parse_args()

    if args.variants is None:
        variants = DEFAULT_VARIANTS
    elif args.variants.endswith(".json"):
        variants = json.loads(Path(args.variants).read_text())
    else:
        variants = json.loads(args.variants)

    library = ForcingLibrary.load(LIB_PATH) if LIB_PATH.is_file() else None
    model = PortableRendererModel(default_model_dir() / "renderer_global_h80.bin")
    rows = None if args.no_texture else holdout_rows(args.rows)

    head = (f"{'variant':30s} {'texture':>7s} | {'circle':>6s} {'fig8':>6s} {'ramp':>6s} "
            f"{'random':>6s} {'p.mean':>6s} {'step':>6s} | {'jitPk':>5s} {'jitRms':>6s} "
            f"{'revOS':>5s} {'ring':>4s} {'reacq':>5s} {'reaOS':>5s} | {'replan':>7s}")
    print(head)
    print("-" * len(head))
    results = []
    for v in variants:
        cfg = ClosedLoopConfig(**v.get("cfg", {}))
        print(f"  running {v['name']} ...", end="\r", file=sys.stderr, flush=True)
        tex = float("nan") if rows is None else texture_distance(cfg, rows, model, library)
        timings: list = []
        pur = pursuit(cfg, model, library, timings=timings)
        adv = adversarial(cfg, model)
        replan_us = 1e6 * float(np.mean(timings))
        res = {"name": v["name"], "cfg": v.get("cfg", {}), "texture": tex, **pur, **adv,
               "replan_us": replan_us}
        results.append(res)
        print(f"{v['name'][:30]:30s} {tex:7.3f} | {pur['circle']:6.1f} {pur['figure8']:6.1f} "
              f"{pur['ramp']:6.1f} {pur['random']:6.1f} {pur['pursuit_mean']:6.1f} "
              f"{pur['step']:6.1f} | {adv['jitter_peak_ratio']:5.2f} {adv['jitter_rms_ratio']:6.2f} "
              f"{adv['reversal_overshoot']:5.1f} {adv['reversal_ringing']:4d} "
              f"{adv['reacquire_ms']:5d} {adv['reacquire_overshoot']:5.1f} | {replan_us:5.0f}us",
              flush=True)

    print("\ntexture: lower = more human (human-vs-human between ~0.50, within ~0.17) | "
          "pursuit rms: lower = more accurate\n"
          "adversarial gates (tests): jitPk/jitRms <= 1.3, revOS < 20, ring <= 3, "
          "reacq < 30 ms, reaOS < 30 | replan = mean on_target_update cost")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
