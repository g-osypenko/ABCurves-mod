"""Measure the hardware-texture human-likeness of the closed-loop controller.

Runs the controller's 1 kHz integer output through the repository's own frozen
``Texture19`` descriptor (:func:`abcurves.texture.texture_features`) and compares
it to recorded human 1 kHz reports with the repository's distance method
(robust per-feature standardized Wasserstein-1, as in ``evaluation.floors``).

The headline rule mirrors the detection study: texture is "inside the human
range" when the generated-vs-human distance is not larger than the distance
*between humans themselves*.  Each generated reach uses the renderer-context
(hand/mouse texture profile) of the human row it is matched to, so only the
generator's texture differs, not the device.

Usage::

    python tools/texture_eval.py --rows 256
    python tools/texture_eval.py --rows 128 --data examples/aim_test.npz
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from abcurves.closed_loop import ClosedLoopConfig, ClosedLoopProDMPController
from abcurves.forcing_library import ForcingLibrary
from abcurves.model_store import default_model_dir
from abcurves.portable_renderer import PortableRendererModel
from abcurves.texture import TEXTURE_FEATURE_NAMES, texture_features

LIB_PATH = Path("models/continuous/forcing_lib.npz")


# --- repository distance method (copied from evaluation/floors.py, torch-free) ---
def _robust_scale(reference: np.ndarray) -> np.ndarray:
    values = np.asarray(reference, dtype=np.float64)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = values.std(axis=0)
    return np.where(scale >= 1e-8, scale, np.where(fallback >= 1e-8, fallback, 1.0))


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    a = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    support = np.sort(np.concatenate([a, b]))
    if len(support) < 2:
        return 0.0
    widths = np.diff(support)
    cdf_a = np.searchsorted(a, support[:-1], side="right") / float(len(a))
    cdf_b = np.searchsorted(b, support[:-1], side="right") / float(len(b))
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def standardized_panel_w1(left, right, *, scale_reference) -> float:
    a, b = np.asarray(left, float), np.asarray(right, float)
    scale = _robust_scale(np.asarray(scale_reference, float))
    return float(np.mean([
        _wasserstein_1d(a[:, c] / scale[c], b[:, c] / scale[c]) for c in range(a.shape[1])
    ]))


def per_feature_w1(left, right, *, scale_reference) -> np.ndarray:
    a, b = np.asarray(left, float), np.asarray(right, float)
    scale = _robust_scale(np.asarray(scale_reference, float))
    return np.array([
        _wasserstein_1d(a[:, c] / scale[c], b[:, c] / scale[c]) for c in range(a.shape[1])
    ])


def _stable_half(ids, namespace: str) -> np.ndarray:
    return np.array([hashlib.sha256(f"{namespace}|{s}".encode()).digest()[0] & 1 for s in ids],
                    dtype=np.int8)


# --- generate one closed-loop reach as 1 kHz integer reports --------------------
def generate_reach(model, ctx_int16, target, duration, *, library, gain, update_ms=8):
    cfg = ClosedLoopConfig(event_seed=7, forcing_gain=gain)
    ctl = ClosedLoopProDMPController(
        config=cfg, renderer_model=model, renderer_context=ctx_int16, forcing_library=library
    )
    cursor = np.zeros(2)
    reports = []
    t = 0
    while t < duration:
        off = np.asarray(target, float) - cursor
        ctl.on_target_update((off[0], off[1]), update_ms)
        for _ in range(update_ms):
            if t >= duration:
                break
            r = ctl.step().astype(np.float64)
            cursor += r
            reports.append(r)
            t += 1
    return np.asarray(reports)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="examples/aim_test.npz")
    ap.add_argument("--rows", type=int, default=256, help="number of human rows to match")
    ap.add_argument("--gain", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = np.load(args.data)
    n_all = len(data["future_mask"])
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(n_all, min(args.rows, n_all), replace=False)

    human_dxdy = data["future_raw_dxdy"][sel].astype(np.float64)
    human_mask = data["future_mask"][sel].astype(np.float64)
    prefix_dxdy = data["prefix_raw_dxdy"][sel].astype(np.float64)
    prefix_mask = data["prefix_mask"][sel].astype(np.float64)
    tx = data["target_rel_x_at_B"][sel].astype(np.float64)
    ty = data["target_rel_y_at_B"][sel].astype(np.float64)
    users = data["user_id"][sel].astype(str)
    durations = human_mask.sum(axis=1).astype(int)

    library = ForcingLibrary.load(LIB_PATH) if LIB_PATH.is_file() else None
    model = PortableRendererModel(default_model_dir() / "renderer_global_h80.bin")
    print(f"renderer: {model.receipt.mode} | rows: {len(sel)} | "
          f"B2' library: {'yes' if library else 'no'}\n")

    gens = {"B1": [], "B2'": []}
    gmask = {"B1": [], "B2'": []}
    hmax = int(human_dxdy.shape[1])
    for i in range(len(sel)):
        pre = prefix_dxdy[i][prefix_mask[i] > 0.5]
        ctx = np.zeros((256, 2), dtype=np.int16)
        if len(pre):
            pre = np.rint(pre[-256:]).astype(np.int16)
            ctx[-len(pre):] = pre
        dur = int(min(max(durations[i], 32), hmax))
        for label, gain in (("B1", 0.0), ("B2'", args.gain)):
            lib = None if gain == 0.0 else library
            reach = generate_reach(model, ctx, (tx[i], ty[i]), dur, library=lib, gain=gain)
            buf = np.zeros((hmax, 2)); m = np.zeros(hmax)
            k = min(len(reach), hmax)
            buf[:k] = reach[:k]; m[:k] = 1.0
            gens[label].append(buf); gmask[label].append(m)

    # Texture19 descriptors
    H = texture_features(human_dxdy, human_mask)
    G = {k: texture_features(np.asarray(v), np.asarray(gmask[k])) for k, v in gens.items()}

    # human<->human anchors (scale always fit on all human rows)
    bh = _stable_half(users, "between-human")        # split by nothing-but-id parity
    rh = _stable_half([str(i) for i in range(len(sel))], "within-human")
    between = standardized_panel_w1(H[bh == 0], H[bh == 1], scale_reference=H)
    within = standardized_panel_w1(H[rh == 0], H[rh == 1], scale_reference=H)

    print("Texture19 standardized Wasserstein-1 distance (lower = closer):")
    print(f"  human split (within-population floor) : {within:.3f}")
    print(f"  human split (between halves)          : {between:.3f}")
    for label in ("B1", "B2'"):
        d = standardized_panel_w1(G[label], H, scale_reference=H)
        verdict = "INSIDE human range" if d <= between else ("near edge" if d <= 1.5 * between else "OUTSIDE")
        print(f"  {label:4s} output vs human               : {d:.3f}   [{verdict}]")

    print("\nTop per-feature deviations (B2' vs human, standardized W1):")
    pf = per_feature_w1(G["B2'"], H, scale_reference=H)
    for idx in np.argsort(pf)[::-1][:6]:
        print(f"  {TEXTURE_FEATURE_NAMES[idx]:22s} {pf[idx]:.3f}")

    print("\nReading it: a value near the human-split anchors means the texture is "
          "statistically inside\nhuman variation; a few large per-feature gaps point at "
          "exactly which texture property\nto improve. Scale is fit on human rows, per the "
          "repository's floors method.")


if __name__ == "__main__":
    main()
