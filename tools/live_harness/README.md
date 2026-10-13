# Live harness — closed-loop ProDMP controller

A zero-dependency (stdlib-only) local test rig for the
`ClosedLoopProDMPController`. Move your mouse over the canvas; the server runs
the real 1 kHz generator and draws the trajectory it produces as it tracks your
pointer.

## Run

```bash
python tools/live_harness/server.py
```

It prints a local and a LAN URL (it binds `0.0.0.0` so you can open it from a
phone or another PC on the same network). Pass `--host 127.0.0.1` to keep it
local only.

```bash
python tools/live_harness/server.py --host 127.0.0.1 --port 9000
python tools/live_harness/server.py --update-ms 8 --gain 0.45 --reach 200 --period-ms 2600
# compare predictive pursuit live (defaults follow ClosedLoopConfig):
python tools/live_harness/server.py --ff off --horizon 250        # old behaviour
python tools/live_harness/server.py --ff raw --port 8766          # plain dx/dt (jitter-prone)
```

On Python 3.11+ run it directly; this repo's continuous stack targets 3.11+, but
the controller + harness also run on 3.9 (they avoid `hashlib.file_digest`).

## Controls

- **drive** — `manual · mouse` (follow your pointer) or an auto scenario
  (`circle`, `figure-8`, `steps`, `random`). In auto modes **the program drives
  the target itself**, so a run is reproducible without touching the mouse.
- **pause / resume** — freezes the simulation so you can inspect the trajectory;
  the trail stays on screen. Resume continues from where it stopped.
- **curvature (B2′) slider** — live `forcing_gain`. `0` = pure minimum-jerk
  homing (B1); `~0.45` = baked human-like macro curvature. Disabled with no lib.
- **deadzone toggle**, **reset**.
- **zoom into any region** — mouse **wheel** zooms at the pointer; **Shift+drag** draws a
  box and zooms into it; **right-drag** pans (left-drag also pans in auto modes or while
  paused); **double-click** or **fit** resets the view. Zoom goes up to 400x; the grid
  adapts (its spacing in counts is shown in the footer) and once single counts are
  resolvable every 1 ms report is drawn as a dot. **follow** keeps the generated cursor
  centred (auto modes / paused).

## What you see

- **orange crosshair** — the target (mouse or scripted), in native-count space.
- **blue dot + fading trail** — the generated cursor.
- footer line 1: residual error, buffered macro ticks, total 1 kHz ticks, renderer.
- footer line 2 — **objective score** over the last ~8 s: RMS tracking error,
  two-thirds-law β and R² (β≈0.33 + high R² = human curvature), smoothness
  (LDLJ, SPARC; nearer 0 = smoother), and 8–12 Hz physiological-tremor fraction.

## How it works

The server owns a **simulation clock**. The browser only advances sim time (one
tick = 1 ms), picks the mode, and — in manual mode — sends the mouse target. The
server re-reads the target and calls `on_target_update` **once every
`--update-ms` ticks (8 ms by default)**, independent of the browser frame rate,
then steps the 1 kHz generator and returns the produced trajectory. This makes
the update cadence exact and the auto scenarios reproducible. All controller state
(persistent GRU renderer, current plan, analytic boundary velocity) lives
server-side; `/api/score` reads the rolling 8 s log through
`abcurves.trajectory_metrics`.
