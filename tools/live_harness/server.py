"""Live local harness for the closed-loop ProDMP controller.

A zero-dependency (stdlib-only) test rig: open the page on this machine or any
device on the LAN, move your mouse over the canvas, and watch the generated
cursor track it in real time.  The browser sends the target position; the server
owns a :class:`ClosedLoopProDMPController`, computes the closed-loop error, runs
the 1 kHz generator, and returns the trajectory it produced.

Run::

    python tools/live_harness/server.py            # http://0.0.0.0:8765
    python tools/live_harness/server.py --port 9000 --scale 2.0

Then open the printed URL.  Binding to 0.0.0.0 exposes it to your local network
(so you can test from a phone/another PC); pass ``--host 127.0.0.1`` to keep it
local only.
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from collections import deque

from abcurves.closed_loop import ClosedLoopConfig, ClosedLoopProDMPController
from abcurves.forcing_library import ForcingLibrary
from abcurves.trajectory_metrics import evaluate

HERE = Path(__file__).resolve().parent
MAX_BATCH_MS = 80  # clamp a single request's worth of stepping
LOG_MS = 8000      # rolling 1 kHz window kept for objective scoring


def scenario_target(mode: str, t_ms: float, manual, reach: float,
                    period_ms: float, knots: np.ndarray) -> np.ndarray:
    """Target position (native counts) for ``mode`` at simulation time ``t_ms``.

    The *paths* match the headless evaluator and the old client patterns; only
    the update cadence changes (see :meth:`Engine.frame`).
    """
    w = 2.0 * np.pi / period_ms
    if mode == "circle":
        return np.array([reach * np.cos(w * t_ms), reach * np.sin(w * t_ms)])
    if mode == "figure8":
        return np.array([reach * np.sin(w * t_ms), 0.5 * reach * np.sin(2 * w * t_ms)])
    if mode == "step":
        pts = np.array([[reach, 0], [reach, reach], [-reach, reach * 0.5],
                        [-reach * 0.6, -reach], [0, 0]], dtype=float)
        return pts[int(t_ms // period_ms) % len(pts)]
    if mode == "random":
        k = t_ms / period_ms
        i = int(k) % len(knots)
        a = k - int(k)
        return (1 - a) * knots[i] + a * knots[(i + 1) % len(knots)]
    return np.asarray(manual if manual is not None else (0.0, 0.0), dtype=np.float64)


class Engine:
    """Server-side controller + simulation clock, guarded by a lock.

    The simulation owns a fixed-cadence control update: the target is re-read and
    ``on_target_update`` is called once every ``update_ms`` simulated ticks,
    independent of the browser's frame rate.  The browser only advances sim time
    and chooses the mode; this makes the 8 ms cadence exact and reproducible.
    """

    def __init__(self, config: ClosedLoopConfig, library: ForcingLibrary | None,
                 *, update_ms: int, reach: float, period_ms: float) -> None:
        self._lock = threading.Lock()
        self._ctl = ClosedLoopProDMPController(config=config, forcing_library=library)
        self.update_ms = int(max(1, update_ms))
        self.reach = float(reach)
        self.period_ms = float(period_ms)
        self._knots = np.random.default_rng(1).normal(size=(48, 2)) * self.reach
        self._cursor = np.zeros(2)
        self._sim_t = 0
        self._since_update = 0
        self._cur_target = np.zeros(2)
        self._log_cur: deque = deque(maxlen=LOG_MS)
        self._log_tgt: deque = deque(maxlen=LOG_MS)
        self.mode = self._ctl.renderer_model.receipt.mode
        self.has_library = library is not None

    def reset(self) -> None:
        with self._lock:
            self._ctl.reset()
            self._cursor = np.zeros(2)
            self._sim_t = 0
            self._since_update = 0
            self._cur_target = np.zeros(2)
            self._log_cur.clear()
            self._log_tgt.clear()

    def frame(self, advance_ms: int, mode: str, manual, gain: float,
              apply_deadzone: bool) -> dict:
        """Advance the simulation by ``advance_ms`` ms (1 tick = 1 ms)."""
        n = int(max(0, min(MAX_BATCH_MS, advance_ms)))
        with self._lock:
            self._ctl.config.forcing_gain = float(gain)
            trail = []
            for _ in range(n):
                if self._since_update == 0:
                    self._cur_target = scenario_target(
                        mode, self._sim_t, manual, self.reach, self.period_ms, self._knots
                    )
                    offset = self._cur_target - self._cursor
                    self._ctl.on_target_update(
                        (offset[0], offset[1]), self.update_ms, apply_deadzone=apply_deadzone
                    )
                report = self._ctl.step().astype(np.float64)
                self._cursor = self._cursor + report
                self._log_cur.append(self._cursor.copy())
                self._log_tgt.append(self._cur_target.copy())
                trail.append([float(self._cursor[0]), float(self._cursor[1])])
                self._sim_t += 1
                self._since_update = (self._since_update + 1) % self.update_ms
            err = float(np.hypot(*(self._cur_target - self._cursor)))
            return {
                "cursor": [float(self._cursor[0]), float(self._cursor[1])],
                "target": [float(self._cur_target[0]), float(self._cur_target[1])],
                "trail": trail,
                "pending": int(self._ctl.pending_ticks),
                "ticks": int(self._ctl.ticks),
                "sim_t": int(self._sim_t),
                "update_ms": self.update_ms,
                "err": err,
            }

    def score(self) -> dict:
        with self._lock:
            if len(self._log_cur) < 128:
                return {"samples": len(self._log_cur)}
            cur = np.asarray(self._log_cur)
            tgt = np.asarray(self._log_tgt)
        return evaluate(cur, tgt).as_dict()


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # set on the class before serving

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}', "application/json")
            return
        if self.path == "/api/reset":
            self.engine.reset()
            self._send(200, b'{"ok":true}', "application/json")
            return
        if self.path == "/api/score":
            self._send(200, json.dumps(self.engine.score()).encode(), "application/json")
            return
        if self.path in ("/api/frame", "/api/update"):
            result = self.engine.frame(
                int(payload.get("advance", payload.get("dt", 16))),
                str(payload.get("mode", "manual")),
                payload.get("manual"),
                float(payload.get("gain", 0.45)),
                bool(payload.get("deadzone", True)),
            )
            result["has_library"] = self.engine.has_library
            result["renderer"] = self.engine.mode
            self._send(200, json.dumps(result).encode(), "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--horizon", type=int, default=250, help="ProDMP homing horizon (ms)")
    ap.add_argument("--commit", type=int, default=12, help="committed ticks per replan")
    ap.add_argument("--r-inner", type=float, default=2.0, help="deadzone inner radius (counts)")
    ap.add_argument("--lam", type=float, default=1.0, help="deadzone sigmoid steepness")
    ap.add_argument("--forcing-lib", default="models/continuous/forcing_lib.npz",
                    help="baked B2' library; ignored if the file is absent")
    ap.add_argument("--gain", type=float, default=0.45, help="initial B2' curvature gain")
    ap.add_argument("--update-ms", type=int, default=8,
                    help="target-update cadence in ms (one on_target_update per this many ticks)")
    ap.add_argument("--reach", type=float, default=200.0, help="auto-scenario size (counts)")
    ap.add_argument("--period-ms", type=float, default=2600.0, help="auto-scenario period")
    args = ap.parse_args()

    config = ClosedLoopConfig(
        horizon_ms=args.horizon,
        commit_ms=args.commit,
        r_inner=args.r_inner,
        deadzone_lambda=args.lam,
        forcing_gain=args.gain,
    )
    lib_path = Path(args.forcing_lib)
    library = ForcingLibrary.load(lib_path) if lib_path.is_file() else None
    Handler.engine = Engine(config, library, update_ms=args.update_ms,
                            reach=args.reach, period_ms=args.period_ms)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    shown = _lan_ip() if args.host == "0.0.0.0" else args.host
    print(f"Renderer: {Handler.engine.mode}  |  B2' library: "
          f"{'loaded ' + str(len(library)) + ' shapes' if library else 'none (B1 homing)'}")
    print(f"Live harness on  http://{shown}:{args.port}  (local: http://127.0.0.1:{args.port})")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
