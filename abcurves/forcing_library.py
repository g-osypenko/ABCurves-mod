"""Baked ProDMP forcing-weight library (decision B2').

The macro acceleration *shape* (human-like curvature along a goal-directed
move) is learned by the continuous ``motor.onnx``.  Running that motor live in
the control loop would drag back the whole ``_continuous`` finite-difference /
FSM path (the source of the instability we are removing), so instead we harvest
its behaviour **offline once** into a small, dimensionless library of forcing
shapes (see ``tools/bake_forcing_library.py``) and sample from it at runtime.

Each stored shape is *canonical*: expressed in a frame where the net move points
along +x and spans unit distance.  :meth:`orient` rotates and scales a shape to
the live goal, so one compact library generalizes across directions and sizes.
A library file is a ``.npz`` with ``shapes`` ``[K, n_basis, 2]`` and ``n_basis``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ForcingLibrary:
    """A sample-able set of canonical ProDMP forcing shapes."""

    def __init__(self, shapes: np.ndarray, n_basis: int) -> None:
        shapes = np.asarray(shapes, dtype=np.float64)
        if shapes.ndim != 3 or shapes.shape[1:] != (int(n_basis), 2):
            raise ValueError(f"shapes must be [K, {n_basis}, 2], got {shapes.shape}")
        if len(shapes) == 0:
            raise ValueError("forcing library is empty")
        self.shapes = shapes
        self.n_basis = int(n_basis)

    @classmethod
    def load(cls, path: str | Path) -> "ForcingLibrary":
        with np.load(Path(path)) as archive:
            return cls(archive["shapes"], int(archive["n_basis"]))

    def __len__(self) -> int:
        return len(self.shapes)

    def pick(self, rng: np.random.Generator) -> np.ndarray:
        """Return one canonical shape ``[n_basis, 2]`` (sticky per movement)."""
        return self.shapes[int(rng.integers(0, len(self.shapes)))].copy()

    def orient(self, shape: np.ndarray, goal) -> np.ndarray:
        """Rotate a canonical shape onto ``goal`` and scale by its magnitude.

        Returns forcing rows ``[n_basis, 2]`` ready to drop into ``w_g[:-1]``.
        """
        g = np.asarray(goal, dtype=np.float64).reshape(2)
        dist = float(np.hypot(g[0], g[1]))
        if dist < 1e-9:
            return np.zeros_like(shape)
        c, s = g[0] / dist, g[1] / dist  # rotation mapping +x -> goal direction
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (shape @ rot.T) * dist
