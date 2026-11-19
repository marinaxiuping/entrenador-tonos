"""Features de F0 por sílaba para el baseline (gradient boosting)."""

from __future__ import annotations

import numpy as np

from tonos.f0 import HOP_S, fixed_length_contour

N_PTS = 10


def frames(f0_st: np.ndarray, start: float, end: float) -> np.ndarray:
    a, b = int(round(start / HOP_S)), int(round(end / HOP_S))
    return f0_st[a:max(b, a + 1)]


def contour_features(seg: np.ndarray, prefix: str = "", n_pts: int = N_PTS) -> dict[str, float]:
    """Features de un tramo de F0 en semitonos (NaN = sordo). NaN si no hay F0 suficiente."""
    f = {f"{prefix}dur": seg.size * HOP_S,
         f"{prefix}voiced": float(np.isfinite(seg).mean()) if seg.size else 0.0}
    c = fixed_length_contour(seg, n=n_pts, min_voiced=3)
    for i, v in enumerate(c):
        f[f"{prefix}c{i}"] = v
    names = ["mean", "std", "min", "max", "range", "argmin", "argmax",
             "slope", "p2_a", "p2_b", "p2_c", "delta", "delta_1h", "delta_2h"]
    if np.isnan(c).all():
        f.update({f"{prefix}{k}": np.nan for k in names})
        return f
    t = np.linspace(0, 1, n_pts)
    a, b, c0 = np.polyfit(t, c, 2)
    h = n_pts // 2
    vals = [c.mean(), c.std(), c.min(), c.max(), c.max() - c.min(),
            c.argmin() / (n_pts - 1), c.argmax() / (n_pts - 1),
            np.polyfit(t, c, 1)[0], a, b, c0, c[-1] - c[0], c[h] - c[0], c[-1] - c[h]]
    f.update({f"{prefix}{k}": float(v) for k, v in zip(names, vals)})
    return f
