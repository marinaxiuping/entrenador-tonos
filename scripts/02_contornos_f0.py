"""Extrae F0, normaliza por hablante (semitonos) y dibuja contornos por tono.

Uso:
  uv run python scripts/02_contornos_f0.py --method pyin --n-per-cell 40
  uv run python scripts/02_contornos_f0.py --method praat --n-per-cell 40
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from tonos.data import build_index
from tonos.f0 import (
    extract_corpus_f0,
    extract_f0,
    fixed_length_contour,
    hz_to_st,
    speaker_reference_hz,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--out", default="outputs/contornos")
ap.add_argument("--method", choices=["pyin", "praat"], default="pyin")
ap.add_argument("--n-per-cell", type=int, default=40, help="muestras por hablante x tono (0 = todas)")
ap.add_argument("--n-points", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--no-adapt-range", dest="adapt_range", action="store_false",
                help="una sola pasada con el rango genérico de F0 (sin corrección de octava)")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

df, _ = build_index(args.root)
if df.empty:
    raise SystemExit("No hay datos en --root")
if args.n_per_cell > 0:
    df = (
        df.sample(frac=1, random_state=args.seed)
        .groupby(["speaker", "tone"]).head(args.n_per_cell)
        .reset_index(drop=True)
    )
print(f"Extrayendo F0 ({args.method}) de {len(df)} ficheros...")

if args.adapt_range:
    f0s = extract_corpus_f0(df.path.tolist(), df.speaker.tolist(), args.method, args.jobs)
else:
    extract_f0(df.path.iloc[0], args.method)  # caché numba antes de paralelizar (ver extract_corpus_f0)
    f0s = Parallel(n_jobs=args.jobs)(delayed(extract_f0)(p, args.method) for p in df.path)

# Referencia por hablante (media en semitonos) y contornos normalizados
refs = {s: speaker_reference_hz([f0s[i] for i in g.index]) for s, g in df.groupby("speaker")}
contours = np.stack(
    [fixed_length_contour(hz_to_st(f0, refs[s]), n=args.n_points) for f0, s in zip(f0s, df.speaker)]
)
df["failed"] = np.isnan(contours).all(axis=1)
df["voiced_ratio"] = [np.isfinite(f).mean() if f.size else 0.0 for f in f0s]

print("\nReferencia por hablante (Hz):", {k: round(v, 1) for k, v in refs.items()})
print("\nTasa de fallo (pocos frames sonoros) por tono (%):")
print((df.groupby("tone").failed.mean() * 100).round(2))
print("\nRatio de frames sonoros por tono (mediana):")
print(df.groupby("tone").voiced_ratio.median().round(3))

np.savez_compressed(out / f"contornos_{args.method}.npz", contours=contours, refs=refs)
df.to_csv(out / f"meta_{args.method}.csv", index=False)

# --- Figura 1: contornos individuales + media por tono ---
t = np.linspace(0, 1, args.n_points)
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for tone, ax in zip(range(1, 5), axes):
    m = (df.tone == tone).to_numpy() & ~df.failed.to_numpy()
    for c, g in zip(contours[m], df.gender[m]):
        ax.plot(t, c, color="tab:red" if g == "F" else "tab:blue", alpha=0.08, lw=0.8)
    ax.plot(t, np.nanmean(contours[m], axis=0), color="k", lw=2.5)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_title(f"Tono {tone} (n={m.sum()})")
    ax.set_xlabel("tiempo normalizado")
axes[0].set_ylabel("semitonos respecto a la media del hablante")
fig.suptitle(f"Contornos de F0 – {args.method} (rojo=F, azul=M, negro=media)")
fig.tight_layout()
fig.savefig(out / f"contornos_por_tono_{args.method}.png", dpi=130)

# --- Figura 2: media por tono y hablante ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for tone, ax in zip(range(1, 5), axes):
    for spk in sorted(df.speaker.unique()):
        m = ((df.tone == tone) & (df.speaker == spk) & ~df.failed).to_numpy()
        if m.any():
            ax.plot(t, np.nanmean(contours[m], axis=0), label=spk, lw=1.8)
    ax.set_title(f"Tono {tone}")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
axes[0].set_ylabel("semitonos")
axes[-1].legend(fontsize=8)
fig.suptitle(f"Media por hablante – {args.method}")
fig.tight_layout()
fig.savefig(out / f"contornos_por_hablante_{args.method}.png", dpi=130)
print(f"\nFiguras y datos en {out}/")
