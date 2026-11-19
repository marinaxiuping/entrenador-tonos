"""Tabla de features de F0 por sílaba de Tone Perfect.

F0 con corrección de octava por hablante (extract_corpus_f0), semitonos respecto a la
media geométrica del hablante y las mismas features que el baseline de AISHELL-3
(tonos.features.contour_features) sobre la sílaba entera tras recortar silencios.

Salidas en data/processed/:
  - feats_toneperfect_<method>.parquet: una fila por grabación (baseline).
  - contornos_toneperfect_<method>.npz: contorno de longitud fija por grabación (CNN).

Uso: uv run python scripts/03_features_toneperfect.py --method praat
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tonos.data import build_index
from tonos.f0 import extract_corpus_f0, fixed_length_contour, hz_to_st, speaker_reference_hz
from tonos.features import contour_features

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--method", choices=["pyin", "praat", "crepe"], default="praat")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--contour-points", type=int, default=64, help="longitud del contorno para la CNN")
args = ap.parse_args()

index, _ = build_index(args.root)
if index.empty:
    raise SystemExit("No hay datos en --root")
print(f"Extrayendo F0 ({args.method}) de {len(index)} ficheros...")
f0s = extract_corpus_f0(index.path.tolist(), index.speaker.tolist(), args.method, args.jobs)

refs = {s: speaker_reference_hz([f0s[i] for i in g.index]) for s, g in index.groupby("speaker")}
f0s_st = [hz_to_st(f0, refs[s]) for f0, s in zip(f0s, index.speaker)]
features = pd.DataFrame([contour_features(f0) for f0 in f0s_st])
table = pd.concat([index.drop(columns="path"), features], axis=1)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
out_file = out / f"feats_toneperfect_{args.method}.parquet"
table.to_parquet(out_file, index=False)

# Contornos de longitud fija para la CNN, en el mismo orden de filas que la tabla.
contours = np.stack([fixed_length_contour(f0, n=args.contour_points) for f0 in f0s_st])
np.savez_compressed(out / f"contornos_toneperfect_{args.method}.npz", contours=contours,
                    **{c: table[c].to_numpy() for c in ["syllable", "tone", "speaker", "gender"]})
print(f"{len(table)} sílabas, {table.speaker.nunique()} hablantes, "
      f"sin F0 suficiente: {table['mean'].isna().mean() * 100:.1f}% | guardado en {out_file}")
