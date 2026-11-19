"""Exporta al frontend la guía tarareada de cada tono: contorno medio y duración típica.

El contorno es el mismo que el contorno objetivo del modelo de la app (media de Tone
Perfect por tono, en semitonos respecto a la media de cada hablante; script 25), así que
lo que se oye coincide con la línea discontinua de la gráfica. La duración es la mediana
de la sílaba tras recortar silencios: la duración no entra en el modelo porque no
transfiere entre hablantes, pero sí ayuda a imitar (el tono 3 es el más largo).

El frontend la sintetiza a la altura de la voz del usuario (referencia de su perfil).
No reproduce audio de ningún hablante.

Uso: uv run python scripts/06_exportar_guia_tonos.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--contours", default="data/processed/contornos_toneperfect_praat.npz")
ap.add_argument("--features", default="data/processed/feats_toneperfect_praat.parquet")
ap.add_argument("--out", default="frontend/src/data/tone_guide.json")
args = ap.parse_args()

TONES = [1, 2, 3, 4]
data = np.load(args.contours, allow_pickle=True)
contours, tones = data["contours"], data["tone"].astype(int)
durations = pd.read_parquet(args.features, columns=["tone", "dur"]).groupby("tone").dur.median()

guide = {str(t): {"contour_st": np.nanmean(contours[tones == t], axis=0).round(3).tolist(),
                  "duration_s": round(float(durations[t]), 3)} for t in TONES}
Path(args.out).write_text(json.dumps(guide, indent=1), encoding="utf-8")
print("Duración por tono (s):", {t: g["duration_s"] for t, g in guide.items()}, "| guardado en", args.out)
