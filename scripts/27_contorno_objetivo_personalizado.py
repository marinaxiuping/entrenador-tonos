"""¿Mejora un modelo ligero a la plantilla fija para predecir el contorno objetivo?

Dejando fuera un hablante cada vez, se predice el contorno de cada una de sus sílabas
(64 puntos, semitonos respecto a su media) con tres opciones ajustadas en los otros cinco:
  - plantilla:        contorno medio del tono;
  - duracion:         Ridge por tono sobre la duración de la sílaba;
  - duracion_rango:   Ridge por tono sobre la duración y el rango de F0 del hablante.
El rango se calcula como en la app: percentiles 10-90 del F0 corregido de "ma" en los
cuatro tonos. Con 5 hablantes de entrenamiento, el efecto del rango se estima con solo
5 valores distintos, así que el riesgo de sobreajuste es alto: de ahí la evaluación.

Métrica: RMSE (st) entre contorno predicho y real, media por hablante de test.
Salida: outputs/contorno_objetivo/metrics.json y rmse_por_hablante.png.

Uso: uv run python scripts/27_contorno_objetivo_personalizado.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tonos.data import build_index
from tonos.f0 import fix_octave_jumps, hz_to_st, load, speaker_pitch_range, speaker_reference_hz, waveform_f0

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--contours", default="data/processed/contornos_toneperfect_praat.npz")
ap.add_argument("--features", default="data/processed/feats_toneperfect_praat.parquet")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--out", default="outputs/contorno_objetivo")
args = ap.parse_args()

TONES = [1, 2, 3, 4]
CALIBRATION_SYLLABLE = "ma"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

data = np.load(args.contours, allow_pickle=True)
contours, tones, speakers = data["contours"], data["tone"].astype(int), data["speaker"]
durations = pd.read_parquet(args.features)["dur"].to_numpy()
paths = build_index(args.root)[0].set_index(["syllable", "tone", "speaker"]).path


def calibration_range_st(speaker: str) -> float:
    """Percentiles 10-90 del F0 de la calibración, con el mismo proceso que el perfil de voz."""
    waveforms = [load(paths[(CALIBRATION_SYLLABLE, t, speaker)]) for t in TONES]
    fmin, fmax = speaker_pitch_range([waveform_f0(y, "praat") for y in waveforms])
    bounded = [waveform_f0(y, "praat", fmin=fmin, fmax=fmax) for y in waveforms]
    reference = speaker_reference_hz(bounded)
    voiced = np.concatenate([f[np.isfinite(f)] for f in (fix_octave_jumps(f, reference, fmin, fmax) for f in bounded)])
    low, high = np.percentile(hz_to_st(voiced, reference), [10, 90])
    return float(high - low)


speaker_ranges = {s: calibration_range_st(s) for s in sorted(set(speakers))}
print("rango de calibración (st):", {s: round(r, 2) for s, r in speaker_ranges.items()})
ranges = np.array([speaker_ranges[s] for s in speakers])
FEATURES = {"duracion": np.c_[durations], "duracion_rango": np.c_[durations, ranges]}

rmse = {name: np.full(len(contours), np.nan) for name in ["plantilla", *FEATURES]}
for speaker in speaker_ranges:
    for tone in TONES:
        train = (speakers != speaker) & (tones == tone)
        test = (speakers == speaker) & (tones == tone)
        template = contours[train].mean(axis=0)
        rmse["plantilla"][test] = np.sqrt(((contours[test] - template) ** 2).mean(axis=1))
        for name, x in FEATURES.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha)).fit(x[train], contours[train])
            rmse[name][test] = np.sqrt(((contours[test] - model.predict(x[test])) ** 2).mean(axis=1))

table = pd.DataFrame(rmse).assign(speaker=speakers, tone=tones)
per_speaker = table.groupby("speaker")[list(rmse)].mean()
metrics = {
    "rango_calibracion_st": speaker_ranges,
    "rmse_medio_st": table[list(rmse)].mean().round(4).to_dict(),
    "rmse_por_hablante_st": per_speaker.round(4).to_dict(orient="index"),
    "rmse_por_tono_st": table.groupby("tone")[list(rmse)].mean().round(4).to_dict(orient="index"),
    "hablantes_donde_gana_duracion_rango": int((per_speaker.duracion_rango < per_speaker.plantilla).sum()),
    "alpha": args.alpha,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: metrics[k] for k in ["rmse_medio_st", "rmse_por_hablante_st", "rmse_por_tono_st"]}, indent=2))

ax = per_speaker.plot.bar(figsize=(7, 3.5), rot=0)
ax.set_ylabel("RMSE frente al contorno real (st)")
ax.set_title("Contorno objetivo para un hablante no visto")
plt.tight_layout()
plt.savefig(out / "rmse_por_hablante.png", dpi=120)
print(f"\nResultados en {out}/")
