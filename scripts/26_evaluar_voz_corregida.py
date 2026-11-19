"""Evaluación objetiva de la voz corregida (resíntesis con WORLD) sobre Tone Perfect.

Para cada hablante, sin que intervenga en nada de lo que se usa para evaluarlo:
  - CNN entrenada con los otros cinco (mismo esquema que 23_cnn_toneperfect.py);
  - plantilla objetivo de cada tono = contorno medio de los otros cinco;
  - perfil de voz calculado como en la app, con "ma" en los cuatro tonos (por eso "ma" no
    se usa como sílaba de prueba).
Cada sílaba de prueba se resintetiza hacia los cuatro tonos (incluido el suyo, como
control) y se mide si la CNN reconoce el tono objetivo y cuánto se aleja el F0 resultante
de la plantilla. Es una medida objetiva y circular (un clasificador juzga a otro sistema);
no sustituye a escuchar los ejemplos que se guardan en ejemplos/.

Salidas en outputs/voz_corregida/: metrics.json, resultados.parquet,
accuracy_origen_objetivo.png, error_f0_por_desplazamiento.png y ejemplos/*.wav.

Uso: uv run python scripts/26_evaluar_voz_corregida.py --per-cell 25
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from joblib import Parallel, delayed

from tonos.cnn import TrainingConfig, contour_inputs, predict_proba, train_cnn
from tonos.data import build_index
from tonos.f0 import SR, load
from tonos.inference import NoVoiceDetected, SpeakerProfile, build_speaker_profile, speaker_contour
from tonos.resynthesis import resynthesize_with_contour

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--contours", default="data/processed/contornos_toneperfect_praat.npz")
ap.add_argument("--per-cell", type=int, default=25, help="sílabas de prueba por hablante y tono de origen")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/voz_corregida")
args = ap.parse_args()

TONES = [1, 2, 3, 4]
F0_METHOD = "praat"
CALIBRATION_SYLLABLE = "ma"
out = Path(args.out)
(out / "ejemplos").mkdir(parents=True, exist_ok=True)

data = np.load(args.contours, allow_pickle=True)
contours, tones, speakers = data["contours"], data["tone"].astype(int), data["speaker"]
n_points = contours.shape[1]
paths = build_index(args.root)[0].set_index(["syllable", "tone", "speaker"]).path
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(args.seed)


def resynthesize_all_targets(path: str, profile: SpeakerProfile,
                             templates: dict[int, np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Contorno original y contornos de la sílaba resintetizada hacia cada tono (NaN si falla)."""
    waveform = load(path)
    missing = np.full(n_points, np.nan)

    def contour_of(audio: np.ndarray) -> np.ndarray:
        try:
            return speaker_contour(audio, profile, F0_METHOD, n_points)
        except NoVoiceDetected:
            return missing

    resynthesized = [
        resynthesize_with_contour(waveform, SR, templates[t], profile.reference_hz, profile.fmin, profile.fmax)
        for t in TONES
    ]
    return contour_of(waveform), [contour_of(audio) for audio in resynthesized]


records = []
for speaker in sorted(set(speakers)):
    train = speakers != speaker
    labels = np.searchsorted(TONES, tones)
    model = train_cnn(contour_inputs(contours[train]), labels[train], len(TONES), TrainingConfig(), device, args.seed)
    templates = {t: np.nanmean(contours[train & (tones == t)], axis=0) for t in TONES}
    profile = build_speaker_profile([load(paths[(CALIBRATION_SYLLABLE, t, speaker)]) for t in TONES], F0_METHOD)

    candidates = paths.xs(speaker, level="speaker").drop(CALIBRATION_SYLLABLE, level="syllable", errors="ignore")
    sample = pd.concat([
        candidates.xs(t, level="tone", drop_level=False).sample(args.per_cell, random_state=int(rng.integers(1 << 31)))
        for t in TONES
    ])
    results = Parallel(n_jobs=args.jobs)(
        delayed(resynthesize_all_targets)(path, profile, templates) for path in sample
    )
    originals = np.stack([original for original, _ in results])
    converted = np.stack([c for _, per_target in results for c in per_target])
    valid = np.isfinite(converted).all(axis=1)
    proba = np.full((len(converted), len(TONES)), np.nan)
    proba[valid] = predict_proba(model, contour_inputs(converted[valid]), device)
    original_pred = np.array(TONES)[predict_proba(model, contour_inputs(np.nan_to_num(originals)), device).argmax(1)]

    for i, ((syllable, source_tone), original) in enumerate(zip(sample.index, originals)):
        for j, target_tone in enumerate(TONES):
            k = i * len(TONES) + j
            records.append({
                "speaker": speaker, "syllable": syllable, "source_tone": source_tone, "target_tone": target_tone,
                "predicted": int(TONES[int(np.nanargmax(proba[k]))]) if valid[k] else None,
                "f0_error_st": float(np.sqrt(np.mean((converted[k] - templates[target_tone]) ** 2))) if valid[k] else None,
                "shift_st": float(np.mean(np.abs(templates[target_tone] - original))),
                "original_predicted": int(original_pred[i]),
            })

    # Ejemplos para escuchar: la primera sílaba de T1 de este hablante hacia los cuatro tonos.
    (syllable, source_tone), path = next(iter(sample.items()))
    waveform = load(path)
    sf.write(out / "ejemplos" / f"{speaker}_{syllable}{source_tone}_original.wav", waveform, SR)
    for t in TONES:
        audio = resynthesize_with_contour(waveform, SR, templates[t], profile.reference_hz, profile.fmin, profile.fmax)
        sf.write(out / "ejemplos" / f"{speaker}_{syllable}{source_tone}_a_T{t}.wav", audio, SR)
    print(f"{speaker}: {len(sample)} sílabas x {len(TONES)} tonos objetivo", flush=True)

results = pd.DataFrame(records)
results.to_parquet(out / "resultados.parquet", index=False)
results["correct"] = results.predicted == results.target_tone
changed = results[results.source_tone != results.target_tone]
same = results[results.source_tone == results.target_tone]
shift_bins = pd.cut(changed.shift_st, [0, 3, 6, 9, np.inf])

metrics = {
    "n_resintesis": len(results),
    "fallos_sin_voz": int(results.predicted.isna().sum()),
    "accuracy_originales": float((same.original_predicted == same.source_tone).mean()),
    "accuracy_control_mismo_tono": float(same.correct.mean()),
    "accuracy_tono_cambiado": float(changed.correct.mean()),
    "accuracy_tono_cambiado_por_objetivo": changed.groupby("target_tone").correct.mean().round(4).to_dict(),
    "accuracy_tono_cambiado_por_hablante": changed.groupby("speaker").correct.mean().round(4).to_dict(),
    "error_f0_st_mediana": float(results.f0_error_st.median()),
    "error_f0_st_mediana_por_desplazamiento": changed.groupby(shift_bins, observed=True).f0_error_st.median().round(3)
    .rename(lambda b: str(b)).to_dict(),
    "accuracy_por_desplazamiento": changed.groupby(shift_bins, observed=True).correct.mean().round(4)
    .rename(lambda b: str(b)).to_dict(),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))

matrix = results.pivot_table(index="source_tone", columns="target_tone", values="correct", aggfunc="mean")
fig, ax = plt.subplots(figsize=(4.4, 3.9))
ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
for i in range(len(TONES)):
    for j in range(len(TONES)):
        value = matrix.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > .5 else "black", fontsize=9)
ax.set_xticks(range(len(TONES)), [f"T{t}" for t in TONES])
ax.set_yticks(range(len(TONES)), [f"T{t}" for t in TONES])
ax.set_xlabel("tono objetivo de la resíntesis")
ax.set_ylabel("tono original")
ax.set_title("La CNN reconoce el tono objetivo")
fig.tight_layout()
fig.savefig(out / "accuracy_origen_objetivo.png", dpi=120)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.scatter(changed.shift_st, changed.f0_error_st, s=4, alpha=0.3)
ax.set_xlabel("desplazamiento medio de F0 pedido (st)")
ax.set_ylabel("error de F0 tras resintetizar (st)")
ax.set_title("Error de F0 según el desplazamiento")
fig.tight_layout()
fig.savefig(out / "error_f0_por_desplazamiento.png", dpi=120)
print(f"\nResultados en {out}/")
