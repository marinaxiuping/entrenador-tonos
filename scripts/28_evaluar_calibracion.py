"""¿Con qué sílabas calibrar el perfil de voz?

El perfil de la app (referencia y rango de F0) sale de grabaciones de calibración: cada
sílaba del conjunto en los cuatro tonos. Para cada conjunto candidato y cada hablante de
Tone Perfect se mide:
  - la desviación de su referencia respecto a la media de todo su corpus (en st);
  - la accuracy de una CNN entrenada sin ese hablante sobre sus sílabas normalizadas con
    el perfil, excluidas las de calibración de todos los conjuntos para comparar igual.

Salida: outputs/calibracion_perfil/metrics.json.

Uso: uv run python scripts/28_evaluar_calibracion.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from tonos.cnn import TrainingConfig, contour_inputs, predict_proba, train_cnn
from tonos.data import build_index
from tonos.f0 import load
from tonos.inference import NoVoiceDetected, SpeakerProfile, build_speaker_profile, speaker_contour

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--contours", default="data/processed/contornos_toneperfect_praat.npz")
ap.add_argument("--reference-contours", default="outputs/contornos/contornos_praat.npz",
                help="referencia por hablante con todo su corpus (salida de 02_contornos_f0.py)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/calibracion_perfil")
args = ap.parse_args()

TONES = [1, 2, 3, 4]
F0_METHOD = "praat"
CANDIDATE_SETS = {
    "ma": ["ma"],
    "ma+yi": ["ma", "yi"],
    "ma+yi+wu": ["ma", "yi", "wu"],
    "ma+yi+wu+li": ["ma", "yi", "wu", "li"],
}
calibration_syllables = sorted({s for syllables in CANDIDATE_SETS.values() for s in syllables})

data = np.load(args.contours, allow_pickle=True)
contours, tones, speakers, syllables = data["contours"], data["tone"].astype(int), data["speaker"], data["syllable"]
n_points = contours.shape[1]
corpus_references = np.load(args.reference_contours, allow_pickle=True)["refs"].item()
index = build_index(args.root)[0]
paths = index.set_index(["syllable", "tone", "speaker"]).path
device = "cuda" if torch.cuda.is_available() else "cpu"


def contour_or_nan(path: str, profile: SpeakerProfile) -> np.ndarray:
    try:
        return speaker_contour(load(path), profile, F0_METHOD, n_points)
    except NoVoiceDetected:
        return np.full(n_points, np.nan)


results: dict[str, dict] = {name: {"desviacion_st": {}, "accuracy": {}} for name in CANDIDATE_SETS}
for speaker in sorted(set(speakers)):
    train = speakers != speaker
    model = train_cnn(contour_inputs(contours[train]), np.searchsorted(TONES, tones[train]), len(TONES),
                      TrainingConfig(), device, args.seed)
    test_rows = index[(index.speaker == speaker) & ~index.syllable.isin(calibration_syllables)]
    for name, calibration in CANDIDATE_SETS.items():
        profile = build_speaker_profile([load(paths[(s, t, speaker)]) for s in calibration for t in TONES], F0_METHOD)
        test_contours = np.stack(Parallel(n_jobs=args.jobs)(
            delayed(contour_or_nan)(path, profile) for path in test_rows.path))
        usable = np.isfinite(test_contours).all(axis=1)
        predicted = np.array(TONES)[predict_proba(model, contour_inputs(test_contours[usable]), device).argmax(1)]
        results[name]["desviacion_st"][speaker] = round(float(12 * np.log2(profile.reference_hz / corpus_references[speaker])), 3)
        results[name]["accuracy"][speaker] = round(float((predicted == test_rows.tone.to_numpy()[usable]).mean()), 4)
    print(speaker, {name: (r["desviacion_st"][speaker], r["accuracy"][speaker]) for name, r in results.items()}, flush=True)

summary = {
    name: {
        "grabaciones": 4 * len(CANDIDATE_SETS[name]),
        "desviacion_abs_media_st": round(float(np.mean(np.abs(list(r["desviacion_st"].values())))), 3),
        "desviacion_abs_max_st": round(float(np.max(np.abs(list(r["desviacion_st"].values())))), 3),
        "accuracy_media": round(float(np.mean(list(r["accuracy"].values()))), 4),
        "accuracy_minima": round(float(np.min(list(r["accuracy"].values()))), 4),
        **r,
    }
    for name, r in results.items()
}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
for name, s in summary.items():
    print(f"{name:12s} {s['grabaciones']:2d} grabaciones | desviación media {s['desviacion_abs_media_st']:.2f} st "
          f"(máx {s['desviacion_abs_max_st']:.2f}) | accuracy {s['accuracy_media']:.4f} (mín {s['accuracy_minima']:.4f})")
print(f"\nResultados en {out}/")
