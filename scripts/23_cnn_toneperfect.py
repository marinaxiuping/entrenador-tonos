"""CNN 1D sobre el contorno de F0 de Tone Perfect, dejando fuera un hablante por fold.

Mismo esquema de evaluación que el baseline (21_baseline_toneperfect.py) para que las
cifras sean comparables. El contorno tiene longitud fija, así que el modelo no ve la
duración de la sílaba, igual que el conjunto `base` del baseline.

Salidas en outputs/cnn_toneperfect/<method>/: metrics.json, oof_cnn.parquet (mismo formato
que el baseline: se calibra con 22_calibrar_toneperfect.py --baseline-dir
outputs/cnn_toneperfect --set cnn) y confusion_cnn.png.

Uso: uv run python scripts/23_cnn_toneperfect.py --method praat
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tonos.cnn import TrainingConfig, contour_inputs, predict_proba, train_cnn
from tonos.evaluation import save_confusion, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="praat")
ap.add_argument("--contours", default=None)
ap.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/cnn_toneperfect")
args = ap.parse_args()

data = np.load(args.contours or f"data/processed/contornos_toneperfect_{args.method}.npz", allow_pickle=True)
meta = pd.DataFrame({c: data[c] for c in ["syllable", "tone", "speaker", "gender"]})
out = Path(args.out) / args.method
out.mkdir(parents=True, exist_ok=True)

TONES = [1, 2, 3, 4]
SPEAKERS = sorted(meta.speaker.unique())
y = meta.tone.to_numpy()
labels = meta.tone.map({t: i for i, t in enumerate(TONES)}).to_numpy()
x = contour_inputs(data["contours"])
# Sílabas sin F0 suficiente: no entrenan y se predicen con probabilidad uniforme.
has_contour = np.isfinite(x).all(axis=(1, 2))
config = TrainingConfig(epochs=args.epochs)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(meta)} sílabas ({(~has_contour).sum()} sin contorno), dispositivo {device}")

proba = np.full((len(meta), len(TONES)), 1 / len(TONES))
for speaker in SPEAKERS:
    test = (meta.speaker == speaker).to_numpy()
    train = ~test & has_contour
    model = train_cnn(x[train], labels[train], len(TONES), config, device, args.seed)
    predictable = test & has_contour
    proba[predictable] = predict_proba(model, x[predictable], device)
    print(f"  {speaker}: accuracy {(proba[test].argmax(1) == labels[test]).mean():.3f}")
pred = np.array(TONES)[proba.argmax(1)]

per_speaker = {s: summarize(y[meta.speaker == s], pred[meta.speaker == s], TONES) for s in SPEAKERS}
speaker_acc = [r["accuracy"] for r in per_speaker.values()]
results = {
    "global": summarize(y, pred, TONES),
    "accuracy_hablante_media_std": [float(np.mean(speaker_acc)), float(np.std(speaker_acc))],
    "por_hablante": per_speaker,
    "por_genero": {g: summarize(y[meta.gender == g], pred[meta.gender == g], TONES)
                   for g in sorted(meta.gender.unique())},
    "entrenamiento": {**asdict(config), "seed": args.seed, "n_puntos": x.shape[-1]},
}
g = results["global"]
print(f"\n=== {args.method} | cnn ===")
print(f"accuracy {g['accuracy']:.3f} (hablantes {np.mean(speaker_acc):.3f} ± {np.std(speaker_acc):.3f}) | "
      f"macro-F1 {g['macro_f1']:.3f}")
print("recall por tono:", g["recall_por_tono"])

oof = meta.copy()
for i, t in enumerate(TONES):
    oof[f"p{t}"] = proba[:, i]
oof.to_parquet(out / "oof_cnn.parquet", index=False)
save_confusion(y, pred, TONES, f"{args.method} · cnn · acc {g['accuracy']:.3f}", out / "confusion_cnn.png")
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nResultados en {out}/")
