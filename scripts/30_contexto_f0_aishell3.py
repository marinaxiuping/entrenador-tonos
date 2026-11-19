"""Modelos de contexto sobre el F0 de AISHELL-3: ¿cuánto del techo de la CNN por sílaba es falta de contexto?

Mismas sílabas evaluadas, folds por hablante y calibración sin fuga que el script 29, para
que las cifras se comparen directamente con su condición "en dominio, completo":
  - ventana: CNN sobre los contornos de la sílaba anterior, la actual y la siguiente.
  - frase: CNN por sílaba + BiGRU sobre toda la frase (tonos.context.UtteranceTagger).
Las sílabas neutras y las que no tienen F0 no se evalúan, pero sí sirven de contexto.

Salidas en outputs/escalera_dominio/aishell3_contexto/: metrics.json,
predicciones_<modelo>.parquet y confusion_<modelo>.png.

Uso: uv run python scripts/30_contexto_f0_aishell3.py
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tonos.aishell3 import TONES, load_syllables, speaker_fold_ids
from tonos.calibration import calibrate_by_group
from tonos.cnn import TrainingConfig, predict_logits, train_cnn
from tonos.context import TaggerConfig, predict_tagger_logits, syllable_inputs, train_tagger, window_inputs
from tonos.evaluation import domain_metrics, prediction_table, save_confusion

ap = argparse.ArgumentParser()
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train-seed", type=int,
                help="semilla solo de la inicialización y el orden de entrenamiento de las redes; por defecto, --seed. Cambiarla repite el experimento con los mismos folds y datos")
ap.add_argument("--out", default="outputs/escalera_dominio/aishell3_contexto")
args = ap.parse_args()
train_seed = args.seed if args.train_seed is None else args.train_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

syllables = load_syllables(args.syllables, args.contours)
table, labels, evaluated = syllables.table, syllables.labels, syllables.evaluated
fold_of = speaker_fold_ids(table, args.folds, args.seed)
bounds = syllables.utterance_bounds()
utterance_fold = fold_of[bounds[:, 0]]
print(f"{evaluated.sum()} sílabas evaluadas de {len(table)} ({len(bounds)} frases) | dispositivo {device}")


def window_logits() -> np.ndarray:
    x = window_inputs(syllables.contours, table.utt.to_numpy())
    config = TrainingConfig()
    logits = np.zeros((len(table), len(TONES)), dtype=np.float32)
    for fold in range(args.folds):
        train, test = evaluated & (fold_of != fold), evaluated & (fold_of == fold)
        model = train_cnn(x[train], labels[train], len(TONES), config, device, train_seed)
        logits[test] = predict_logits(model, x[test], device)
        print(f"  fold {fold}: accuracy {(logits[test].argmax(1) == labels[test]).mean():.3f}")
    return logits


def utterance_logits() -> np.ndarray:
    x = syllable_inputs(syllables.contours)
    config = TaggerConfig()
    logits = np.zeros((len(table), len(TONES)), dtype=np.float32)
    for fold in range(args.folds):
        test_bounds = bounds[utterance_fold == fold]
        model = train_tagger(x, labels, bounds[utterance_fold != fold], len(TONES), config, device, train_seed)
        rows = np.concatenate([np.arange(start, end) for start, end in test_bounds])
        logits[rows] = predict_tagger_logits(model, x, test_bounds, device)
        test = evaluated & (fold_of == fold)
        print(f"  fold {fold}: accuracy {(logits[test].argmax(1) == labels[test]).mean():.3f}")
    return logits


evaluated_table = table[evaluated].reset_index(drop=True)
evaluated_labels, evaluated_folds = labels[evaluated], fold_of[evaluated]
results = {"n_silabas": int(evaluated.sum()), "seed": args.seed, "train_seed": train_seed, "folds": args.folds,
           "entrenamiento": {"ventana": asdict(TrainingConfig()), "frase": asdict(TaggerConfig())},
           "modelos": {}}
for name, compute_logits in [("ventana", window_logits), ("frase", utterance_logits)]:
    print(f"\nContexto: {name}")
    logits = compute_logits()[evaluated]
    proba = calibrate_by_group(logits, evaluated_labels, evaluated_folds)
    metrics = domain_metrics(proba, evaluated_labels, evaluated_table, TONES)
    results["modelos"][name] = metrics
    predictions = prediction_table(evaluated_table, proba, TONES)
    predictions.to_parquet(out / f"predicciones_{name}.parquet", index=False)
    save_confusion(predictions.tone.to_numpy(), predictions.pred.to_numpy(), TONES,
                   f"AISHELL-3 · contexto {name} · acc {metrics['global']['accuracy']:.3f}",
                   out / f"confusion_{name}.png")
    mean, std = metrics["accuracy_hablante_media_std"]
    print(f"{name}: accuracy {metrics['global']['accuracy']:.3f} (hablantes {mean:.3f} ± {std:.3f}) | "
          f"macro-F1 {metrics['global']['macro_f1']:.3f} | ECE {metrics['ece']:.3f} | "
          f"recall {metrics['global']['recall_por_tono']}")

(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nResultados en {out}/")
