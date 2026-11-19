"""Curva de aprendizaje por número de hablantes de entrenamiento en AISHELL-3.

Pregunta: ¿más datos mejorarían los modelos de F0? El script 29 ya mostró que multiplicar
por 14 las sílabas (de 9.839 a ~150.000, con los mismos 32 hablantes) solo añade 2 puntos
a la CNN por sílaba. Aquí se varía lo otro, la variedad de voces: en cada fold se entrena
con 4, 8, 16 y 32 hablantes de los otros folds (mitad mujeres y mitad hombres), con todas
sus sílabas, y se evalúa en los hablantes del fold. Los subconjuntos son anidados (los 8
incluyen a los 4, y así) para que las diferencias entre puntos no las ponga el azar de
qué hablantes tocan.

Modelos: CNN por sílaba (script 29) y CNN + BiGRU sobre la frase (script 30), con su misma
configuración de épocas fijas. Con menos datos hay menos pasos de entrenamiento: es lo
mismo que hizo el control de tamaño del script 29, y así los puntos son comparables.

Salidas en outputs/escalera_dominio/curva_hablantes/: curva.csv (una fila por fold,
número de hablantes y modelo), resumen.csv y curva_hablantes.png.

Uso: uv run python scripts/37_curva_hablantes_aishell3.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tonos.aishell3 import TONES, load_syllables, speaker_fold_ids
from tonos.cnn import TrainingConfig, contour_inputs, predict_logits, train_cnn
from tonos.context import TaggerConfig, predict_tagger_logits, syllable_inputs, train_tagger

ap = argparse.ArgumentParser()
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--speaker-counts", type=int, nargs="+", default=[4, 8, 16, 32])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/curva_hablantes")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

syllables = load_syllables(args.syllables, args.contours)
table, labels, evaluated = syllables.table, syllables.labels, syllables.evaluated
speaker_of = table.speaker.to_numpy()
fold_of = speaker_fold_ids(table, args.folds, args.seed)
bounds = syllables.utterance_bounds()
x_syllable = contour_inputs(syllables.contours)
x_utterance = syllable_inputs(syllables.contours)
genders = table.drop_duplicates("speaker").set_index("speaker").gender
print(f"{evaluated.sum()} sílabas evaluables de {len(genders)} hablantes | dispositivo {device}", flush=True)


def nested_speaker_order(pool: list[str], rng: np.random.Generator) -> list[str]:
    """Orden de los hablantes que alterna géneros: los primeros n están siempre equilibrados."""
    by_gender = [list(rng.permutation([s for s in pool if genders[s] == g])) for g in sorted(genders.unique())]
    return [s for pair in zip(*by_gender) for s in pair]


def accuracy(logits: np.ndarray, rows: np.ndarray) -> float:
    return float((logits.argmax(1) == labels[rows]).mean())


results = []
for fold in range(args.folds):
    rng = np.random.default_rng(args.seed + fold)
    test = evaluated & (fold_of == fold)
    test_bounds = bounds[fold_of[bounds[:, 0]] == fold]
    test_rows = np.concatenate([np.arange(start, end) for start, end in test_bounds])
    order = nested_speaker_order(sorted(set(speaker_of[fold_of != fold])), rng)
    for n_speakers in args.speaker_counts:
        chosen = np.isin(speaker_of, order[:n_speakers])
        train = evaluated & chosen
        cnn = train_cnn(x_syllable[train], labels[train], len(TONES), TrainingConfig(), device, args.seed)
        cnn_accuracy = accuracy(predict_logits(cnn, x_syllable[test], device), np.flatnonzero(test))
        tagger = train_tagger(x_utterance, labels, bounds[chosen[bounds[:, 0]]], len(TONES), TaggerConfig(),
                              device, args.seed)
        tagger_logits = np.zeros((len(table), len(TONES)), dtype=np.float32)
        tagger_logits[test_rows] = predict_tagger_logits(tagger, x_utterance, test_bounds, device)
        tagger_accuracy = accuracy(tagger_logits[test], np.flatnonzero(test))
        for model, value in [("cnn_silaba", cnn_accuracy), ("frase", tagger_accuracy)]:
            results.append({"fold": fold, "hablantes": n_speakers, "silabas_entrenamiento": int(train.sum()),
                            "modelo": model, "accuracy": round(value, 4)})
        print(f"fold {fold} | {n_speakers:2d} hablantes ({train.sum()} sílabas): "
              f"CNN {cnn_accuracy:.3f} | frase {tagger_accuracy:.3f}", flush=True)
        pd.DataFrame(results).to_csv(out / "curva.csv", index=False)

curve = pd.DataFrame(results)
summary = (curve.groupby(["modelo", "hablantes"])
           .agg(silabas_entrenamiento=("silabas_entrenamiento", "mean"), accuracy_media=("accuracy", "mean"),
                accuracy_desv=("accuracy", "std"))
           .round(4).reset_index())
summary.to_csv(out / "resumen.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 3.6))
for model, label in [("cnn_silaba", "CNN por sílaba"), ("frase", "CNN + BiGRU")]:
    rows = summary[summary.modelo == model]
    ax.errorbar(rows.hablantes, rows.accuracy_media, yerr=rows.accuracy_desv, marker="o", capsize=3, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(args.speaker_counts, [str(n) for n in args.speaker_counts])
ax.set_xlabel("hablantes de entrenamiento (todas sus sílabas)")
ax.set_ylabel("accuracy con hablantes no vistos")
ax.set_title("Curva de aprendizaje por número de hablantes (media ± desv. entre folds)")
ax.legend()
fig.tight_layout()
fig.savefig(out / "curva_hablantes.png", dpi=120)
print("\n" + summary.to_string(index=False))
print(f"\nResultados en {out}/")
