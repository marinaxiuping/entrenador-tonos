"""Embeddings congelados + clasificador lineal (regresión logística) sobre Tone Perfect.

Mismo esquema que el baseline: cada fold deja fuera un hablante. La capa del modelo se
elige dentro de cada fold, sin ver al hablante de test: con los cinco de entrenamiento se
hace a su vez una validación dejando fuera uno, y se queda la capa con mejor accuracy media.

Además se guarda la accuracy por capa evaluada directamente sobre el hablante de test
(accuracy_por_capa.png). Es solo diagnóstico: quedarse con su máximo sería optimista.

Salidas en outputs/lineal_toneperfect/<modelo>/: metrics.json, oof_lineal.parquet (se
calibra con 22_calibrar_toneperfect.py), confusion_lineal.png, accuracy_por_capa.png.

Uso: uv run python scripts/24_lineal_embeddings_toneperfect.py
"""

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tonos.evaluation import save_confusion, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--embeddings", default="data/interim/embeddings/toneperfect_chinese-hubert-base.npz")
ap.add_argument("--C", type=float, default=1.0, help="inversa de la regularización L2")
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/lineal_toneperfect")
args = ap.parse_args()

data = np.load(args.embeddings, allow_pickle=True)
embeddings = data["embeddings"]
meta = pd.DataFrame({c: data[c] for c in ["syllable", "tone", "speaker", "gender"]})
model_name = str(data["model"]).split("/")[-1]
out = Path(args.out) / model_name
out.mkdir(parents=True, exist_ok=True)

TONES = [1, 2, 3, 4]
SPEAKERS = sorted(meta.speaker.unique())
LAYERS = range(embeddings.shape[1])
y = meta.tone.to_numpy()
speaker_of = meta.speaker.to_numpy()


def fit_probe(x: np.ndarray, labels: np.ndarray, C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(x, labels)


def held_out_accuracy(x: np.ndarray, labels: np.ndarray, speakers: np.ndarray,
                      excluded: list[str], test_speaker: str, C: float) -> float:
    """Entrena con todos los hablantes salvo ``excluded`` y evalúa en ``test_speaker``."""
    train, test = ~np.isin(speakers, excluded), speakers == test_speaker
    probe = fit_probe(x[train], labels[train], C)
    return float((probe.predict(x[test]) == labels[test]).mean())


# Un array float32 por capa, pasado como argumento para que joblib lo comparta por memmap
# en vez de copiarlo a cada tarea.
layer_inputs = [np.ascontiguousarray(embeddings[:, layer], dtype=np.float32) for layer in LAYERS]
# Validación interna: (hablante de test, hablante de validación, capa) → accuracy.
inner_tasks = [(t, v, layer) for t, v, layer in product(SPEAKERS, SPEAKERS, LAYERS) if t != v]
outer_tasks = list(product(SPEAKERS, LAYERS))
print(f"{len(inner_tasks)} ajustes de validación interna + {len(outer_tasks)} de diagnóstico...", flush=True)
scores = Parallel(n_jobs=args.jobs, verbose=5)(
    [delayed(held_out_accuracy)(layer_inputs[layer], y, speaker_of, [t, v], v, args.C)
     for t, v, layer in inner_tasks]
    + [delayed(held_out_accuracy)(layer_inputs[layer], y, speaker_of, [t], t, args.C)
       for t, layer in outer_tasks]
)
inner = pd.DataFrame(inner_tasks, columns=["test", "val", "layer"]).assign(accuracy=scores[:len(inner_tasks)])
diagnostic = pd.DataFrame(outer_tasks, columns=["test", "layer"]).assign(accuracy=scores[len(inner_tasks):])

chosen_layer = inner.groupby(["test", "layer"]).accuracy.mean().groupby("test").idxmax().map(lambda k: k[1])
proba = np.zeros((len(meta), len(TONES)))
for speaker in SPEAKERS:
    test = speaker_of == speaker
    x = layer_inputs[chosen_layer[speaker]]
    proba[test] = fit_probe(x[~test], y[~test], args.C).predict_proba(x[test])
pred = np.array(TONES)[proba.argmax(1)]

per_speaker = {s: summarize(y[speaker_of == s], pred[speaker_of == s], TONES) for s in SPEAKERS}
speaker_acc = [r["accuracy"] for r in per_speaker.values()]
layer_curve = diagnostic.groupby("layer").accuracy.mean()
results = {
    "modelo": f"{data['model']}@{data['revision']}",
    "global": summarize(y, pred, TONES),
    "accuracy_hablante_media_std": [float(np.mean(speaker_acc)), float(np.std(speaker_acc))],
    "capa_elegida_por_hablante": {s: int(chosen_layer[s]) for s in SPEAKERS},
    "por_hablante": per_speaker,
    "por_genero": {g: summarize(y[meta.gender == g], pred[meta.gender == g], TONES)
                   for g in sorted(meta.gender.unique())},
    "diagnostico_accuracy_por_capa": layer_curve.round(4).to_dict(),
    "C": args.C,
}
g = results["global"]
print(f"\n=== {model_name} | lineal ===")
print("capa elegida por hablante:", results["capa_elegida_por_hablante"])
print(f"accuracy {g['accuracy']:.3f} (hablantes {np.mean(speaker_acc):.3f} ± {np.std(speaker_acc):.3f}) | "
      f"macro-F1 {g['macro_f1']:.3f}")
print("recall por tono:", g["recall_por_tono"])
print("por hablante:", {s: round(r["accuracy"], 3) for s, r in per_speaker.items()})
print("diagnóstico, accuracy por capa:", layer_curve.round(3).to_dict())

oof = meta.copy()
for i, t in enumerate(TONES):
    oof[f"p{t}"] = proba[:, i]
oof.to_parquet(out / "oof_lineal.parquet", index=False)
save_confusion(y, pred, TONES, f"{model_name} · lineal · acc {g['accuracy']:.3f}", out / "confusion_lineal.png")
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

fig, ax = plt.subplots(figsize=(6, 3.5))
for speaker in SPEAKERS:
    rows = diagnostic[diagnostic.test == speaker]
    ax.plot(rows.layer, rows.accuracy, lw=0.8, alpha=0.6, label=speaker)
ax.plot(layer_curve.index, layer_curve.values, color="k", lw=2.2, label="media")
ax.set_xlabel("capa (0 = salida de la CNN de entrada)")
ax.set_ylabel("accuracy en el hablante de test")
ax.set_title(f"Diagnóstico por capa – {model_name}")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / "accuracy_por_capa.png", dpi=120)
print(f"\nResultados en {out}/")
