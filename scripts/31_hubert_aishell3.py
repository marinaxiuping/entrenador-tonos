"""HuBERT congelado + clasificador lineal sobre las sílabas de AISHELL-3.

Embeddings del script 16: el modelo ve la frase completa, así que es también un modelo
con contexto, a diferencia de la CNN por sílaba. Mismas sílabas evaluadas, folds por
hablante y calibración sin fuga que los scripts 29 y 30.

La capa se elige dentro de cada fold sin ver sus hablantes: con los demás folds se hace
una validación interna (cada uno de ellos como validación) y se queda la capa con mejor
accuracy media. La accuracy de cada capa sobre el fold de test se guarda solo como
diagnóstico: quedarse con su máximo sería optimista.

Salidas en outputs/escalera_dominio/aishell3_hubert/: metrics.json,
predicciones_hubert.parquet, confusion_hubert.png y accuracy_por_capa.csv.

Uso: uv run python scripts/31_hubert_aishell3.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tonos.aishell3 import TONES, load_syllables, speaker_fold_ids
from tonos.calibration import calibrate_by_group
from tonos.evaluation import domain_metrics, prediction_table, save_confusion
from tonos.linear_probe import LinearProbe

ap = argparse.ArgumentParser()
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--embeddings", default="data/interim/embeddings/aishell3_chinese-hubert-base")
ap.add_argument("--C", type=float, default=1.0, help="inversa de la regularización L2")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/aishell3_hubert")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# Se evalúan las mismas sílabas que los modelos de F0 (con contorno), para comparar.
syllables = load_syllables(args.syllables, args.contours)
evaluated = syllables.evaluated
table = syllables.table[evaluated].reset_index(drop=True)
labels = syllables.labels[evaluated]
fold_of = speaker_fold_ids(table, args.folds, args.seed)
layer_files = sorted(Path(args.embeddings).glob("layer_[0-9][0-9].npy"))
if not layer_files:
    raise SystemExit(f"No hay embeddings en {args.embeddings}: lanza antes el script 16")
labels_device = torch.from_numpy(labels).long().to(device)
folds_device = torch.from_numpy(np.array(fold_of)).to(device)


def accuracy(probe: LinearProbe, x: torch.Tensor, rows: torch.Tensor) -> float:
    return float((probe.logits(x[rows]).argmax(1) == labels_device[rows]).float().mean())


inner_rows, test_accuracy_rows = [], []
test_logits = {}
for layer, path in enumerate(layer_files):
    x = torch.from_numpy(np.load(path, mmap_mode="r")[evaluated].astype(np.float32)).to(device)
    for test_fold in range(args.folds):
        test = folds_device == test_fold
        probe = LinearProbe(len(TONES), args.C).fit(x[~test], labels_device[~test])
        test_logits[layer, test_fold] = probe.logits(x[test]).cpu().numpy()
        test_accuracy_rows.append({"capa": layer, "fold": test_fold, "accuracy": accuracy(probe, x, test)})
        for validation_fold in set(range(args.folds)) - {test_fold}:
            validation = folds_device == validation_fold
            train = ~test & ~validation
            probe = LinearProbe(len(TONES), args.C).fit(x[train], labels_device[train])
            inner_rows.append({"capa": layer, "fold": test_fold, "validacion": validation_fold,
                               "accuracy": accuracy(probe, x, validation)})
    del x
    print(f"  capa {layer:2d}: accuracy en test (diagnóstico) "
          f"{np.mean([r['accuracy'] for r in test_accuracy_rows if r['capa'] == layer]):.3f}")

inner = pd.DataFrame(inner_rows)
chosen_layer = inner.groupby(["fold", "capa"]).accuracy.mean().groupby("fold").idxmax().map(lambda key: key[1])
logits = np.zeros((len(table), len(TONES)), dtype=np.float32)
for test_fold, layer in chosen_layer.items():
    logits[fold_of == test_fold] = test_logits[layer, test_fold]
proba = calibrate_by_group(logits, labels, fold_of)
metrics = domain_metrics(proba, labels, table, TONES)

layer_curve = pd.DataFrame(test_accuracy_rows).groupby("capa").accuracy.mean().round(4)
layer_curve.to_csv(out / "accuracy_por_capa.csv")
results = {
    "modelo": Path(args.embeddings).name,
    "n_silabas": int(len(table)),
    "C": args.C,
    "capa_elegida_por_fold": {int(k): int(v) for k, v in chosen_layer.items()},
    "diagnostico_accuracy_por_capa": layer_curve.to_dict(),
    **metrics,
}
predictions = prediction_table(table, proba, TONES)
predictions.to_parquet(out / "predicciones_hubert.parquet", index=False)
save_confusion(predictions.tone.to_numpy(), predictions.pred.to_numpy(), TONES,
               f"AISHELL-3 · HuBERT lineal · acc {metrics['global']['accuracy']:.3f}", out / "confusion_hubert.png")
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

mean, std = metrics["accuracy_hablante_media_std"]
print(f"\ncapa elegida por fold: {results['capa_elegida_por_fold']}")
print(f"HuBERT lineal: accuracy {metrics['global']['accuracy']:.3f} (hablantes {mean:.3f} ± {std:.3f}) | "
      f"macro-F1 {metrics['global']['macro_f1']:.3f} | ECE {metrics['ece']:.3f} | "
      f"recall {metrics['global']['recall_por_tono']}")
print(f"Resultados en {out}/")
