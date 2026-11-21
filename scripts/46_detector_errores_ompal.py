"""Detector de errores de tono entrenado con alumnos reales de OMPAL, con sus 5 particiones.

Los jueces de los scripts 41 y 44 son clasificadores de tono entrenados con nativos: nunca
han visto los tonos intermedios de un alumno. Aquí se entrena directamente la pregunta de la
app, «¿la sílaba lleva bien el tono esperado?», con la mayoría de los tres profesores como
etiqueta, sobre tres conjuntos de entrada:
  - jueces: log p(tono esperado) de la CNN, la frase y HuBERT (script 41). Mide cuánto se gana
    solo con recalibrar y combinar los jueces que ya existen.
  - hubert: embedding de la capa de HuBERT del juez (script 45).
  - hubert_contorno: el embedding más el contorno de F0 de la sílaba (script 40).
Detector de tonos.error_detector: una regresión logística por tono esperado, con C elegido
por validación cruzada agrupada por hablante dentro del train.

Protocolo por partición (hablantes de train y test disjuntos): el umbral de aviso se fija con
predicciones fuera de fold del train (GroupKFold por hablante), para que no salga optimista,
y se mide en el test. Como referencia, los jueces sin entrenar con τ fijado igual que en el
script 44 (1 - p(tono esperado)). Se resume con media y rango por partición porque los test
de las 5 particiones se solapan.

Salidas en --out: particiones.csv y resumen.json.

Uso: uv run python scripts/46_detector_errores_ompal.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from tonos.error_detector import ToneErrorDetector, false_alarm_threshold, out_of_fold_scores
from tonos.ompal import is_tone_error, partition_utterances

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--predictions", default="outputs/escalera_dominio/ompal/predicciones.parquet")
ap.add_argument("--contours", default="data/processed/contornos_ompal_praat.npz")
ap.add_argument("--embeddings", default="data/processed/embeddings_ompal_hubert.npz")
ap.add_argument("--targets", default="0.10,0.20", help="falsas alarmas objetivo, separadas por comas")
ap.add_argument("--partitions", type=int, default=5)
ap.add_argument("--inner-folds", type=int, default=5, help="folds por hablante para el umbral fuera de fold")
ap.add_argument("--out", default="outputs/escalera_dominio/ompal/detector")
args = ap.parse_args()

JUDGES = ["cnn_silaba", "frase", "hubert"]
LOG_FLOOR = 1e-6
targets = [float(value) for value in args.targets.split(",")]


def operating_points(train_scores, train_labels, test_scores, test_labels, native_scores) -> dict:
    """AUC en test y, por cada falsa alarma objetivo, el umbral del train medido en test."""
    metrics = {"auc": roc_auc_score(test_labels, test_scores)}
    for target in targets:
        threshold = false_alarm_threshold(train_scores, train_labels, target)
        alerts = test_scores > threshold
        metrics[f"detecta_{target:.2f}"] = alerts[test_labels].mean()
        metrics[f"falsa_alarma_{target:.2f}"] = alerts[~test_labels].mean()
        metrics[f"falsa_alarma_nativos_{target:.2f}"] = (native_scores > threshold).mean()
    return metrics


predictions = pd.read_parquet(args.predictions)
contours = np.load(args.contours)["contours"]
embeddings = np.load(args.embeddings)["embeddings"]
if not (len(predictions) == len(contours) == len(embeddings)):
    raise ValueError("predicciones, contornos y embeddings no tienen las mismas sílabas: repetir 40, 41 y 45")

judge_log_probabilities = np.log(np.clip(predictions[[f"p_esperado_{j}" for j in JUDGES]].to_numpy(), LOG_FLOOR, 1))
feature_sets = {
    "jueces": judge_log_probabilities,
    "hubert": embeddings,
    "hubert_contorno": np.hstack([embeddings, contours]),
}
evaluated = predictions.evaluada.to_numpy()
feature_sets = {name: features[evaluated] for name, features in feature_sets.items()}
predictions = predictions[evaluated].reset_index(drop=True)
labels = is_tone_error(predictions).to_numpy()
learners = (predictions.group == "alumno").to_numpy()
tones = predictions.tone.to_numpy()
speakers = predictions.speaker.to_numpy()

rows = []
for index in tqdm(range(1, args.partitions + 1), desc="particiones"):
    in_train = predictions.utt.isin(partition_utterances(args.root, "train", index)).to_numpy() & learners
    in_test = predictions.utt.isin(partition_utterances(args.root, "test", index)).to_numpy()
    if set(speakers[in_train]) & set(speakers[in_test]):
        raise ValueError(f"la partición {index} comparte hablantes entre train y test")
    test_learners, test_natives = in_test & learners, in_test & ~learners

    for judge in JUDGES:
        error_score = 1 - predictions[f"p_esperado_{judge}"].to_numpy()
        rows.append({"particion": index, "modelo": f"juez_{judge}", **operating_points(
            error_score[in_train], labels[in_train], error_score[test_learners], labels[test_learners],
            error_score[test_natives])})

    for name, features in feature_sets.items():
        detector = ToneErrorDetector.fit(features[in_train], tones[in_train], labels[in_train], speakers[in_train])
        train_scores = out_of_fold_scores(features[in_train], tones[in_train], labels[in_train], speakers[in_train],
                                          args.inner_folds)
        rows.append({"particion": index, "modelo": f"detector_{name}", **operating_points(
            train_scores, labels[in_train], detector.score(features[test_learners], tones[test_learners]),
            labels[test_learners], detector.score(features[test_natives], tones[test_natives]))})

by_partition = pd.DataFrame(rows)
summary = by_partition.drop(columns="particion").groupby("modelo", sort=False).agg(["mean", "min", "max"]).round(4)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
by_partition.round(4).to_csv(out / "particiones.csv", index=False)
(out / "resumen.json").write_text(json.dumps(
    {model: {measure: summary.loc[model, measure].to_dict() for measure in summary.columns.levels[0]}
     for model in summary.index}, indent=2, ensure_ascii=False), encoding="utf-8")

print("Media por partición en test (min-max en particiones.csv):")
print(summary.xs("mean", axis=1, level=1).to_string())
print(f"Resultados en {out}/")
