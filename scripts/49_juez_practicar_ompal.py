"""El juez de Practicar (CNN de Tone Perfect, script 25) ante los errores reales de OMPAL.

Practicar juzga sílabas sueltas, y OMPAL solo tiene frases de 5 a 21 sílabas: no hay una
validación limpia. Esto es una cota inferior: el mismo modelo y la misma regla que la app
(avisa si el tono más probable no es el esperado, src/tonos/api.py), aplicados a sílabas de
habla continua de alumnos, con el contorno del script 40 (mismo pipeline de F0). En AISHELL-3,
ese salto de dominio ya bajaba su accuracy de 0,988 a 0,566 (script 29), así que un mal
resultado aquí mezcla el error de dominio con el de juzgar a alumnos.

Mismas sílabas y etiqueta que el script 41. Se compara con la CNN de la demo (entrenada con
AISHELL-3), con AUC pareada por hablantes, y con el umbral para un 10 % de falsa alarma fijado
en el train de cada partición oficial, como en el script 44.

Salida: --out (metrics.json).

Uso: uv run python scripts/49_juez_practicar_ompal.py
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from tonos.calibration import apply_temperature
from tonos.cnn import contour_inputs, predict_logits
from tonos.evaluation import speaker_bootstrap_accuracy, speaker_bootstrap_auc
from tonos.inference import ToneClassifier
from tonos.ompal import is_tone_error, partition_utterances

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--predictions", default="outputs/escalera_dominio/ompal/predicciones.parquet")
ap.add_argument("--contours", default="data/processed/contornos_ompal_praat.npz")
ap.add_argument("--model", default=os.environ.get("TONOS_MODEL_PATH", "models/cnn_toneperfect_praat.pt"))
ap.add_argument("--false-alarm", type=float, default=0.10)
ap.add_argument("--partitions", type=int, default=5)
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/ompal/practicar/metrics.json")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
predictions = pd.read_parquet(args.predictions)
contours = np.load(args.contours)["contours"]
classifier = ToneClassifier(args.model, device)
if classifier.n_points != contours.shape[1]:
    raise ValueError("los contornos de OMPAL no tienen la longitud del modelo de Practicar")

evaluated = predictions.evaluada.to_numpy()
rows = predictions[evaluated].reset_index(drop=True)
probabilities = apply_temperature(predict_logits(classifier.model, contour_inputs(contours[evaluated]), device),
                                  classifier.temperature)
expected_index = np.searchsorted(classifier.tones, rows.tone.to_numpy())
rows["p_esperado_practicar"] = probabilities[np.arange(len(rows)), expected_index]
rows["alerta_practicar"] = probabilities.argmax(axis=1) != expected_index

learners = (rows.group == "alumno").to_numpy()
is_error = is_tone_error(rows).to_numpy()
speakers = rows.speaker.to_numpy()


def rate_and_interval(flags: np.ndarray, selection: np.ndarray) -> dict:
    boot = speaker_bootstrap_accuracy(pd.DataFrame({"tasa": flags[selection]}), speakers[selection],
                                      args.resamples, args.seed)[:, 0]
    return {"tasa": round(float(flags[selection].mean()), 4),
            "ic95": np.percentile(boot, [2.5, 97.5]).round(4).tolist()}


alerts = rows.alerta_practicar.to_numpy()
judges = {"practicar": 1 - rows.p_esperado_practicar, "cnn_demo": 1 - rows.p_esperado_cnn_silaba}
error_scores = pd.DataFrame({name: score.to_numpy() for name, score in judges.items()})[learners]
boot_auc = speaker_bootstrap_auc(is_error[learners], error_scores, speakers[learners], args.resamples, args.seed)
auc = {name: round(float(roc_auc_score(is_error[learners], error_scores[name])), 4) for name in judges}

detection_at_target = {name: [] for name in judges}
for index in range(1, args.partitions + 1):
    train = rows.utt.isin(partition_utterances(args.root, "train", index)).to_numpy() & learners
    test = rows.utt.isin(partition_utterances(args.root, "test", index)).to_numpy() & learners
    for name, score in judges.items():
        threshold = np.quantile(score[train & ~is_error], 1 - args.false_alarm)
        detection_at_target[name].append(float((score[test & is_error] > threshold).mean()))

result = {
    "n": {"silabas_alumnos": int(learners.sum()), "errores": int(is_error[learners].sum()),
          "silabas_nativos": int((~learners).sum())},
    "regla_de_la_app": {"detecta": rate_and_interval(alerts, learners & is_error),
                        "falsa_alarma": rate_and_interval(alerts, learners & ~is_error),
                        "falsa_alarma_nativos": rate_and_interval(alerts, ~learners),
                        "accuracy_nativos": round(float((~alerts[~learners]).mean()), 4)},
    "auc": auc,
    "auc_practicar_menos_cnn_demo": {
        "diferencia": round(auc["practicar"] - auc["cnn_demo"], 4),
        "ic95": np.percentile(boot_auc[:, 0] - boot_auc[:, 1], [2.5, 97.5]).round(4).tolist()},
    f"detecta_con_falsa_alarma_{args.false_alarm:.2f}": {
        name: {"media": round(float(np.mean(values)), 4), "por_particion": np.round(values, 4).tolist()}
        for name, values in detection_at_target.items()},
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=1, ensure_ascii=False))
