"""Detector de errores de la demo "¿quién te corrige?", entrenado con todos los alumnos de OMPAL.

Es el detector sobre HuBERT que el script 46 evaluó con las 5 particiones, ahora ajustado con
los 46 alumnos. El umbral de aviso se fija con predicciones fuera de fold (GroupKFold por
hablante) para la falsa alarma objetivo, igual que en el 46: así la tasa de avisos a quien lo
dice bien se espera cercana al objetivo con hablantes nuevos (en el 46, de 0,07 a 0,17 para
0,10, según los alumnos del test).

Usa la capa y el codificador de HuBERT del paquete de la demo (script 35) a través de los
embeddings del script 45: comprueba que la capa coincide.

Salida: --out (detector_ompal.pt junto a laboratorio.pt).

Uso: uv run python scripts/47_detector_laboratorio.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from tonos.error_detector import ToneErrorDetector, false_alarm_threshold, out_of_fold_scores
from tonos.lab import BUNDLE_FILE, DETECTOR_FILE
from tonos.ompal import is_tone_error

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", default="outputs/escalera_dominio/ompal/predicciones.parquet")
ap.add_argument("--embeddings", default="data/processed/embeddings_ompal_hubert.npz")
ap.add_argument("--lab", default="models/laboratorio")
ap.add_argument("--false-alarm", type=float, default=0.10, help="falsa alarma objetivo con alumnos")
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

lab_dir = Path(args.lab)
bundle_layer = torch.load(lab_dir / BUNDLE_FILE, map_location="cpu", weights_only=True)["hubert"]["layer"]
stored = np.load(args.embeddings)
if int(stored["layer"]) != bundle_layer:
    raise ValueError(f"embeddings de la capa {int(stored['layer'])} y demo con la capa {bundle_layer}: repetir 45")

predictions = pd.read_parquet(args.predictions)
rows = (predictions.evaluada & (predictions.group == "alumno")).to_numpy()
learners = predictions[rows]
features, tones = stored["embeddings"][rows], learners.tone.to_numpy()
labels, speakers = is_tone_error(learners).to_numpy(), learners.speaker.to_numpy()

detector = ToneErrorDetector.fit(features, tones, labels, speakers)
held_out_scores = out_of_fold_scores(features, tones, labels, speakers, args.folds)
threshold = false_alarm_threshold(held_out_scores, labels, args.false_alarm)
alerts = held_out_scores > threshold

torch.save({
    "detector": detector.state_dict(),
    "threshold": threshold,
    "hubert_layer": bundle_layer,
    "training": {"corpus": "OMPAL", "speakers": int(learners.speaker.nunique()), "syllables": int(len(learners)),
                 "errors": int(labels.sum()), "false_alarm_target": args.false_alarm},
}, lab_dir / DETECTOR_FILE)
print(f"{len(learners)} sílabas de {learners.speaker.nunique()} alumnos ({labels.sum()} errores) | "
      f"C por tono {detector.chosen_c()} | umbral {threshold:.3f}")
print(f"Fuera de fold: AUC {roc_auc_score(labels, held_out_scores):.3f}, detecta {alerts[labels].mean():.3f}, "
      f"falsa alarma {alerts[~labels].mean():.3f} | guardado en {lab_dir / DETECTOR_FILE}")
