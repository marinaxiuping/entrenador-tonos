"""Peldaño 2 con errores reales: ¿qué juez detecta los tonos mal dichos por alumnos de OMPAL?

Los contrafactuales de los scripts 33 y 34 fabrican errores con WORLD. Aquí los errores son
de verdad: 46 alumnos francófonos, cada sílaba marcada por tres profesores como tono
correcto o incorrecto (script 40). Los jueces son los de la demo (paquete del script 35,
entrenados con 32 de los 40 hablantes de AISHELL-3); ningún hablante de OMPAL se vio al
entrenar.

Un juez «avisa» cuando el tono que oye no es el esperado. Medidas:
  - detecta: fracción de sílabas con el tono mal (mayoría de anotadores) en las que avisa.
  - falsa alarma: fracción de sílabas con el tono bien en las que avisa.
  - AUC: separación entre las dos clases con 1 - p(tono esperado), sin depender de un
    umbral ni de la calibración de cada juez.
La línea base léxica predice siempre el tono esperado: no avisa nunca, por construcción.

Se evalúan las sílabas de tono pleno con F0 (para que los cuatro jueces juzguen lo mismo) y
sin el sandhi ambiguo del T3 ante T3. Intervalos al 95 % por bootstrap de hablantes.

Salidas en --out: metrics.json y predicciones.parquet.

Uso: uv run python scripts/41_jueces_ompal.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from tonos.align import load_16k
from tonos.evaluation import speaker_bootstrap_accuracy
from tonos.lab import ToneJudgeLab

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--syllables", default="data/processed/silabas_ompal.parquet")
ap.add_argument("--contours", default="data/processed/contornos_ompal_praat.npz")
ap.add_argument("--lab", default="models/laboratorio")
ap.add_argument("--device", help="cuda o cpu; por defecto, cuda si está disponible")
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/ompal")
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
table = pd.read_parquet(args.syllables)
contours = np.load(args.contours)["contours"]
lab = ToneJudgeLab(args.lab, device)
AUDIO_JUDGES = ["cnn_silaba", "frase", "hubert"]

# --- Juicio de cada frase completa: la frase y HuBERT necesitan el contexto ---
logits = {judge: np.full((len(table), len(lab.tones)), np.nan) for judge in AUDIO_JUDGES}
for utterance_id, rows in tqdm(table.groupby("utt", sort=False), desc="jueces"):
    waveform = load_16k(str(Path(args.root) / "wav" / rows.speaker.iloc[0] / f"{utterance_id}.wav"))
    spans = list(zip(rows.start, rows.end))
    for judge, values in lab.utterance_logits(waveform, spans, contours[rows.index]).items():
        logits[judge][rows.index] = values

evaluated = (table.tone.isin(lab.tones) & ~table.ambiguous_sandhi & ~np.isnan(contours).any(axis=1)).to_numpy()
expected_index = np.searchsorted(lab.tones, table.tone.clip(1, 4).to_numpy())
predictions = table.copy()
for judge in AUDIO_JUDGES:
    probabilities = torch.softmax(torch.from_numpy(logits[judge]), dim=1).numpy()
    predictions[f"tono_{judge}"] = np.array(lab.tones)[np.nan_to_num(logits[judge], nan=-np.inf).argmax(axis=1)]
    predictions[f"p_esperado_{judge}"] = probabilities[np.arange(len(table)), expected_index]
predictions["evaluada"] = evaluated
predictions.to_parquet(out / "predicciones.parquet", index=False)

# --- Métricas sobre las sílabas evaluables de los alumnos y de los nativos ---
learners = predictions[evaluated & (predictions.group == "alumno")]
natives = predictions[evaluated & (predictions.group == "nativo")]
is_error = (learners.tone_error_votes >= 2).to_numpy()
is_unanimous_error = (learners.tone_error_votes == 3).to_numpy()
is_unanimous_correct = (learners.tone_error_votes == 0).to_numpy()
alerts = pd.DataFrame({judge: learners[f"tono_{judge}"].to_numpy() != learners.tone.to_numpy() for judge in AUDIO_JUDGES})
native_alerts = pd.DataFrame({judge: natives[f"tono_{judge}"].to_numpy() != natives.tone.to_numpy() for judge in AUDIO_JUDGES})
speakers = learners.speaker.to_numpy()


def interval(rows: np.ndarray) -> np.ndarray:
    """IC 95 % por hablante de la tasa de aviso de cada juez en las filas indicadas."""
    boot = speaker_bootstrap_accuracy(alerts[rows].reset_index(drop=True), speakers[rows], args.resamples, args.seed)
    return np.percentile(boot, [2.5, 97.5], axis=0)


detection_ci, false_alarm_ci = interval(is_error), interval(~is_error)
results = {
    "n": {"silabas_alumnos": int(len(learners)), "errores_mayoria": int(is_error.sum()),
          "errores_unanimes": int(is_unanimous_error.sum()), "hablantes": int(learners.speaker.nunique()),
          "silabas_nativos": int(len(natives)),
          "fuera_t3_ante_t3": int(table.ambiguous_sandhi[table.group == "alumno"].sum()),
          "fuera_sin_f0": int((np.isnan(contours).any(axis=1) & (table.group == "alumno").to_numpy()).sum())},
    "jueces": {},
}
for i, judge in enumerate(AUDIO_JUDGES):
    flag = alerts[judge].to_numpy()
    results["jueces"][judge] = {
        "detecta": round(float(flag[is_error].mean()), 4),
        "detecta_ic95": detection_ci[:, i].round(4).tolist(),
        "falsa_alarma": round(float(flag[~is_error].mean()), 4),
        "falsa_alarma_ic95": false_alarm_ci[:, i].round(4).tolist(),
        "detecta_unanimes": round(float(flag[is_unanimous_error].mean()), 4),
        "falsa_alarma_unanimes": round(float(flag[is_unanimous_correct].mean()), 4),
        "auc": round(float(roc_auc_score(is_error, 1 - learners[f"p_esperado_{judge}"])), 4),
        "falsa_alarma_nativos": round(float(native_alerts[judge].mean()), 4),
    }
results["jueces"]["lexico_sin_audio"] = {"detecta": 0.0, "falsa_alarma": 0.0, "auc": 0.5,
                                         "nota": "predice siempre el tono esperado"}
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"{results['n']['silabas_alumnos']} sílabas de {results['n']['hablantes']} alumnos, "
      f"{results['n']['errores_mayoria']} con el tono mal por mayoría ({is_error.mean():.1%})")
print(pd.DataFrame(results["jueces"]).T[["detecta", "falsa_alarma", "detecta_unanimes", "falsa_alarma_unanimes",
                                         "auc", "falsa_alarma_nativos"]].to_string())
print(f"Resultados en {out}/")
