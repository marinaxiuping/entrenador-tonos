"""El detector entrenado con alumnos (script 47) frente a errores fabricados con WORLD.

Se juzgan las frases de la demo (paquete del script 35, hablantes de AISHELL-3 no vistos):
cada sílaba de tono pleno tal cual y con cada uno de los otros tres tonos impuestos, igual
que en «¿Quién te corrige?». Para cada juez: tasa de aviso con la sílaba original (falsa
alarma) y con el tono cambiado (detecta). Un juez de tono avisa si el tono que oye no es el
del texto; el detector, si su p(tono mal dicho) supera su umbral.

Son 24 frases de 8 hablantes (sin intervalos): sirve para contrastar el orden de los jueces
con el de errores reales (scripts 41 y 46), no como estimación precisa.

Salida: --out (json).

Uso: uv run python scripts/48_detector_errores_fabricados.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from tonos.lab import ToneJudgeLab

ap = argparse.ArgumentParser()
ap.add_argument("--lab", default="models/laboratorio")
ap.add_argument("--device", help="cuda o cpu; por defecto, cuda si está disponible")
ap.add_argument("--out", default="outputs/escalera_dominio/ompal/detector/demo_fabricados.json")
args = ap.parse_args()

TONE_JUDGES = ["cnn_silaba", "frase", "hubert"]

lab = ToneJudgeLab(args.lab, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
rows = []
for utterance in tqdm(lab.utterance_list(), desc="frases"):
    for index, syllable in enumerate(utterance["syllables"]):
        if not syllable["evaluable"]:
            continue
        for imposed in [None] + [tone for tone in lab.tones if tone != syllable["tone"]]:
            verdicts = {verdict.judge: verdict for verdict in lab.judge(utterance["id"], index, imposed).verdicts}
            alerts = {judge: verdicts[judge].tone not in (None, syllable["tone"]) for judge in TONE_JUDGES}
            rows.append({"cambiada": imposed is not None, "detector": verdicts["detector"].alert, **alerts})

judged = pd.DataFrame(rows)
rates = judged.groupby("cambiada")[TONE_JUDGES + ["detector"]].mean()
result = {
    "n": {"frases": len(lab.utterance_list()), "silabas": int((~judged.cambiada).sum()),
          "juicios": len(judged)},
    "jueces": {judge: {"falsa_alarma": round(float(rates.loc[False, judge]), 4),
                       "detecta": round(float(rates.loc[True, judge]), 4)} for judge in rates.columns},
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(pd.DataFrame(result["jueces"]).T.to_string())
print(f"Resultados en {out}")
