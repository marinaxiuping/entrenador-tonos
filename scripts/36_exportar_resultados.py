"""Datos de la pestaña "Resultados" del frontend: cifras de los experimentos y ejemplos para escuchar.

Las cifras se leen de las salidas de los scripts 29 y 32-34, sin copiarlas a mano. Los
ejemplos se calculan con el mismo juez que la demo en vivo (tonos.lab, paquete del script
35): sílaba y tono impuesto al azar con semilla, sin elegir los casos que salen bien.

Salidas: frontend/src/data/resultados.json y frontend/public/resultados/*.wav.

Uso: uv run python scripts/36_exportar_resultados.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tonos.lab import ToneJudgeLab

ap = argparse.ArgumentParser()
ap.add_argument("--experiments", default="outputs/escalera_dominio")
ap.add_argument("--lab", default="models/laboratorio")
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--examples", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data-out", default="frontend/src/data/resultados.json")
ap.add_argument("--audio-out", default="frontend/public/resultados")
args = ap.parse_args()

experiments = Path(args.experiments)
audio_out = Path(args.audio_out)
audio_out.mkdir(parents=True, exist_ok=True)


def interval(row: pd.Series, lower: str = "ic95_inf", upper: str = "ic95_sup") -> list[float]:
    return [float(row[lower]), float(row[upper])]


rung = json.loads((experiments / "aishell3" / "metrics.json").read_text(encoding="utf-8"))
conditions = rung["condiciones"]
ladder = [
    {"etiqueta": "Tone Perfect, hablante no visto", "accuracy": rung["peldano_0_tone_perfect_hablante_no_visto"],
     "ece": None},
    {"etiqueta": "AISHELL-3, modelo de la app sin tocar", "accuracy": conditions["cero_disparos"]["global"]["accuracy"],
     "ece": conditions["cero_disparos"]["ece"]},
    {"etiqueta": "AISHELL-3, entrenado en el dominio con el tamaño de Tone Perfect",
     "accuracy": conditions["en_dominio_tamano_tp"]["global"]["accuracy"], "ece": conditions["en_dominio_tamano_tp"]["ece"]},
    {"etiqueta": "AISHELL-3, entrenado en el dominio con todo (14 veces más)",
     "accuracy": conditions["en_dominio_completo"]["global"]["accuracy"], "ece": conditions["en_dominio_completo"]["ece"]},
]

CONTEXT_LABELS = {
    "cero_disparos_tone_perfect": "Modelo de la app sin tocar",
    "cnn_silaba": "CNN por sílaba",
    "ventana": "CNN con la sílaba anterior y la siguiente",
    "frase": "CNN + BiGRU sobre la frase",
    "hubert": "HuBERT congelado + lineal",
    "frase_mas_hubert": "Frase + HuBERT",
    "lexico_sin_audio": "Léxico, sin audio",
}
comparison = pd.read_csv(experiments / "comparacion_contexto" / "comparacion.csv")
# El léxico va al final: es la referencia sin audio con la que se cierra la comparación.
comparison = comparison.iloc[comparison.modelo.eq("lexico_sin_audio").argsort(kind="stable")]
context = [{"modelo": r.modelo, "etiqueta": CONTEXT_LABELS[r.modelo], "accuracy": float(r.accuracy),
            "ic95": interval(r)} for _, r in comparison.iterrows()]

FLAT_LABELS = {"original": "Audio original", "resintesis": "Resíntesis sin tocar el F0",
               "f0_plano": "F0 plano (sin tonos)", "lexico_sin_audio": "Léxico, sin audio"}
flat = pd.read_csv(experiments / "contrafactual_f0_plano" / "comparacion.csv")
flat_f0 = [{"condicion": r.condicion, "etiqueta": FLAT_LABELS[r.condicion], "accuracy": float(r.accuracy),
            "ic95": interval(r)} for _, r in flat.iterrows()]

swap = pd.read_csv(experiments / "contrafactual_cambio_tono" / "comparacion.csv").set_index(["modelo", "metrica"])
JUDGE_OF_MODEL = {"cnn_silaba": "cnn_silaba", "frase": "frase", "hubert": "hubert", "lexico_sin_audio": "lexico"}
inversion = [{"juez": judge,
              "accuracy_nativa": float(swap.loc[(model, "acierto_original"), "tasa"]),
              "detecta": float(swap.loc[(model, "detecta_cambio"), "tasa"]),
              "detecta_ic95": interval(swap.loc[(model, "detecta_cambio")]),
              "acepta": float(swap.loc[(model, "acepta_error"), "tasa"])}
             for model, judge in JUDGE_OF_MODEL.items()]
swap_metrics = json.loads((experiments / "contrafactual_cambio_tono" / "metrics.json").read_text(encoding="utf-8"))

# --- Ejemplos con el juez de la demo ---
lab = ToneJudgeLab(args.lab)
rng = np.random.default_rng(args.seed)
examples = []
for utterance in rng.choice(lab.utterance_list(), size=args.examples, replace=False):
    index = int(rng.choice([i for i, s in enumerate(utterance["syllables"]) if s["evaluable"]]))
    syllable = utterance["syllables"][index]
    imposed = int(rng.choice([t for t in lab.tones if t != syllable["tone"]]))
    original, changed = lab.judge(utterance["id"], index, None), lab.judge(utterance["id"], index, imposed)
    names = {"original": f"{utterance['id']}_original.wav", "cambiado": f"{utterance['id']}_{index}_T{imposed}.wav"}
    (audio_out / names["original"]).write_bytes(original.audio_wav)
    (audio_out / names["cambiado"]).write_bytes(changed.audio_wav)
    examples.append({
        "frase": utterance["text"], "silaba": syllable["hanzi"], "pinyin": syllable["pinyin"],
        "tono_original": syllable["tone"], "tono_impuesto": imposed,
        "audio_original": f"resultados/{names['original']}", "audio_cambiado": f"resultados/{names['cambiado']}",
        "veredictos": [{"juez": v.judge, "tono": v.tone} for v in changed.verdicts],
    })

results = {
    "escalera": ladder,
    "contexto": context,
    "f0_plano": flat_f0,
    "inversion": inversion,
    "ejemplos": examples,
    "n": {"silabas_aishell3": rung["n_silabas"], "hablantes_aishell3": int(pd.read_parquet(args.syllables, columns=["speaker"]).speaker.nunique()),
          "silabas_cambiadas": swap_metrics["n_silabas_cambiadas"]},
}
Path(args.data_out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Guardado en {args.data_out} y {audio_out}/ ({len(examples)} ejemplos)")
