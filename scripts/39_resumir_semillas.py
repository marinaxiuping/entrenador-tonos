"""Variabilidad por semilla de entrenamiento de los scripts 29, 30 y 34.

La semilla 0 son las salidas de siempre; las demás, las repeticiones con --train-seed
(mismos folds, frases y contornos donantes; solo cambian la inicialización y el orden de
entrenamiento de las redes). La línea base léxica no se entrena y HuBERT es una regresión
logística con L-BFGS, así que su variación es de no determinismo de la GPU, no de semilla.

Salidas en --seeds-dir: semillas.csv (una fila por métrica y semilla) y resumen.csv (media,
desviación típica, mínimo y máximo por métrica). También dice si el orden de los jueces por
detección del cambio de tono es el mismo en todas las semillas.

Uso (después de lanzar 29, 30 y 34 con --train-seed 1 y 2 y --out <seeds-dir>/<carpeta>/semilla_<n>):
  uv run python scripts/39_resumir_semillas.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--experiments", default="outputs/escalera_dominio", help="salidas de la semilla 0")
ap.add_argument("--seeds-dir", default="outputs/escalera_dominio/semillas")
ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
args = ap.parse_args()

EXPERIMENT_DIRS = {"29": "aishell3", "30": "aishell3_contexto", "34": "contrafactual_cambio_tono"}
CHANGE_JUDGES = ["cnn_silaba", "frase", "hubert"]


def metrics_path(script: str, seed: int) -> Path:
    folder = EXPERIMENT_DIRS[script]
    base = Path(args.experiments) / folder if seed == 0 else Path(args.seeds_dir) / folder / f"semilla_{seed}"
    return base / "metrics.json"


def seed_metrics(seed: int) -> dict[str, float]:
    """Las cifras que cita el TFM, con nombre «script · métrica»."""
    domain = json.loads(metrics_path("29", seed).read_text(encoding="utf-8"))["condiciones"]
    context = json.loads(metrics_path("30", seed).read_text(encoding="utf-8"))["modelos"]
    change = json.loads(metrics_path("34", seed).read_text(encoding="utf-8"))["tasas"]
    values = {
        "29 · CNN por sílaba, tamaño de Tone Perfect": domain["en_dominio_tamano_tp"]["global"]["accuracy"],
        "29 · CNN por sílaba, completo": domain["en_dominio_completo"]["global"]["accuracy"],
        "30 · ventana ±1": context["ventana"]["global"]["accuracy"],
        "30 · frase (CNN + BiGRU)": context["frase"]["global"]["accuracy"],
    }
    for judge in CHANGE_JUDGES:
        for rate in ["detecta_cambio", "acepta_error", "acierto_original"]:
            values[f"34 · {judge} · {rate}"] = change[judge][rate]
    return values


rows = [{"metrica": name, "semilla": seed, "valor": value}
        for seed in [0, *args.seeds] for name, value in seed_metrics(seed).items()]
by_seed = pd.DataFrame(rows)
summary = (by_seed.groupby("metrica", sort=False).valor
           .agg(media="mean", sd="std", minimo="min", maximo="max").round(4).reset_index())

out = Path(args.seeds_dir)
out.mkdir(parents=True, exist_ok=True)
by_seed.to_csv(out / "semillas.csv", index=False)
summary.to_csv(out / "resumen.csv", index=False)

print(f"Semillas de entrenamiento: {[0, *args.seeds]}")
print(summary.to_string(index=False))

detection = by_seed[by_seed.metrica.str.endswith("detecta_cambio")].copy()
detection["juez"] = detection.metrica.str.split(" · ").str[1]
rankings = {seed: tuple(group.sort_values("valor", ascending=False).juez)
            for seed, group in detection.groupby("semilla")}
print("\nOrden por detección del cambio de tono:")
for seed, ranking in rankings.items():
    print(f"  semilla {seed}: {' > '.join(ranking)}")
print("Mismo orden en todas las semillas." if len(set(rankings.values())) == 1 else "EL ORDEN CAMBIA ENTRE SEMILLAS.")
print(f"\nResultados en {out}/")
