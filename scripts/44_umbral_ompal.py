"""Umbral de aviso de los jueces de frase con las 5 particiones oficiales de OMPAL.

En el script 41 un juez avisa cuando el tono más probable no es el esperado, y así da a los
alumnos un 25-42 % de falsas alarmas. Aquí el juez avisa cuando p(tono esperado) < τ, y τ
se fija en el train de cada partición para que la falsa alarma con los alumnos sea la
objetivo (cuantil de p(tono esperado) en sus sílabas con el tono bien por mayoría). Se mide
en el test de la misma partición: detecta, falsa alarma con alumnos y falsa alarma con el
nativo. Como referencia, la regla del script 41 (argmax) en el mismo test.

Las particiones separan hablantes (42 en train, 5 en test: 4 alumnos y 1 nativo), pero los
test de las 5 se solapan entre sí: se resume con la media y el rango por partición, no
juntando los test. Con 4 alumnos por test, la variación entre particiones es grande.

Entrada: predicciones.parquet del script 41. Salidas en --out: particiones.csv (una fila
por partición, juez y regla) y resumen.json.

Uso: uv run python scripts/44_umbral_ompal.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tonos.ompal import is_tone_error, partition_utterances

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--predictions", default="outputs/escalera_dominio/ompal/predicciones.parquet")
ap.add_argument("--targets", default="0.05,0.10,0.20", help="falsas alarmas objetivo, separadas por comas")
ap.add_argument("--partitions", type=int, default=5)
ap.add_argument("--out", default="outputs/escalera_dominio/ompal/umbral")
args = ap.parse_args()

JUDGES = ["cnn_silaba", "frase", "hubert"]
targets = [float(value) for value in args.targets.split(",")]


def alert_rates(alerts: np.ndarray, rows: pd.DataFrame) -> dict:
    """Detecta y falsa alarma con alumnos (mayoría de 3 votos) y falsa alarma con nativos."""
    learners = (rows.group == "alumno").to_numpy()
    is_error = is_tone_error(rows).to_numpy()
    return {"detecta": alerts[learners & is_error].mean(),
            "falsa_alarma": alerts[learners & ~is_error].mean(),
            "falsa_alarma_nativos": alerts[~learners].mean(),
            "n_errores": int((learners & is_error).sum()),
            "n_bien": int((learners & ~is_error).sum()),
            "n_nativos": int((~learners).sum())}


predictions = pd.read_parquet(args.predictions)
predictions = predictions[predictions.evaluada].reset_index(drop=True)
root = Path(args.root)
rows = []
for index in range(1, args.partitions + 1):
    train = predictions[predictions.utt.isin(partition_utterances(root, "train", index))]
    test = predictions[predictions.utt.isin(partition_utterances(root, "test", index))]
    if set(train.speaker) & set(test.speaker):
        raise ValueError(f"la partición {index} comparte hablantes entre train y test")
    train_correct = train[(train.group == "alumno") & ~is_tone_error(train)]
    for judge in JUDGES:
        p_expected = test[f"p_esperado_{judge}"].to_numpy()
        argmax_alerts = (test[f"tono_{judge}"] != test.tone).to_numpy()
        rows.append({"particion": index, "juez": judge, "regla": "argmax", "umbral": np.nan,
                     **alert_rates(argmax_alerts, test)})
        for target in targets:
            threshold = float(np.quantile(train_correct[f"p_esperado_{judge}"], target))
            rows.append({"particion": index, "juez": judge, "regla": f"fa_{target:.2f}", "umbral": threshold,
                         **alert_rates(p_expected < threshold, test)})

by_partition = pd.DataFrame(rows)
measures = ["umbral", "detecta", "falsa_alarma", "falsa_alarma_nativos"]
summary = by_partition.groupby(["juez", "regla"], sort=False)[measures].agg(["mean", "min", "max"]).round(4)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
by_partition.round(4).to_csv(out / "particiones.csv", index=False)
# La regla argmax no tiene umbral: se omite para no escribir NaN, que no es JSON válido.
summary_records = {f"{judge} {rule}": {measure: summary.loc[(judge, rule), measure].to_dict() for measure in measures
                                       if summary.loc[(judge, rule), measure].notna().all()}
                   for judge, rule in summary.index}
(out / "resumen.json").write_text(json.dumps(summary_records, indent=2, ensure_ascii=False), encoding="utf-8")

print("Media por partición en test (min-max entre particiones en particiones.csv):")
print(summary.xs("mean", axis=1, level=1).to_string())
print(f"Resultados en {out}/")
