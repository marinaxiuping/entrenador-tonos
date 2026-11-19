"""Comparación de los modelos sobre AISHELL-3: sin contexto, con contexto de F0 y HuBERT.

Une las predicciones out-of-fold de los scripts 29 (CNN por sílaba), 30 (ventana y frase)
y 31 (HuBERT) sobre las mismas sílabas. Añade un ensemble de frase + HuBERT (media de sus
probabilidades ya calibradas sin fuga, así que tampoco tiene fuga).

Incluye una línea base léxica sin audio: el tono más frecuente de cada carácter en los
folds de entrenamiento. Un modelo que reconoce qué se dice puede acertar el tono por el
léxico sin oír el pitch, y para un entrenador de tonos ese acierto no sirve: esta línea
base mide cuánto da ese atajo por sí solo.

Intervalos de confianza al 95 % por bootstrap de hablantes: la unidad independiente es el
hablante, no la sílaba, y remuestrear sílabas daría intervalos falsamente estrechos. Las
diferencias frente a la CNN por sílaba son pareadas (mismos hablantes en cada remuestreo).

Salidas en outputs/escalera_dominio/comparacion_contexto/: comparacion.csv,
desglose.csv y solapamiento_errores.json.

Uso: uv run python scripts/32_comparar_contexto_aishell3.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tonos.aishell3 import TONES, lexical_tone_baseline, load_syllables, speaker_fold_ids
from tonos.evaluation import accuracy_breakdown, continuous_speech_factors, speaker_bootstrap_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--rung", default="outputs/escalera_dominio/aishell3/predicciones.parquet")
ap.add_argument("--context-dir", default="outputs/escalera_dominio/aishell3_contexto")
ap.add_argument("--hubert", default="outputs/escalera_dominio/aishell3_hubert/predicciones_hubert.parquet")
ap.add_argument("--folds", type=int, default=5, help="los mismos folds que los scripts 29-31")
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/comparacion_contexto")
args = ap.parse_args()

KEYS = ["utt", "word_idx", "syl_idx"]
REFERENCE = "cnn_silaba"
PROBABILITY_COLUMNS = [f"p{t}" for t in TONES]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

syllables = load_syllables(args.syllables, args.contours)
table = syllables.table[syllables.evaluated].reset_index(drop=True)


def aligned(predictions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Columnas de ``predictions`` en el orden de filas de ``table``."""
    merged = table[KEYS].merge(predictions[KEYS + columns], on=KEYS, how="left", validate="one_to_one")
    if merged[columns].isna().any().any():
        raise SystemExit("Las predicciones no cubren las mismas sílabas: relanza los scripts 29-31")
    return merged[columns]


rung = pd.read_parquet(args.rung)
context_dir = Path(args.context_dir)
probabilities = {
    "frase": aligned(pd.read_parquet(context_dir / "predicciones_frase.parquet"), PROBABILITY_COLUMNS).to_numpy(),
    "hubert": aligned(pd.read_parquet(args.hubert), PROBABILITY_COLUMNS).to_numpy(),
}
predicted = {
    "lexico_sin_audio": lexical_tone_baseline(table, speaker_fold_ids(table, args.folds, args.seed)),
    "cero_disparos_tone_perfect": aligned(rung, ["pred_cero_disparos"]).iloc[:, 0].to_numpy(),
    REFERENCE: aligned(rung, ["pred_en_dominio_completo"]).iloc[:, 0].to_numpy(),
    "ventana": aligned(pd.read_parquet(context_dir / "predicciones_ventana.parquet"), ["pred"]).iloc[:, 0].to_numpy(),
    "frase": np.array(TONES)[probabilities["frase"].argmax(1)],
    "hubert": np.array(TONES)[probabilities["hubert"].argmax(1)],
    "frase_mas_hubert": np.array(TONES)[(probabilities["frase"] + probabilities["hubert"]).argmax(1)],
}
correct = pd.DataFrame({name: pred == table.tone.to_numpy() for name, pred in predicted.items()})

# --- Bootstrap de hablantes ---
boot_accuracy = speaker_bootstrap_accuracy(correct, table.speaker.to_numpy(), args.resamples, args.seed)
reference_column = correct.columns.get_loc(REFERENCE)
rows = []
for i, name in enumerate(correct.columns):
    difference = boot_accuracy[:, i] - boot_accuracy[:, reference_column]
    rows.append({
        "modelo": name,
        "accuracy": round(float(correct[name].mean()), 4),
        "ic95_inf": round(float(np.percentile(boot_accuracy[:, i], 2.5)), 4),
        "ic95_sup": round(float(np.percentile(boot_accuracy[:, i], 97.5)), 4),
        f"dif_vs_{REFERENCE}": round(float(correct[name].mean() - correct[REFERENCE].mean()), 4),
        "dif_ic95_inf": round(float(np.percentile(difference, 2.5)), 4),
        "dif_ic95_sup": round(float(np.percentile(difference, 97.5)), 4),
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(out / "comparacion.csv", index=False)

accuracy_breakdown(correct, continuous_speech_factors(table)).to_csv(out / "desglose.csv", index=False)

# --- ¿Se equivocan en las mismas sílabas el F0 con contexto y HuBERT? ---
frase_errors, hubert_errors = ~correct.frase, ~correct.hubert
overlap = {
    "errores_frase": int(frase_errors.sum()),
    "errores_hubert": int(hubert_errors.sum()),
    "errores_comunes": int((frase_errors & hubert_errors).sum()),
    "accuracy_oraculo": round(float((correct.frase | correct.hubert).mean()), 4),
}
(out / "solapamiento_errores.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")

print(comparison.to_string(index=False))
print("\nSolapamiento de errores frase / HuBERT:", overlap)
print(f"\nResultados en {out}/")
