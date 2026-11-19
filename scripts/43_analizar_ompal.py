"""Dos análisis de OMPAL sobre las predicciones del script 41.

1. Techo humano. Cada sílaba tiene el voto de tres profesores. Se compara a cada anotador con
   los otros dos cuando estos coinciden (referencia sin el propio anotador) y a cada juez con
   esa misma referencia: así juez y profesor se miden en la misma escala. También el kappa de
   Cohen entre pares de anotadores y entre cada juez y cada anotador. Supuesto: la posición
   del voto en el JSON es siempre el mismo anotador (el README no lo dice).
2. Tono frente a segmentos. Cada sílaba trae también si la consonante y la vocal están bien.
   Si un juez decide por la palabra que reconoce, debería dejar pasar más errores de tono
   cuando la sílaba está bien articulada (la palabra se reconoce) que cuando no.

Se usan las sílabas de alumnos evaluadas en el script 41. Intervalos al 95 % por bootstrap de
hablantes.

Salidas en --out: techo_humano.csv, kappa.csv, segmentos.csv y metrics_analisis.json.

Uso: uv run python scripts/43_analizar_ompal.py
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from tonos.evaluation import speaker_bootstrap_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--predictions", default="outputs/escalera_dominio/ompal/predicciones.parquet")
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/ompal")
args = ap.parse_args()

JUDGES = ["cnn_silaba", "frase", "hubert"]
ANNOTATORS = [0, 1, 2]


def annotator_votes(root: Path) -> pd.DataFrame:
    """Una fila por sílaba de alumno con el «mal» (1) o «bien» (0) de cada anotador y aspecto."""
    detail = json.loads((root / "non-native_scores-detail.json").read_text(encoding="utf-8"))
    rows = []
    for utterance_id, annotation in detail.items():
        for index, word in enumerate(annotation["words"]):
            row = {"utt": utterance_id, "syl_idx": index}
            for aspect, key in [("tono", "tone"), ("consonante", "phoneme_consonant"), ("vocal", "phoneme_vowel")]:
                for annotator, vote in enumerate(word[key]):
                    row[f"{aspect}_mal_{annotator}"] = int(vote) == 0
            rows.append(row)
    return pd.DataFrame(rows)


def wrong_by_majority(data: pd.DataFrame, aspect: str) -> pd.Series:
    return data[[f"{aspect}_mal_{a}" for a in ANNOTATORS]].sum(axis=1) >= 2


def rate_with_interval(flags: pd.DataFrame, speakers: np.ndarray) -> pd.DataFrame:
    """Tasa de cada columna booleana y su IC 95 % por hablante."""
    boot = speaker_bootstrap_accuracy(flags.reset_index(drop=True), speakers, args.resamples, args.seed)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"tasa": flags.mean().to_numpy(), "ic95_bajo": low, "ic95_alto": high}, index=flags.columns)


predictions = pd.read_parquet(args.predictions)
learners = predictions[predictions.evaluada & (predictions.group == "alumno")]
data = learners.merge(annotator_votes(Path(args.root)), on=["utt", "syl_idx"], how="left", validate="one_to_one")
if data["tono_mal_0"].isna().any():
    raise ValueError("hay sílabas evaluadas sin votos de anotador: revisar el script 40")
alerts = pd.DataFrame({judge: (data[f"tono_{judge}"] != data.tone).to_numpy() for judge in JUDGES})
speakers = data.speaker.to_numpy()
out = Path(args.out)

# --- 1. Techo humano: cada anotador y cada juez frente a los otros dos anotadores ---
ceiling_rows = []
for annotator in ANNOTATORS:
    first, second = [f"tono_mal_{other}" for other in ANNOTATORS if other != annotator]
    agree = (data[first] == data[second]).to_numpy()
    reference = data[first].to_numpy()[agree]
    candidates = pd.concat([data[[f"tono_mal_{annotator}"]].rename(columns={f"tono_mal_{annotator}": "anotador"}),
                            alerts], axis=1)[agree].reset_index(drop=True)
    for kind, rows in [("detecta", reference), ("falsa_alarma", ~reference)]:
        rates = rate_with_interval(candidates[rows], speakers[agree][rows])
        for who, rate in rates.iterrows():
            ceiling_rows.append({"anotador_fuera": annotator, "medida": kind, "quien": who, **rate.round(4).to_dict()})
ceiling = pd.DataFrame(ceiling_rows)
ceiling_summary = ceiling.groupby(["quien", "medida"]).tasa.mean().unstack().round(4)

kappas = [{"par": f"anotador {a} - anotador {b}",
           "kappa": cohen_kappa_score(data[f"tono_mal_{a}"], data[f"tono_mal_{b}"])}
          for a, b in combinations(ANNOTATORS, 2)]
kappas += [{"par": f"{judge} - anotador {a}", "kappa": cohen_kappa_score(alerts[judge], data[f"tono_mal_{a}"])}
           for judge in JUDGES for a in ANNOTATORS]
kappa = pd.DataFrame(kappas).round(4)
kappa["quien"] = kappa.par.str.split(" - ").str[0].where(~kappa.par.str.startswith("anotador"), "anotadores")
kappa_summary = kappa.groupby("quien").kappa.mean().round(4)

# --- 2. Tono frente a segmentos: ¿se detecta peor el error de tono si la sílaba se entiende? ---
tone_wrong = wrong_by_majority(data, "tono").to_numpy()
segments_right = (~wrong_by_majority(data, "consonante") & ~wrong_by_majority(data, "vocal")).to_numpy()
segment_rows = []
for tone_label, tone_rows in [("tono mal", tone_wrong), ("tono bien", ~tone_wrong)]:
    for segment_label, rows in [("consonante y vocal bien", tone_rows & segments_right),
                                ("consonante o vocal mal", tone_rows & ~segments_right)]:
        rates = rate_with_interval(alerts[rows], speakers[rows])
        for judge, rate in rates.iterrows():
            segment_rows.append({"tono": tone_label, "segmentos": segment_label, "juez": judge,
                                 "n": int(rows.sum()), **rate.round(4).to_dict()})
segments = pd.DataFrame(segment_rows)

ceiling.to_csv(out / "techo_humano.csv", index=False)
kappa.to_csv(out / "kappa.csv", index=False)
segments.to_csv(out / "segmentos.csv", index=False)
(out / "metrics_analisis.json").write_text(json.dumps({
    "techo_humano_medio": ceiling_summary.to_dict(orient="index"),
    "kappa_medio": kappa_summary.to_dict(),
    "segmentos": segment_rows,
}, indent=2, ensure_ascii=False), encoding="utf-8")

print("1. Frente a los otros dos anotadores (media de los tres repartos):")
print(ceiling_summary.to_string())
print("\nKappa de Cohen medio:")
print(kappa_summary.to_string())
print("\n2. Tasa de aviso según el tono y los segmentos (tono mal: detecta; tono bien: falsa alarma):")
print(segments.pivot_table(index=["tono", "segmentos", "n"], columns="juez", values="tasa").to_string())
print(f"\nResultados en {out}/")
