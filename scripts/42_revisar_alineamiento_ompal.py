"""Audios para revisar a oído el alineamiento de OMPAL (script 40).

Con alumnos que se equivocan, MMS_FA puede colocar mal las sílabas, y entonces el juez
evaluaría un trozo de audio que no es la sílaba anotada. Por cada frase de la muestra:
  - <frase>_silabas.wav: cada sílaba recortada con sus tiempos, separadas por silencios.
    Si el alineamiento está bien, se oye una sílaba completa en cada trozo.
  - <frase>_original.wav: la frase entera, para comparar.
La muestra son frases al azar de alumnos y de nativos y las de peor score CTC medio, que es
donde más probable es que falle. guia.csv lista, por frase y en orden, cada sílaba con su
pinyin esperado, los votos de error de tono y el score, y tiene una columna vacía para
anotar si el corte está bien.

Salidas en --out: los wav y guia.csv.

Uso: uv run python scripts/42_revisar_alineamiento_ompal.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from tonos.align import SR, load_16k

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--syllables", default="data/processed/silabas_ompal.parquet")
ap.add_argument("--learners", type=int, default=12, help="frases de alumnos al azar")
ap.add_argument("--natives", type=int, default=3, help="frases de nativos al azar")
ap.add_argument("--worst", type=int, default=5, help="frases de alumnos con peor score CTC medio")
ap.add_argument("--gap-seconds", type=float, default=0.4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/ompal/revision_alineamiento")
args = ap.parse_args()

table = pd.read_parquet(args.syllables)
utterances = table.groupby("utt").agg(group=("group", "first"), speaker=("speaker", "first"),
                                      score=("align_score", "mean"))
learner_ids = utterances[utterances.group == "alumno"]
worst = learner_ids.nsmallest(args.worst, "score").index
random_learners = learner_ids.drop(worst).sample(args.learners, random_state=args.seed).index
random_natives = utterances[utterances.group == "nativo"].sample(args.natives, random_state=args.seed).index
sample = [("peor_score", u) for u in worst] + [("alumno", u) for u in random_learners] + \
         [("nativo", u) for u in random_natives]

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
gap = np.zeros(int(args.gap_seconds * SR), dtype=np.float32)
guide = []
for order, (reason, utterance_id) in enumerate(sample, start=1):
    rows = table[table.utt == utterance_id].sort_values("syl_idx")
    waveform = load_16k(str(Path(args.root) / "wav" / rows.speaker.iloc[0] / f"{utterance_id}.wav"))
    pieces = []
    for row in rows.itertuples():
        pieces += [waveform[int(row.start * SR):int(row.end * SR)], gap]
        guide.append({"orden_frase": order, "frase": utterance_id, "motivo": reason, "posicion": row.syl_idx + 1,
                      "hanzi": row.hanzi, "pinyin_esperado": row.pinyin,
                      "votos_tono_mal": row.tone_error_votes, "score_ctc": round(row.align_score, 3),
                      "corte_bien": ""})
    prefix = f"{order:02d}_{reason}_{utterance_id}"
    sf.write(out / f"{prefix}_silabas.wav", np.concatenate(pieces), SR)
    sf.write(out / f"{prefix}_original.wav", waveform, SR)

pd.DataFrame(guide).to_csv(out / "guia.csv", index=False, encoding="utf-8-sig")
print(f"{len(sample)} frases y {len(guide)} sílabas para revisar en {out}/")
