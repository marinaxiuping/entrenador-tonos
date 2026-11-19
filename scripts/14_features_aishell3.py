"""Tabla de features por sílaba de los bisílabos alineados de AISHELL-3.

Entrada: alineamiento_silabas.csv (script 12) + F0 por frase (script 13).
Salida : data/processed/feats_<method>.parquet (una fila por sílaba de bisílabo).

Normalización: semitonos respecto a la media geométrica del F0 de TODO el hablante
(tras clean_f0), como en el alcance. Contexto = la otra sílaba del mismo bisílabo.

Uso: uv run python scripts/14_features_aishell3.py --method pyin
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tonos.f0 import clean_f0, hz_to_st, speaker_reference_hz
from tonos.features import N_PTS, contour_features, frames

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="pyin")
ap.add_argument("--align", default="outputs/aishell3/alineamiento_40/alineamiento_silabas.csv")
ap.add_argument("--f0-dir", default="data/interim/f0")
ap.add_argument("--speakers-file", default="data/aishell3_hablantes.txt")
ap.add_argument("--out", default="data/processed")
args = ap.parse_args()

spk_info = pd.read_csv(args.speakers_file, sep="\t").set_index("speaker")
syl = pd.read_csv(args.align)
utt_dur = syl.groupby("utt").end.max()
bi = syl[syl.word_len == 2].sort_values(["utt", "word_idx", "syl_idx"])

rows = []
for spk, g in tqdm(bi.groupby("speaker"), desc=f"features {args.method}"):
    npz = Path(args.f0_dir) / args.method / f"{spk}.npz"
    if not npz.exists():
        print(f"  {spk}: sin F0, se omite")
        continue
    f0 = {k: clean_f0(v) for k, v in np.load(npz).items()}
    ref = speaker_reference_hz(list(f0.values()))
    st = {k: hz_to_st(v, ref) for k, v in f0.items()}
    for (utt, wi), w in g.groupby(["utt", "word_idx"]):
        if len(w) != 2 or utt not in st:
            continue
        segs = [frames(st[utt], r.start, r.end) for r in w.itertuples()]
        feats = [contour_features(s) for s in segs]
        for pos, r in enumerate(w.itertuples()):
            ctx = {f"ctx_{k}": v for k, v in feats[1 - pos].items()}
            rows.append({
                "speaker": spk, "gender": spk_info.loc[spk, "gender"], "utt": utt, "word_idx": wi,
                "hanzi": "".join(w.hanzi), "pinyin": r.pinyin, "word_tones": "".join(map(str, w.tone)),
                "tone": r.tone, "pos": pos, "rel_pos_utt": r.start / utt_dur[utt], "score": r.score,
                **feats[pos], **ctx,
                "diff_mean": feats[pos]["mean"] - feats[1 - pos]["mean"],
                "diff_join": feats[1]["c0"] - feats[0][f"c{N_PTS - 1}"],  # salto σ1→σ2
            })

df = pd.DataFrame(rows)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / f"feats_{args.method}.parquet", index=False)
print(f"{len(df)} sílabas, {df.speaker.nunique()} hablantes, "
      f"sin F0 suficiente: {df['mean'].isna().mean() * 100:.1f}%  → {out}/feats_{args.method}.parquet")
print(df.groupby("tone").size().rename("n"))
