"""Selecciona hablantes de AISHELL-3 (train) equilibrados por género → data/aishell3_hablantes.txt

Uso: uv run python scripts/09_seleccionar_hablantes.py --per-gender 20
"""

import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--per-gender", type=int, default=20)
ap.add_argument("--min-bisilabos", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/aishell3_hablantes.txt")
args = ap.parse_args()

spk = pd.read_csv("data/raw/aishell3/spk-info.txt", sep=r"\s+", comment="#",
                  names=["speaker", "age", "gender", "accent"])
n_bi = pd.read_csv("outputs/aishell3/bisilabos_por_hablante.csv")
n_bi.columns = ["speaker", "n_bi"]
m = n_bi.merge(spk, on="speaker")
m = m[m.n_bi >= args.min_bisilabos]
sel = (m.sample(frac=1, random_state=args.seed).groupby("gender").head(args.per_gender)
       .sort_values(["gender", "speaker"]))
print(sel.groupby(["gender", "accent"]).size().unstack(fill_value=0))
print(f"Total: {len(sel)} hablantes, {sel.n_bi.sum()} bisílabos")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
sel[["speaker", "gender", "age", "accent", "n_bi"]].to_csv(args.out, sep="\t", index=False)
print("Guardado en", args.out)
