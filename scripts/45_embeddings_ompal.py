"""Embeddings de HuBERT de cada sílaba de OMPAL, para entrenar el detector de errores (46).

Misma capa y mismo codificador que el juez HuBERT de la demo (paquete del script 35): media
de la capa dentro del tramo de cada sílaba, con la frase entera como entrada. Las filas
siguen el orden de silabas_ompal.parquet (script 40).

Salida: --out (npz con «embeddings», n_sílabas x dimensión, y la capa usada).

Uso: uv run python scripts/45_embeddings_ompal.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tonos.align import load_16k
from tonos.lab import ToneJudgeLab

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--syllables", default="data/processed/silabas_ompal.parquet")
ap.add_argument("--lab", default="models/laboratorio")
ap.add_argument("--device", help="cuda o cpu; por defecto, cuda si está disponible")
ap.add_argument("--out", default="data/processed/embeddings_ompal_hubert.npz")
args = ap.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
table = pd.read_parquet(args.syllables)
lab = ToneJudgeLab(args.lab, device)
embeddings = None
for utterance_id, rows in tqdm(table.groupby("utt", sort=False), desc="HuBERT"):
    waveform = load_16k(str(Path(args.root) / "wav" / rows.speaker.iloc[0] / f"{utterance_id}.wav"))
    layer = lab.encoder.segment_means(waveform, list(zip(rows.start, rows.end)))[:, lab.hubert_layer]
    if embeddings is None:
        embeddings = np.full((len(table), layer.shape[1]), np.nan, dtype=np.float32)
    embeddings[rows.index] = layer

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, embeddings=embeddings, layer=lab.hubert_layer)
print(f"{len(table)} sílabas, capa {lab.hubert_layer}, dimensión {embeddings.shape[1]} -> {out}")
