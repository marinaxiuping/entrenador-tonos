"""Embeddings congelados por capa de cada sílaba alineada de AISHELL-3.

El modelo procesa la frase completa y se promedia cada capa dentro de los tiempos de la
sílaba (alineamiento del script 12). Mismo modelo y commit que en Tone Perfect (script 04).

Salida: data/interim/embeddings/aishell3_<modelo>/layer_<NN>.npy, un array (N, dim) en
float16 por capa, con las filas en el orden de data/processed/silabas_aishell3_praat.parquet.
Se guarda por capa para poder cargar una sola en memoria (cada una ocupa ~0,3 GB).

Uso: uv run python scripts/16_embeddings_aishell3.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tonos.align import load_16k
from tonos.embeddings import SpeechEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--model", default="TencentGameMate/chinese-hubert-base")
ap.add_argument("--revision", default="fce0375452b1dd6c080ac3248d423d4d037bc831")
ap.add_argument("--out", default="data/interim/embeddings")
args = ap.parse_args()

table = pd.read_parquet(args.syllables)
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SpeechEncoder(args.model, args.revision, device)
hidden_size = encoder.model.config.hidden_size
out = Path(args.out) / f"aishell3_{args.model.split('/')[-1]}"
out.mkdir(parents=True, exist_ok=True)
print(f"{args.model}@{args.revision[:10]} en {device}: {encoder.n_layers} capas, "
      f"{len(table)} sílabas de {table.utt.nunique()} frases")

# Se escribe en .partial y se renombra al final: un corte nunca deja capas a medias.
partial = [out / f"layer_{layer:02d}.partial.npy" for layer in range(encoder.n_layers)]
layers = [np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(table), hidden_size))
          for path in partial]
wav_root = Path(args.root) / "train" / "wav"
for (speaker, utt), rows in tqdm(table.groupby(["speaker", "utt"], sort=False), desc="frases"):
    waveform = load_16k(str(wav_root / speaker / f"{utt}.wav"))
    means = encoder.segment_means(waveform, list(zip(rows.start, rows.end))).astype(np.float16)
    for layer, array in enumerate(layers):
        array[rows.index] = means[:, layer]
for array, path in zip(layers, partial):
    array.flush()
    del array
layers.clear()
for path in partial:
    path.replace(path.with_name(path.name.replace(".partial", "")))
print(f"Guardado en {out}/")
