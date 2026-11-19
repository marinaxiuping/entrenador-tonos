"""Embeddings congelados por capa de cada sílaba de Tone Perfect.

Mismo preprocesado de audio que el pipeline de F0 (16 kHz, recorte de silencios) y media
temporal de cada capa del modelo. El modelo va fijado a un commit para que sea reproducible.

Salida: data/interim/embeddings/toneperfect_<modelo>.npz con `embeddings` (N, capas, dim) en
float16 y los metadatos de cada fila (sílaba, tono, hablante, género).

Uso: uv run python scripts/04_embeddings_toneperfect.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tonos.data import build_index
from tonos.embeddings import SpeechEncoder
from tonos.f0 import load, trim_silence

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--model", default="TencentGameMate/chinese-hubert-base")
ap.add_argument("--revision", default="fce0375452b1dd6c080ac3248d423d4d037bc831")
ap.add_argument("--out", default="data/interim/embeddings")
args = ap.parse_args()

index, _ = build_index(args.root)
if index.empty:
    raise SystemExit("No hay datos en --root")
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SpeechEncoder(args.model, args.revision, device)
print(f"{args.model}@{args.revision[:10]} en {device}: {encoder.n_layers} capas, {len(index)} sílabas")

embeddings = np.stack([
    encoder.layer_means(trim_silence(load(path))) for path in tqdm(index.path, desc="embeddings")
]).astype(np.float16)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
out_file = out / f"toneperfect_{args.model.split('/')[-1]}.npz"
np.savez_compressed(out_file, embeddings=embeddings, model=args.model, revision=args.revision,
                    **{c: index[c].to_numpy() for c in ["syllable", "tone", "speaker", "gender"]})
print(f"{embeddings.shape} | guardado en {out_file}")
