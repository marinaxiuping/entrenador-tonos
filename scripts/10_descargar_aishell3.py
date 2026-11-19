"""Descarga parcial de AISHELL-3 (réplica HF shenyunhang/AISHELL-3, licencia Apache 2.0).

Uso:
  uv run python scripts/10_descargar_aishell3.py                       # solo metadatos
  uv run python scripts/10_descargar_aishell3.py --speakers SSB0005 SSB0009
"""

import argparse

from huggingface_hub import snapshot_download

REPO = "shenyunhang/AISHELL-3"
META = ["ReadMe.txt", "spk-info.txt", "phone_set.txt", "*/content.txt"]

ap = argparse.ArgumentParser()
ap.add_argument("--speakers", nargs="*", default=[], help="IDs tipo SSB0005")
ap.add_argument("--speakers-file", help="TSV con columna 'speaker' (p. ej. data/aishell3_hablantes.txt)")
ap.add_argument("--out", default="data/raw/aishell3")
args = ap.parse_args()
if args.speakers_file:
    import pandas as pd

    args.speakers += pd.read_csv(args.speakers_file, sep="\t").speaker.tolist()

patterns = META + [f"*/wav/{s}/*" for s in args.speakers]
path = snapshot_download(REPO, repo_type="dataset", local_dir=args.out, allow_patterns=patterns)
print("Descargado en", path)
