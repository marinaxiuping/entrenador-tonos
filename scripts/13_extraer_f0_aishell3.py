"""F0 por frase completa (no por segmento) para los hablantes de AISHELL-3 descargados.

Guarda un .npz por hablante en data/interim/f0/<method>/<speaker>.npz con {utt: f0_hz (crudo)}.
Frames cada 10 ms empezando en t=0 (pyin con center=True y crepe con pad=True), así que
el frame i corresponde a i*0.01 s, igual que los tiempos del alineamiento.

Uso:
  uv run python scripts/13_extraer_f0_aishell3.py --method pyin  --speakers-file data/aishell3_hablantes.txt
  uv run python scripts/13_extraer_f0_aishell3.py --method crepe --speakers-file data/aishell3_hablantes.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from tonos.align import load_16k
from tonos.f0 import F0_METHODS

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--method", choices=["pyin", "crepe", "praat"], default="pyin")
ap.add_argument("--speakers", nargs="*", default=[])
ap.add_argument("--speakers-file")
ap.add_argument("--out", default="data/interim/f0")
ap.add_argument("--jobs", type=int, default=6, help="procesos para pyin (~0,6 GB de RAM cada uno)")
args = ap.parse_args()

speakers = list(args.speakers)
if args.speakers_file:
    speakers += pd.read_csv(args.speakers_file, sep="\t").speaker.tolist()
out = Path(args.out) / args.method
out.mkdir(parents=True, exist_ok=True)
fn = F0_METHODS[args.method]


def one(path: Path) -> tuple[str, np.ndarray]:
    return path.stem, fn(load_16k(str(path)))[0].astype(np.float32)


# Compila la caché de numba (pyin) en el proceso principal antes de paralelizar.
if args.method == "pyin":
    fn(np.sin(np.arange(4000) / 10).astype(np.float32))

for spk in tqdm(speakers, desc=args.method):
    dst = out / f"{spk}.npz"
    if dst.exists():
        continue
    wavs = sorted((Path(args.root) / "train/wav" / spk).glob("*.wav"))
    if not wavs:
        print(f"  {spk}: sin audio, se omite")
        continue
    if args.method == "crepe":  # GPU: secuencial
        res = [one(p) for p in wavs]
    else:
        res = Parallel(n_jobs=args.jobs)(delayed(one)(p) for p in wavs)
    tmp = dst.with_suffix(".tmp.npz")  # escritura atómica: un corte nunca deja un .npz a medias
    np.savez_compressed(tmp, **dict(res))
    tmp.replace(dst)
print("F0 en", out)
