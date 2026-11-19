"""Entrena la CNN de producción con todos los hablantes de Tone Perfect y la empaqueta.

El artefacto incluye todo lo que necesita la inferencia: pesos, configuración del F0,
temperatura de calibración y el contorno objetivo de cada tono (media del corpus, en
semitonos respecto a la media del hablante), para dibujarlo frente al del usuario.

La temperatura viene de la calibración out-of-fold (22_calibrar_toneperfect.py): el
modelo final se entrena con más datos que los de cada fold, así que es una aproximación.

Uso: uv run python scripts/25_entrenar_cnn_final.py --method praat
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tonos.cnn import TrainingConfig, contour_inputs, train_cnn

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="praat")
ap.add_argument("--contours", default=None)
ap.add_argument("--temperature", default=None,
                help="temperature.json de la calibración de la CNN (por defecto, el de --method)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="models")
args = ap.parse_args()

data = np.load(args.contours or f"data/processed/contornos_toneperfect_{args.method}.npz", allow_pickle=True)
temperature_file = Path(args.temperature or f"outputs/calibracion_toneperfect/{args.method}/cnn/temperature.json")
temperature = json.loads(temperature_file.read_text(encoding="utf-8"))["temperature"]

TONES = [1, 2, 3, 4]
contours, tones = data["contours"], data["tone"].astype(int)
x = contour_inputs(contours)
usable = np.isfinite(x).all(axis=(1, 2))
labels = np.searchsorted(TONES, tones)
config = TrainingConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = train_cnn(x[usable], labels[usable], len(TONES), config, device, args.seed)

artifact = {
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "n_points": int(contours.shape[1]),
    "tones": TONES,
    "f0_method": args.method,
    "temperature": float(temperature),
    "target_contours": {str(t): np.nanmean(contours[tones == t], axis=0).round(3).tolist() for t in TONES},
    "training": {**asdict(config), "seed": args.seed, "n_samples": int(usable.sum()),
                 "speakers": sorted(set(data["speaker"].tolist()))},
}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
out_file = out / f"cnn_toneperfect_{args.method}.pt"
torch.save(artifact, out_file)
print(f"Modelo entrenado con {usable.sum()} sílabas, T={temperature:.3f} | guardado en {out_file}")
