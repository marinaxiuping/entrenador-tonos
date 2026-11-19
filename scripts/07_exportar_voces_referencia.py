"""Exporta al frontend la altura de voz de cada hablante de Tone Perfect.

La práctica ofrece escuchar la sílaba dicha por un nativo y empieza por la voz más
parecida a la del usuario. Para compararlas, la referencia de cada hablante se calcula
igual que el perfil del usuario: con la misma sílaba de calibración en los cuatro tonos
(ma1-ma4) y el mismo procedimiento (ToneClassifier.build_profile).

Solo se exportan el código del hablante, su género y su referencia en Hz; el audio lo
sirve el backend desde la carpeta del corpus (licencia CC BY-NC 4.0, ver sources/).

Uso: uv run python scripts/07_exportar_voces_referencia.py
"""

import argparse
import json
from pathlib import Path

from tonos.data import build_index
from tonos.f0 import load
from tonos.inference import ToneClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--model", default="models/cnn_toneperfect_praat.pt")
ap.add_argument("--calibration-syllable", default="ma", help="la misma que calibra al usuario en la app")
ap.add_argument("--out", default="frontend/src/data/reference_voices.json")
args = ap.parse_args()

index, _ = build_index(args.root)
calibration = index[index.syllable == args.calibration_syllable]
classifier = ToneClassifier(args.model)
voices = []
for speaker, rows in calibration.groupby("speaker"):
    if sorted(rows.tone) != [1, 2, 3, 4]:
        raise SystemExit(f"{speaker}: faltan tonos de '{args.calibration_syllable}' para calcular su referencia")
    profile = classifier.build_profile([load(path) for path in rows.sort_values("tone").path])
    voices.append({"speaker": speaker, "gender": "female" if speaker.startswith("F") else "male",
                   "reference_hz": round(float(profile.reference_hz), 1)})

Path(args.out).write_text(json.dumps(voices, indent=1), encoding="utf-8")
print({v["speaker"]: v["reference_hz"] for v in voices}, "| guardado en", args.out)
