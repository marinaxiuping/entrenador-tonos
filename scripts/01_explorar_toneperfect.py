"""Exploración de Tone Perfect: hablantes, nº de muestras, balance de tonos, duraciones.

Uso: uv run python scripts/01_explorar_toneperfect.py --root data/raw/toneperfect
"""

import argparse
from pathlib import Path

import librosa
import pandas as pd

from tonos.data import build_index

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/toneperfect")
ap.add_argument("--out", default="outputs/exploracion")
ap.add_argument("--no-durations", action="store_true", help="omite el cálculo de duraciones (más rápido)")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

df, unparsed = build_index(args.root)
print(f"Ficheros reconocidos: {len(df)} | no reconocidos: {len(unparsed)}")
if unparsed:
    print("  ejemplos no reconocidos:", *unparsed[:5], sep="\n    ")
if df.empty:
    raise SystemExit("No hay datos: revisa --root y la convención de nombres en src/tonos/data.py")

print(f"\nHablantes ({df.speaker.nunique()}):", ", ".join(sorted(df.speaker.unique())))
print(f"Sílabas distintas: {df.syllable.nunique()}")

print("\nMuestras por hablante x tono:")
ct = pd.crosstab(df.speaker, df.tone, margins=True)
print(ct)

print("\nBalance de tonos (%):")
print((df.tone.value_counts(normalize=True).sort_index() * 100).round(1))

tones_per_syl = df.groupby("syllable").tone.nunique()
print("\nSílabas por nº de tonos disponibles:")
print(tones_per_syl.value_counts().sort_index())

# Huecos: combinaciones sílaba-tono que no están en todos los hablantes
cov = df.groupby(["syllable", "tone"]).speaker.nunique()
n_spk = df.speaker.nunique()
print(f"\nPares sílaba-tono incompletos (< {n_spk} hablantes): {(cov < n_spk).sum()} de {len(cov)}")

if not args.no_durations:
    df["duration_s"] = [librosa.get_duration(path=p) for p in df.path]
    print("\nDuración (s) por tono:")
    print(df.groupby("tone").duration_s.describe().round(3))

df.to_csv(out / "indice_toneperfect.csv", index=False)
ct.to_csv(out / "hablante_x_tono.csv")
print(f"\nGuardado en {out}/")
