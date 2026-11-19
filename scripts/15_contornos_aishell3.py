"""Contornos de F0 por sílaba de AISHELL-3 con el mismo pipeline que Tone Perfect.

Para medir el cambio de dominio, la entrada del modelo tiene que construirse igual en los
dos corpus: Praat con rango por hablante en dos pasadas, fix_octave_jumps por sílaba,
semitonos respecto a la media geométrica del hablante y contorno de longitud fija. La
única diferencia es que aquí el F0 se calcula sobre la frase completa (Praat necesita
contexto en los bordes) y después se corta cada sílaba con el alineamiento del script 12.

La referencia del hablante sale de todas sus sílabas, con la distribución natural de
tonos del habla continua, no del 25 % por tono de Tone Perfect.

Salidas en data/processed/:
  - silabas_aishell3_praat.parquet: una fila por sílaba alineada, con su contexto.
  - contornos_aishell3_praat.npz: contorno de longitud fija, mismo orden de filas.

Uso: uv run python scripts/15_contornos_aishell3.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tonos.align import load_16k
from tonos.f0 import (
    FMAX,
    FMIN,
    fix_octave_jumps,
    fixed_length_contour,
    hz_to_st,
    speaker_pitch_range,
    speaker_reference_hz,
)
from tonos.features import frames
from tonos.utterance_contours import utterance_f0

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--align", default="outputs/aishell3/alineamiento_40/alineamiento_silabas.csv")
ap.add_argument("--speakers-file", default="data/aishell3_hablantes.txt")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--contour-points", type=int, default=64, help="igual que el modelo de Tone Perfect")
args = ap.parse_args()


def speaker_contours(rows: pd.DataFrame, wav_dir: Path, n_points: int) -> tuple[np.ndarray, float]:
    """Contornos de las sílabas de un hablante y su referencia en Hz."""
    waveforms = {utt: load_16k(str(wav_dir / f"{utt}.wav")) for utt in rows.utt.unique()}
    generic = [utterance_f0(y, FMIN, FMAX) for y in waveforms.values()]
    fmin, fmax = speaker_pitch_range(generic)
    bounded = {utt: utterance_f0(y, fmin, fmax) for utt, y in waveforms.items()}
    segments = [frames(bounded[r.utt], r.start, r.end) for r in rows.itertuples()]
    provisional_reference = speaker_reference_hz(segments)
    corrected = [fix_octave_jumps(s, provisional_reference, fmin, fmax) for s in segments]
    reference = speaker_reference_hz(corrected)
    contours = np.stack([fixed_length_contour(hz_to_st(s, reference), n=n_points) for s in corrected])
    return contours, reference


syllables = (pd.read_csv(args.align)
             .drop(columns=["path"])
             .sort_values(["utt", "word_idx", "syl_idx"])
             .reset_index(drop=True))
tones_in_utt = syllables.groupby("utt").tone
# 0 = no hay sílaba vecina en la frase.
syllables["prev_tone"] = tones_in_utt.shift(1).fillna(0).astype(int)
syllables["next_tone"] = tones_in_utt.shift(-1).fillna(0).astype(int)
syllables["rel_pos_utt"] = syllables.start / syllables.groupby("utt").end.transform("max")
genders = pd.read_csv(args.speakers_file, sep="\t").set_index("speaker").gender
syllables["gender"] = syllables.speaker.map(genders)
syllables = syllables.sort_values(["speaker", "utt", "word_idx", "syl_idx"]).reset_index(drop=True)

by_speaker = list(syllables.groupby("speaker"))
print(f"Extrayendo F0 (praat) de {syllables.utt.nunique()} frases de {len(by_speaker)} hablantes...")
wav_root = Path(args.root) / "train" / "wav"
per_speaker = Parallel(n_jobs=args.jobs)(
    delayed(speaker_contours)(rows, wav_root / speaker, args.contour_points) for speaker, rows in by_speaker
)
contours = np.concatenate([c for c, _ in per_speaker])
references = {speaker: ref for (speaker, _), (_, ref) in zip(by_speaker, per_speaker)}

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
syllables.to_parquet(out / "silabas_aishell3_praat.parquet", index=False)
np.savez_compressed(out / "contornos_aishell3_praat.npz", contours=contours)
missing = np.isnan(contours).all(axis=1)
print(f"{len(syllables)} sílabas, sin F0 suficiente: {missing.mean() * 100:.1f}% "
      f"(por tono: {syllables[missing].tone.value_counts().sort_index().to_dict()})")
print("Referencia por hablante (Hz): mediana %.0f, rango %.0f-%.0f"
      % (np.median(list(references.values())), min(references.values()), max(references.values())))
print(f"Guardado en {out}/")
