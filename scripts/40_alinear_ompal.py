"""Sílabas de OMPAL alineadas, con su tono esperado y su contorno, para juzgar errores reales.

OMPAL (sources/2026-11-19_brief_corpus-ompal.md) son 1.768 frases de 46 alumnos francófonos
de mandarín y 82 de 3 nativos. Tres anotadores marcan cada carácter de los alumnos con tono
correcto (1) o incorrecto (0); no dicen qué tono se oyó ni dónde está cada sílaba.

  - Tono esperado: pypinyin sobre la frase entera (resuelve los polífonos por palabra) con
    tone_sandhi, que aplica el sandhi de 一, 不 y parte del tercer tono. Un T3 seguido de otro
    T3 queda marcado como ambiguo: el sandhi depende de la estructura prosódica y pypinyin no
    siempre lo resuelve, así que no se sabe si lo correcto era T2 o T3.
  - Tiempos: el mismo alineador MMS_FA + fill_gaps que AISHELL-3 (script 12), con el pinyin
    esperado. Con alumnos que se equivocan el alineamiento puede ser peor: el score CTC de
    cada sílaba se guarda para poder filtrar.
  - Contornos: el mismo pipeline que el script 15 (perfil por hablante con todas sus frases,
    F0 sobre la frase completa, 64 puntos).

Salidas en --out: silabas_ompal.parquet (una fila por sílaba) y contornos_ompal_praat.npz
(mismo orden de filas).

Uso: uv run python scripts/40_alinear_ompal.py [--device cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypinyin import Style, lazy_pinyin
from tqdm import tqdm

from tonos.align import SyllableAligner, fill_gaps, load_16k
from tonos.utterance_contours import estimate_speaker_profile, syllable_contours

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/ompal/OMPAL-corpus-main")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--device", help="cuda o cpu; por defecto, cuda si está disponible")
ap.add_argument("--contour-points", type=int, default=64, help="igual que los modelos de AISHELL-3")
args = ap.parse_args()

root = Path(args.root)


def speaker_of(utterance_id: str) -> str:
    # El id es "0" + hablante de 5 cifras + frase de 2 cifras: 00200101 -> SPEAKER02001.
    return f"SPEAKER{utterance_id[1:6]}"


def tone_error_votes(word: dict) -> int | None:
    """Anotadores que marcan el tono como incorrecto; None en los nativos (nota ya agregada)."""
    votes = word["tone"]
    return sum(int(vote) == 0 for vote in votes) if isinstance(votes, list) else None


def utterance_rows(utterance_id: str, annotation: dict, group: str) -> list[dict]:
    text = annotation["text"]
    words = annotation["words"]
    syllables = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=True)
    if not (len(text) == len(words) == len(syllables)):
        return []
    tones = [int(s[-1]) if s[-1].isdigit() else 5 for s in syllables]
    return [{
        "utt": utterance_id,
        "speaker": speaker_of(utterance_id),
        "group": group,
        "syl_idx": i,
        "n_syl": len(text),
        "hanzi": character,
        "pinyin": syllable,
        "tone": tone,
        "ambiguous_sandhi": tone == 3 and i + 1 < len(tones) and tones[i + 1] == 3,
        "tone_error_votes": tone_error_votes(word),
    } for i, (character, word, syllable, tone) in enumerate(zip(text, words, syllables, tones))]


annotations = [("alumno", json.loads((root / "non-native_scores-detail.json").read_text(encoding="utf-8"))),
               ("nativo", json.loads((root / "native_scores.json").read_text(encoding="utf-8")))]
rows, skipped = [], []
for group, by_utterance in annotations:
    for utterance_id, annotation in by_utterance.items():
        new_rows = utterance_rows(utterance_id, annotation, group)
        (rows.extend(new_rows) if new_rows else skipped.append(utterance_id))
# Ordenadas por frase: así las sílabas de cada hablante quedan juntas y en el orden de sus frases.
table = pd.DataFrame(rows).sort_values(["utt", "syl_idx"]).reset_index(drop=True)
print(f"{table.utt.nunique()} frases y {len(table)} sílabas; descartadas por no cuadrar texto y anotación: {skipped}")

aligner = SyllableAligner(args.device)
waveforms, spans, starts, ends, scores = {}, {}, [], [], []
for utterance_id, syllables in tqdm(table.groupby("utt", sort=False).pinyin, desc="alineando"):
    waveform = load_16k(str(root / "wav" / speaker_of(utterance_id) / f"{utterance_id}.wav"))
    aligned = fill_gaps(aligner.align(waveform, list(syllables)))
    waveforms[utterance_id] = waveform
    spans[utterance_id] = [(s.start, s.end) for s in aligned]
    starts += [s.start for s in aligned]
    ends += [s.end for s in aligned]
    scores += [s.score for s in aligned]
table["start"], table["end"], table["align_score"] = starts, ends, scores

contours = np.full((len(table), args.contour_points), np.nan)
references = {}
for speaker, rows_of_speaker in tqdm(table.groupby("speaker", sort=False), desc="contornos"):
    utterance_ids = list(dict.fromkeys(rows_of_speaker.utt))
    speaker_waveforms = [waveforms[u] for u in utterance_ids]
    speaker_spans = [spans[u] for u in utterance_ids]
    profile = estimate_speaker_profile(speaker_waveforms, speaker_spans)
    references[speaker] = profile.reference_hz
    contours[rows_of_speaker.index] = syllable_contours(speaker_waveforms, speaker_spans, profile, args.contour_points)
table["reference_hz"] = table.speaker.map(references)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
table.to_parquet(out / "silabas_ompal.parquet", index=False)
np.savez_compressed(out / "contornos_ompal_praat.npz", contours=contours)
learners = table[table.group == "alumno"]
print(f"sin F0 suficiente: {np.isnan(contours).all(axis=1).mean():.1%} | T3 ante T3 (ambiguos): "
      f"{table.ambiguous_sandhi.sum()} | sílabas de alumnos con tono incorrecto por mayoría: "
      f"{(learners.tone_error_votes >= 2).mean():.1%}")
print(f"Guardado en {out}/")
