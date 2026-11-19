"""Indexado del corpus Tone Perfect a partir de los nombres de fichero.

Convención esperada (verificar al descargar): ``<sílaba><tono>_<hablante>_MP3.mp3``,
p. ej. ``ma3_FV1_MP3.mp3``. Los hablantes son FV* (mujeres) y MV* (hombres).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

AUDIO_EXTS = {".mp3", ".wav", ".flac"}

FILENAME_RE = re.compile(
    r"^(?P<syllable>[a-zü]+?)(?P<tone>[1-4])_(?P<speaker>[FM]V\d+)(?:_MP3)?$",
    re.IGNORECASE,
)


def parse_filename(path: str | Path) -> dict | None:
    m = FILENAME_RE.match(Path(path).stem)
    if m is None:
        return None
    speaker = m["speaker"].upper()
    return {
        "path": str(path),
        "syllable": m["syllable"].lower(),
        "tone": int(m["tone"]),
        "speaker": speaker,
        "gender": speaker[0],
    }


def build_index(root: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Devuelve (índice, ficheros_no_reconocidos)."""
    rows, unparsed = [], []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        rec = parse_filename(p)
        (rows.append(rec) if rec else unparsed.append(str(p)))
    df = pd.DataFrame(rows, columns=["path", "syllable", "tone", "speaker", "gender"])
    return df, unparsed
