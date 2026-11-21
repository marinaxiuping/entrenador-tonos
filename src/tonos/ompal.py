"""Particiones oficiales y etiqueta de error de tono del corpus OMPAL."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MIN_ERROR_VOTES = 2


def partition_utterances(root: str | Path, split: str, index: int) -> set[str]:
    """Identificadores de frase de ``train`` o ``test`` de la partición ``index`` (1-5)."""
    path = Path(root) / split / f"{split}_{index}_scores.json"
    return set(json.loads(path.read_text(encoding="utf-8")))


def is_tone_error(syllables: pd.DataFrame) -> pd.Series:
    """Tono mal dicho según la mayoría de los tres anotadores (las sílabas de nativos no tienen votos)."""
    return syllables.tone_error_votes >= MIN_ERROR_VOTES
