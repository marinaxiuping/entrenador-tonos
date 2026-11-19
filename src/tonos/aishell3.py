"""Sílabas alineadas de AISHELL-3 (scripts 12 y 15) para los experimentos de cambio de dominio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tonos.splits import speaker_folds

TONES = [1, 2, 3, 4]


@dataclass(frozen=True)
class AishellSyllables:
    """Todas las sílabas alineadas, ordenadas por hablante, frase y posición.

    Las neutras y las que no tienen F0 se conservan porque sirven de contexto a los modelos
    de frase; ``evaluated`` marca las que se clasifican y evalúan (tonos 1-4 con contorno).
    """

    table: pd.DataFrame
    contours: np.ndarray
    evaluated: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        """Índice de clase (0..3) de cada sílaba; -1 si no se evalúa."""
        labels = np.full(len(self.table), -1)
        labels[self.evaluated] = np.searchsorted(TONES, self.table.tone.to_numpy()[self.evaluated])
        return labels

    def utterance_bounds(self) -> np.ndarray:
        return utterance_bounds(self.table.utt.to_numpy())


def utterance_bounds(utterance_ids: np.ndarray) -> np.ndarray:
    """(n_frases, 2) con la fila inicial y final (exclusiva) de cada frase; las filas de una
    misma frase deben ser contiguas."""
    if utterance_ids.size == 0:
        return np.empty((0, 2), dtype=int)
    starts = np.flatnonzero(np.r_[True, utterance_ids[1:] != utterance_ids[:-1]])
    return np.stack([starts, np.r_[starts[1:], utterance_ids.size]], axis=1)


def load_syllables(table_path: str | Path, contours_path: str | Path) -> AishellSyllables:
    table = pd.read_parquet(table_path)
    contours = np.load(contours_path)["contours"]
    if len(table) != len(contours):
        raise ValueError("la tabla de sílabas y los contornos no tienen el mismo número de filas")
    evaluated = table.tone.isin(TONES).to_numpy() & np.isfinite(contours).all(axis=1)
    return AishellSyllables(table, contours, evaluated)


def speaker_fold_ids(table: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Fold de cada fila, con los hablantes repartidos por género (tonos.splits)."""
    speakers = table[["speaker", "gender"]].drop_duplicates()
    return table.speaker.map(speaker_folds(speakers, n_folds, seed)).to_numpy()


def lexical_tone_baseline(table: pd.DataFrame, fold_of: np.ndarray) -> np.ndarray:
    """Predicción sin audio: el tono más frecuente del carácter en los otros folds; si no
    aparece, el de la sílaba sin tono y, en último caso, el tono más frecuente.

    Mide el atajo léxico: cuánto se acierta sabiendo solo qué se dice.
    """
    syllable = table.pinyin.str.replace(r"\d", "", regex=True)
    pred = np.zeros(len(table), dtype=int)
    for fold in np.unique(fold_of):
        train, test = fold_of != fold, fold_of == fold
        by_character = table[train].groupby("hanzi").tone.agg(lambda t: t.mode().iloc[0])
        by_syllable = table[train].groupby(syllable[train]).tone.agg(lambda t: t.mode().iloc[0])
        pred[test] = (table.hanzi[test].map(by_character)
                      .fillna(syllable[test].map(by_syllable))
                      .fillna(table.tone[train].mode().iloc[0])
                      .astype(int).to_numpy())
    return pred
