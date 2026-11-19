"""Splits por hablante (leave-speaker-out) equilibrados por género."""

from __future__ import annotations

import numpy as np
import pandas as pd


def speaker_folds(speakers: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Asigna cada hablante a un fold, repartiendo cada género por turnos.

    ``speakers`` necesita columnas 'speaker' y 'gender'. Devuelve {speaker: fold}.
    """
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    offset = 0
    for _, g in speakers.groupby("gender"):
        ids = g.speaker.to_numpy().copy()
        rng.shuffle(ids)
        for i, s in enumerate(ids):
            folds[s] = (i + offset) % n_folds
        offset += len(ids)  # el siguiente género empieza donde acabó el anterior
    return folds
