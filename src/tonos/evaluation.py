"""Métricas, desgloses y figuras comunes a los clasificadores de tono."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from tonos.calibration import brier_score, expected_calibration_error, negative_log_likelihood

PREDICTION_KEYS = ["speaker", "gender", "utt", "word_idx", "syl_idx", "pinyin", "tone"]


def domain_metrics(proba: np.ndarray, labels: np.ndarray, table: pd.DataFrame, tones: list[int]) -> dict:
    """Accuracy global, por hablante y por género, y calibración (NLL, ECE, Brier).

    ``labels`` son índices de columna de ``proba``; ``table`` tiene una fila por predicción.
    """
    y = table.tone.to_numpy()
    pred = np.array(tones)[proba.argmax(1)]
    speaker_accuracy = pd.Series(pred == y).groupby(table.speaker.to_numpy()).mean()
    genders = table.gender.to_numpy()
    return {
        "global": summarize(y, pred, tones),
        "accuracy_hablante_media_std": [float(speaker_accuracy.mean()), float(speaker_accuracy.std(ddof=0))],
        "por_genero": {g: summarize(y[genders == g], pred[genders == g], tones) for g in sorted(set(genders))},
        "nll": negative_log_likelihood(proba, labels),
        "ece": expected_calibration_error(proba, labels),
        "brier": brier_score(proba, labels),
    }


def prediction_table(table: pd.DataFrame, proba: np.ndarray, tones: list[int]) -> pd.DataFrame:
    """Claves de cada sílaba, tono predicho, confianza y probabilidad de cada tono."""
    predictions = table[PREDICTION_KEYS].reset_index(drop=True)
    predictions["pred"] = np.array(tones)[proba.argmax(1)]
    predictions["conf"] = proba.max(1).round(4)
    for i, tone in enumerate(tones):
        predictions[f"p{tone}"] = proba[:, i].round(4)
    return predictions


NEIGHBOUR_NAMES ={0: "ninguno", 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "neutro"}


def continuous_speech_factors(table: pd.DataFrame) -> pd.DataFrame:
    """Factores del habla continua que no existen en sílabas aisladas, uno por columna.

    ``table`` es la tabla de sílabas de AISHELL-3 (scripts 12 y 15).
    """
    position = np.select(
        [table.word_len == 1, table.syl_idx == 0, table.syl_idx == table.word_len - 1],
        ["monosílabo", "inicial", "final"], default="media")
    return pd.DataFrame({
        "tono": "T" + table.tone.astype(str),
        "duracion_ms": pd.qcut(table.dur * 1000, 4, precision=0).astype(str),
        "score_alineamiento": pd.qcut(table.score, 4, precision=2).astype(str),
        "longitud_palabra": table.word_len.clip(upper=4).astype(str).replace("4", "4+"),
        "posicion_en_palabra": position,
        "posicion_en_frase": pd.cut(table.rel_pos_utt, [0, 1 / 3, 2 / 3, 1], include_lowest=True,
                                    labels=["inicio", "medio", "final"]).astype(str),
        "tono_anterior": table.prev_tone.map(NEIGHBOUR_NAMES),
        "tono_siguiente": table.next_tone.map(NEIGHBOUR_NAMES),
        "tono_y_siguiente": "T" + table.tone.astype(str) + " + " + table.next_tone.map(NEIGHBOUR_NAMES),
        "genero": table.gender,
    }, index=table.index)


def accuracy_breakdown(correct: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Accuracy de cada modelo (columnas booleanas de ``correct``) por nivel de cada factor."""
    rows = []
    for factor, levels in factors.items():
        for level, group in correct.groupby(levels.to_numpy()):
            rows.append({"factor": factor, "nivel": level, "n": len(group), **group.mean().round(4).to_dict()})
    return pd.DataFrame(rows)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, tones: list[int]) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=tones, average="macro"),
        "recall_por_tono": dict(zip(map(str, tones),
                                    recall_score(y_true, y_pred, labels=tones, average=None).round(4).tolist())),
    }


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, tones: list[int], title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=tones, normalize="true")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(tones)):
        for j in range(len(tones)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > .5 else "black", fontsize=9)
    ax.set_xticks(range(len(tones)), [f"T{t}" for t in tones])
    ax.set_yticks(range(len(tones)), [f"T{t}" for t in tones])
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def speaker_bootstrap_accuracy(correct: pd.DataFrame, speakers: np.ndarray, n_resamples: int,
                               seed: int) -> np.ndarray:
    """(n_resamples, modelos): accuracy de cada columna de ``correct`` remuestreando hablantes.

    La unidad independiente es el hablante: remuestrear sílabas daría intervalos falsamente
    estrechos. Todos los modelos usan los mismos remuestreos, así que las diferencias entre
    columnas son pareadas.
    """
    correct_by_speaker = correct.groupby(speakers).sum().to_numpy()
    syllables_by_speaker = correct.groupby(speakers).size().to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(syllables_by_speaker),
                                                 size=(n_resamples, len(syllables_by_speaker)))
    return correct_by_speaker[draws].sum(axis=1) / syllables_by_speaker[draws].sum(axis=1)[:, None]
