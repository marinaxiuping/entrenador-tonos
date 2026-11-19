"""Calibración de probabilidades por temperature scaling y métricas de calibración."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

# LightGBM da probabilidades del orden de 1e-7; sin recorte el log puede ser -inf.
MIN_PROBA = 1e-12


def probabilities_to_logits(proba: np.ndarray) -> np.ndarray:
    """log(p): equivale a los logits salvo una constante por fila, que softmax ignora."""
    return np.log(np.clip(proba, MIN_PROBA, 1.0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(proba: np.ndarray, labels: np.ndarray) -> float:
    """``labels`` son índices de columna (0..K-1)."""
    picked = proba[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, MIN_PROBA, 1.0))))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperatura que minimiza la NLL. Se busca log(T) para que T sea siempre positiva."""
    result = minimize_scalar(
        lambda log_t: negative_log_likelihood(apply_temperature(logits, np.exp(log_t)), labels),
        bounds=(np.log(0.05), np.log(20.0)), method="bounded",
    )
    return float(np.exp(result.x))


def calibrate_by_group(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Temperature scaling sin fuga: la temperatura de cada grupo (p. ej. un fold de
    hablantes) se ajusta solo con los demás grupos."""
    proba = np.empty(logits.shape, dtype=float)
    for group in np.unique(groups):
        held_out = groups == group
        temperature = fit_temperature(logits[~held_out], labels[~held_out])
        proba[held_out] = apply_temperature(logits[held_out], temperature)
    return proba


def brier_score(proba: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(proba.shape[1])[labels]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def reliability_bins(proba: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Confianza media, accuracy y nº de muestras por bin de confianza (la de la clase predicha)."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == labels
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bin_ids, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_confidence = np.bincount(bin_ids, confidence, n_bins) / counts
        accuracy = np.bincount(bin_ids, correct, n_bins) / counts
    return {"confidence": mean_confidence, "accuracy": accuracy, "count": counts}


def expected_calibration_error(proba: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(proba, labels, n_bins)
    filled = bins["count"] > 0
    gaps = np.abs(bins["accuracy"][filled] - bins["confidence"][filled])
    return float(np.sum(gaps * bins["count"][filled]) / len(labels))
