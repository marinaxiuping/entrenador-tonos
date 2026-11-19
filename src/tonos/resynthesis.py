"""Voz corregida: resíntesis con WORLD sustituyendo solo el F0 por el contorno objetivo.

Se conservan la envolvente espectral (timbre, vocal) y la aperiodicidad de la grabación
del usuario, así que suena su voz; solo cambia la entonación. Desplazamientos de F0
grandes (más de una octava) producen artefactos audibles en WORLD.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyworld

FRAME_PERIOD_MS = 5.0
OUTPUT_PEAK = 0.9


def replace_voiced_f0(f0_hz: np.ndarray, target_contour_st: np.ndarray, reference_hz: float) -> np.ndarray:
    """Pone el contorno objetivo en los frames con voz, estirado entre el primero y el último.

    Los frames sordos se mantienen sordos (0 Hz): forzar voz donde WORLD solo ve ruido
    suena a zumbido. ``target_contour_st`` está en semitonos respecto a ``reference_hz``.
    """
    new_f0 = np.zeros_like(f0_hz, dtype=np.float64)
    voiced = np.flatnonzero(f0_hz > 0)
    if voiced.size == 0:
        return new_f0
    first, last = voiced[0], voiced[-1]
    span_positions = np.linspace(0, 1, last - first + 1)
    target_positions = np.linspace(0, 1, len(target_contour_st))
    target_hz = reference_hz * 2 ** (np.interp(span_positions, target_positions, target_contour_st) / 12)
    new_f0[voiced] = target_hz[voiced - first]
    return new_f0


def impose_contour_on_span(f0_hz: np.ndarray, start_s: float, end_s: float,
                           target_contour_st: np.ndarray, reference_hz: float) -> np.ndarray:
    """Pone el contorno objetivo solo en los frames con voz del tramo [start_s, end_s)."""
    first = int(round(start_s * 1000 / FRAME_PERIOD_MS))
    last = int(round(end_s * 1000 / FRAME_PERIOD_MS))
    new_f0 = f0_hz.astype(np.float64).copy()
    new_f0[first:last] = replace_voiced_f0(f0_hz[first:last], target_contour_st, reference_hz)
    return new_f0


def flatten_voiced_f0(f0_hz: np.ndarray) -> np.ndarray:
    """F0 constante (media geométrica de los frames con voz) en los frames con voz.

    Quita toda la información tonal y conserva qué frames tienen voz.
    """
    voiced = f0_hz > 0
    if not voiced.any():
        return np.zeros_like(f0_hz, dtype=np.float64)
    return np.where(voiced, np.exp(np.mean(np.log(f0_hz[voiced]))), 0.0)


def resynthesize_with_f0(waveform: np.ndarray, sample_rate: int, fmin: float, fmax: float,
                         transform_f0: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Analiza con WORLD, cambia el F0 con ``transform_f0`` (Hz por frame, 0 = sordo) y resintetiza.

    ``fmin``/``fmax`` acotan Harvest: con el rango del hablante se evitan errores de octava,
    que en la resíntesis se oirían como saltos.
    """
    x = waveform.astype(np.float64)
    f0, times = pyworld.harvest(x, sample_rate, f0_floor=fmin, f0_ceil=fmax, frame_period=FRAME_PERIOD_MS)
    spectral_envelope = pyworld.cheaptrick(x, f0, times, sample_rate)
    aperiodicity = pyworld.d4c(x, f0, times, sample_rate)
    y = pyworld.synthesize(transform_f0(f0), spectral_envelope, aperiodicity, sample_rate, FRAME_PERIOD_MS)
    peak = np.abs(y).max()
    return (y / peak * OUTPUT_PEAK if peak > 0 else y).astype(np.float32)


def resynthesize_with_contour(waveform: np.ndarray, sample_rate: int, target_contour_st: np.ndarray,
                              reference_hz: float, fmin: float, fmax: float) -> np.ndarray:
    """Devuelve la grabación con el F0 sustituido por ``target_contour_st``."""
    return resynthesize_with_f0(waveform, sample_rate, fmin, fmax,
                                lambda f0: replace_voiced_f0(f0, target_contour_st, reference_hz))
