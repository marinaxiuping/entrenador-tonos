"""Contornos de F0 de las sílabas de una frase, con el mismo pipeline que AISHELL-3 (script 15).

El F0 se calcula sobre la frase completa y después se corta cada sílaba por sus tiempos.
Separar el perfil del hablante (rango y referencia) del cálculo de los contornos permite
estimarlo con audio original y aplicarlo sin cambios a audio manipulado, como hace la app
con el perfil de calibración.
"""

from __future__ import annotations

import numpy as np

from tonos.f0 import (
    FMAX,
    FMIN,
    f0_on_frame_grid,
    f0_praat,
    fix_octave_jumps,
    fixed_length_contour,
    hz_to_st,
    speaker_pitch_range,
    speaker_reference_hz,
)
from tonos.features import frames
from tonos.inference import SpeakerProfile

Spans = list[tuple[float, float]]


def utterance_f0(waveform: np.ndarray, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """F0 de Praat en la rejilla de 10 ms que empieza en t=0 (la de los alineamientos)."""
    f0, times = f0_praat(waveform, fmin=fmin, fmax=fmax)
    return f0_on_frame_grid(f0, times, waveform.size)


def syllable_f0_segments(waveforms: list[np.ndarray], spans: list[Spans], fmin: float,
                         fmax: float) -> list[np.ndarray]:
    """F0 en Hz de cada sílaba, en el orden de las frases y de sus tramos."""
    segments = []
    for waveform, utterance_spans in zip(waveforms, spans):
        f0 = utterance_f0(waveform, fmin, fmax)
        segments.extend(frames(f0, start, end) for start, end in utterance_spans)
    return segments


def estimate_speaker_profile(waveforms: list[np.ndarray], spans: list[Spans]) -> SpeakerProfile:
    """Rango en dos pasadas y referencia (media geométrica tras corregir octavas)."""
    fmin, fmax = speaker_pitch_range([utterance_f0(y) for y in waveforms])
    segments = syllable_f0_segments(waveforms, spans, fmin, fmax)
    provisional_reference = speaker_reference_hz(segments)
    corrected = [fix_octave_jumps(s, provisional_reference, fmin, fmax) for s in segments]
    return SpeakerProfile(speaker_reference_hz(corrected), fmin, fmax)


def syllable_contours(waveforms: list[np.ndarray], spans: list[Spans], profile: SpeakerProfile,
                      n_points: int) -> np.ndarray:
    """(n_sílabas, n_points) en semitonos respecto a la referencia del perfil; NaN sin F0."""
    segments = syllable_f0_segments(waveforms, spans, profile.fmin, profile.fmax)
    return np.stack([
        fixed_length_contour(hz_to_st(fix_octave_jumps(s, profile.reference_hz, profile.fmin, profile.fmax),
                                      profile.reference_hz), n=n_points)
        for s in segments
    ])
