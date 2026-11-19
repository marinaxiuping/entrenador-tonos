"""Inferencia de tono para una grabación nueva, sin dependencias del framework web.

Flujo: el usuario se calibra una vez (perfil de voz a partir de varias sílabas) y cada
grabación se analiza con ese perfil, igual que en el entrenamiento: F0 acotado al rango
del hablante, corrección de octava y semitonos respecto a su media.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tonos.calibration import apply_temperature
from tonos.cnn import ContourCNN, contour_inputs, predict_logits
from tonos.f0 import (
    fix_octave_jumps,
    fixed_length_contour,
    hz_to_st,
    speaker_pitch_range,
    speaker_reference_hz,
    waveform_f0,
)


class NoVoiceDetected(ValueError):
    """La grabación no tiene F0 suficiente para analizarla."""


@dataclass(frozen=True)
class SpeakerProfile:
    reference_hz: float
    fmin: float
    fmax: float


@dataclass(frozen=True)
class ToneAnalysis:
    tone: int
    probabilities: dict[int, float]
    contour_st: list[float]
    target_contour_st: list[float] | None


def build_speaker_profile(waveforms: list[np.ndarray], f0_method: str) -> SpeakerProfile:
    """Perfil de voz a partir de grabaciones de calibración.

    La referencia es la media del F0 de esas grabaciones, así que deben cubrir los
    cuatro tonos por igual (p. ej. mā má mǎ mà), como cada hablante de Tone Perfect.
    """
    first_pass = [waveform_f0(y, f0_method) for y in waveforms]
    if not any(np.isfinite(f0).any() for f0 in first_pass):
        raise NoVoiceDetected("ninguna grabación de calibración tiene voz detectable")
    fmin, fmax = speaker_pitch_range(first_pass)
    bounded = [waveform_f0(y, f0_method, fmin=fmin, fmax=fmax) for y in waveforms]
    provisional_reference = speaker_reference_hz(bounded)
    corrected = [fix_octave_jumps(f0, provisional_reference, fmin, fmax) for f0 in bounded]
    return SpeakerProfile(speaker_reference_hz(corrected), fmin, fmax)


def speaker_contour(waveform: np.ndarray, profile: SpeakerProfile, f0_method: str, n_points: int) -> np.ndarray:
    """Contorno de longitud fija en semitonos respecto a la media del hablante."""
    f0 = waveform_f0(waveform, f0_method, fmin=profile.fmin, fmax=profile.fmax)
    f0 = fix_octave_jumps(f0, profile.reference_hz, profile.fmin, profile.fmax)
    contour = fixed_length_contour(hz_to_st(f0, profile.reference_hz), n=n_points)
    if np.isnan(contour).all():
        raise NoVoiceDetected("no se detecta voz suficiente en la grabación")
    return contour


class ToneClassifier:
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        artifact = torch.load(model_path, map_location=device, weights_only=True)
        self.device = device
        self.tones: list[int] = artifact["tones"]
        self.n_points: int = artifact["n_points"]
        self.f0_method: str = artifact["f0_method"]
        self.temperature: float = artifact["temperature"]
        self.target_contours = {int(t): c for t, c in artifact["target_contours"].items()}
        self.model = ContourCNN(n_points=self.n_points, n_classes=len(self.tones)).to(device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def build_profile(self, waveforms: list[np.ndarray]) -> SpeakerProfile:
        return build_speaker_profile(waveforms, self.f0_method)

    def analyze(self, waveform: np.ndarray, profile: SpeakerProfile,
                target_tone: int | None = None) -> ToneAnalysis:
        contour = speaker_contour(waveform, profile, self.f0_method, self.n_points)
        logits = predict_logits(self.model, contour_inputs(contour[None]), self.device)
        probabilities = apply_temperature(logits, self.temperature)[0]
        return ToneAnalysis(
            tone=self.tones[int(probabilities.argmax())],
            probabilities={t: round(float(p), 4) for t, p in zip(self.tones, probabilities)},
            contour_st=contour.round(3).tolist(),
            target_contour_st=self.target_contours[target_tone] if target_tone is not None else None,
        )
