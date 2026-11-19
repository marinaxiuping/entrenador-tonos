"""Embeddings congelados de un modelo de habla preentrenado (HuggingFace), por capa."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from tonos.f0 import SR


class SpeechEncoder:
    """Envoltorio fino sobre un modelo tipo wav2vec2/HuBERT para poder cambiarlo por otro
    sin tocar los scripts: solo expone la media temporal de cada capa."""

    def __init__(self, model_name: str, revision: str, device: str):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()

    @property
    def n_layers(self) -> int:
        """Capas ocultas + la salida de la CNN de entrada (capa 0)."""
        return self.model.config.num_hidden_layers + 1

    @property
    def frame_seconds(self) -> float:
        """Separación entre frames de salida: producto de los strides de la CNN de entrada."""
        return float(np.prod(self.model.config.conv_stride)) / SR

    @torch.no_grad()
    def hidden_states(self, waveform: np.ndarray) -> torch.Tensor:
        """(n_layers, frames, hidden) en el dispositivo del modelo, para una señal mono a 16 kHz."""
        inputs = self.feature_extractor(waveform, sampling_rate=SR, return_tensors="pt")
        outputs = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        return torch.stack([h[0] for h in outputs.hidden_states])

    def layer_means(self, waveform: np.ndarray) -> np.ndarray:
        """(n_layers, hidden): media temporal de cada capa."""
        return self.hidden_states(waveform).mean(dim=1).float().cpu().numpy()

    def segment_means(self, waveform: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
        """(n_spans, n_layers, hidden): media de cada capa dentro de cada tramo (inicio, fin) en s.

        El modelo ve la señal completa, así que por la autoatención cada tramo lleva ya
        información de su contexto.
        """
        states = self.hidden_states(waveform)
        n_frames = states.shape[1]
        means = []
        for start, end in spans:
            first = min(int(round(start / self.frame_seconds)), n_frames - 1)
            last = max(int(round(end / self.frame_seconds)), first + 1)
            means.append(states[:, first:last].mean(dim=1))
        return torch.stack(means).float().cpu().numpy()
