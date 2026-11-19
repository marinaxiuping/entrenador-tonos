"""Clasificadores de tono que ven el contexto de la sílaba dentro de la frase.

Dos niveles de contexto sobre los mismos contornos de F0 que la CNN por sílaba:
  - ventana: la CNN recibe además los contornos de la sílaba anterior y la siguiente.
  - frase: una CNN codifica cada sílaba y una BiGRU recorre la frase entera, de modo que
    cada predicción puede usar la declinación, el sandhi y la coarticulación.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from tonos.cnn import ContourCNN, contour_inputs


def syllable_inputs(contours: np.ndarray) -> np.ndarray:
    """(N, L) → (N, 3, L): contorno, derivada y un canal que vale 1 si la sílaba tiene F0.

    Las sílabas sin F0 se quedan a cero en vez de descartarse porque siguen siendo
    contexto para sus vecinas.
    """
    has_f0 = np.isfinite(contours).all(axis=1)
    x = contour_inputs(np.where(has_f0[:, None], contours, 0.0))
    presence = np.broadcast_to(has_f0[:, None, None], (len(contours), 1, contours.shape[1]))
    return np.concatenate([x, presence.astype(np.float32)], axis=1)


def window_inputs(contours: np.ndarray, utterance_ids: np.ndarray) -> np.ndarray:
    """(N, 9, L): canales de la sílaba anterior, la actual y la siguiente de la misma frase.

    Las filas deben estar ordenadas por frase y posición. Sin vecina (borde de frase) sus
    canales quedan a cero, incluido el de presencia.
    """
    current = syllable_inputs(contours)
    same_as_previous = np.r_[False, utterance_ids[1:] == utterance_ids[:-1]]
    same_as_next = np.r_[utterance_ids[:-1] == utterance_ids[1:], False]
    previous = np.roll(current, 1, axis=0) * same_as_previous[:, None, None]
    following = np.roll(current, -1, axis=0) * same_as_next[:, None, None]
    return np.concatenate([previous, current, following], axis=1)


class UtteranceTagger(nn.Module):
    """Codificador convolucional por sílaba + BiGRU sobre las sílabas de la frase."""

    def __init__(self, n_points: int, n_classes: int, in_channels: int, width: int = 32,
                 hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = ContourCNN(n_points, n_classes, in_channels, width, dropout).features
        self.project = nn.Sequential(nn.Linear(2 * width * (n_points // 8), hidden), nn.ReLU(),
                                     nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.classifier = nn.Linear(2 * hidden, n_classes)

    def forward(self, syllables: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``syllables`` (total, C, L) con las frases concatenadas → logits (total, clases).

        Se codifican solo las sílabas reales (no el relleno) para que BatchNorm no vea ceros.
        """
        encoded = self.project(self.encoder(syllables))
        padded = pad_sequence(list(encoded.split(lengths.tolist())), batch_first=True)
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        real = torch.arange(states.shape[1], device=states.device)[None, :] < lengths[:, None].to(states.device)
        return self.classifier(states[real])


@dataclass(frozen=True)
class TaggerConfig:
    epochs: int = 30
    utterances_per_batch: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3


def _batch_rows(bounds: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Filas de las frases de ``bounds`` concatenadas y la longitud de cada frase."""
    rows = np.concatenate([np.arange(start, end) for start, end in bounds])
    return torch.from_numpy(rows).to(device), torch.from_numpy(bounds[:, 1] - bounds[:, 0])


def train_tagger(x: np.ndarray, labels: np.ndarray, bounds: np.ndarray, n_classes: int,
                 config: TaggerConfig, device: str, seed: int = 0) -> UtteranceTagger:
    """Entrena con las frases de ``bounds`` (filas [inicio, fin)) un número fijo de épocas.

    ``labels`` usa -1 para las sílabas que no se evalúan (neutras o sin F0): entran como
    contexto pero no cuentan en la pérdida.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_device = torch.from_numpy(x).to(device)
    labels_device = torch.from_numpy(labels).long().to(device)
    model = UtteranceTagger(x.shape[-1], n_classes, x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    steps_per_epoch = int(np.ceil(len(bounds) / config.utterances_per_batch))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                    epochs=config.epochs, steps_per_epoch=steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    for _ in range(config.epochs):
        order = rng.permutation(len(bounds))
        for i in range(0, len(order), config.utterances_per_batch):
            rows, lengths = _batch_rows(bounds[order[i:i + config.utterances_per_batch]], device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_device[rows], lengths), labels_device[rows])
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


@torch.no_grad()
def predict_tagger_logits(model: UtteranceTagger, x: np.ndarray, bounds: np.ndarray, device: str,
                          utterances_per_batch: int = 256) -> np.ndarray:
    """Logits de todas las sílabas de las frases de ``bounds``, en el orden de sus filas."""
    model.eval()
    x_device = torch.from_numpy(x).to(device)
    chunks = []
    for i in range(0, len(bounds), utterances_per_batch):
        rows, lengths = _batch_rows(bounds[i:i + utterances_per_batch], device)
        chunks.append(model(x_device[rows], lengths).cpu().numpy())
    return np.concatenate(chunks)
