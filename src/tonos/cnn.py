"""CNN 1D sobre el contorno de F0 de una sílaba."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def contour_inputs(contours: np.ndarray) -> np.ndarray:
    """(N, L) en semitonos → (N, 2, L): contorno y su derivada.

    La derivada se da explícita porque la pendiente es lo que distingue los tonos y así la
    primera capa no tiene que aprender a calcularla.
    """
    return np.stack([contours, np.gradient(contours, axis=1)], axis=1).astype(np.float32)


def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class ContourCNN(nn.Module):
    """Tres bloques conv + pooling y una capa lineal sobre el mapa aplanado.

    Se aplana en vez de usar pooling global para conservar dónde ocurre cada forma en el
    tiempo: T2 y T3 tienen ambos un valle, y lo que los separa es su posición.
    """

    def __init__(self, n_points: int, n_classes: int = 4, in_channels: int = 2,
                 width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, width, 5),
            conv_block(width, 2 * width, 5),
            conv_block(2 * width, 2 * width, 3),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * width * (n_points // 8), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3


def train_cnn(x: np.ndarray, labels: np.ndarray, n_classes: int, config: TrainingConfig,
              device: str, seed: int = 0) -> ContourCNN:
    """Entrena un número fijo de épocas, sin early stopping, para no elegir el modelo
    mirando al hablante de test. ``labels`` son índices de clase (0..K-1)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ContourCNN(n_points=x.shape[-1], n_classes=n_classes, in_channels=x.shape[1]).to(device)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(labels).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                         generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                    epochs=config.epochs, steps_per_epoch=len(loader))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


@torch.no_grad()
def predict_logits(model: ContourCNN, x: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    chunks = [model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu().numpy()
              for i in range(0, len(x), batch_size)]
    return np.concatenate(chunks)


def predict_proba(model: ContourCNN, x: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    return torch.softmax(torch.from_numpy(predict_logits(model, x, device, batch_size)), dim=1).numpy()
