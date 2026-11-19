"""Regresión logística multinomial con L2 ajustada en GPU, para sondear embeddings congelados.

Minimiza el mismo objetivo que sklearn ``LogisticRegression(C)`` sobre datos estandarizados
(salvo el escalado: media de la pérdida + ||W||² / (2·C·n)), pero con L-BFGS en GPU, que
con ~150.000 filas por 768 dimensiones tarda segundos en vez de minutos por ajuste.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe:
    def __init__(self, n_classes: int, C: float = 1.0, max_iter: int = 100):
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter

    def fit(self, x: torch.Tensor, labels: torch.Tensor) -> "LinearProbe":
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0).clamp_min(1e-6)
        z = (x - self.mean) / self.std
        self.linear = nn.Linear(x.shape[1], self.n_classes).to(x.device)
        l2_weight = 1.0 / (2.0 * self.C * len(x))
        optimizer = torch.optim.LBFGS(self.linear.parameters(), max_iter=self.max_iter,
                                      line_search_fn="strong_wolfe")
        loss_fn = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(self.linear(z), labels) + l2_weight * self.linear.weight.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear((x - self.mean) / self.std)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Lo necesario para predecir: estandarización y pesos, en CPU."""
        return {"mean": self.mean.cpu(), "std": self.std.cpu(),
                "weight": self.linear.weight.detach().cpu(), "bias": self.linear.bias.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], device: str = "cpu") -> "LinearProbe":
        probe = cls(n_classes=state["weight"].shape[0])
        probe.mean, probe.std = state["mean"].to(device), state["std"].to(device)
        probe.linear = nn.Linear(state["weight"].shape[1], probe.n_classes).to(device)
        probe.linear.load_state_dict({"weight": state["weight"], "bias": state["bias"]})
        return probe
