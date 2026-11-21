"""Detector de errores de tono: ¿la sílaba lleva bien el tono esperado?

Una regresión logística por tono esperado sobre una representación fija de la sílaba (en la
demo, el embedding de HuBERT): lo que delata un T3 mal dicho no es lo que delata un T4 mal
dicho. Se ajusta con scikit-learn y se guarda como arrays, para cargarlo con
``torch.load(weights_only=True)`` sin pickle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

C_GRID = [0.001, 0.01, 0.1, 1.0]
MAX_ITER = 3000


@dataclass(frozen=True)
class ToneModel:
    """Estandarización y regresión logística de un tono esperado."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    c: float

    def error_probability(self, features: np.ndarray) -> np.ndarray:
        logit = ((features - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-logit))


class ToneErrorDetector:
    def __init__(self, models: dict[int, ToneModel]):
        self.models = models

    @classmethod
    def fit(cls, features: np.ndarray, tones: np.ndarray, labels: np.ndarray, speakers: np.ndarray,
            c_by_tone: dict[int, float] | None = None, c_folds: int = 3) -> ToneErrorDetector:
        """``labels`` es True si el tono está mal dicho. Sin ``c_by_tone``, C se elige por tono con
        validación cruzada agrupada por hablante (AUC)."""
        models = {}
        for tone in np.unique(tones).tolist():
            rows = tones == tone
            scaler = StandardScaler().fit(features[rows])
            scaled = scaler.transform(features[rows])
            if c_by_tone is None:
                splits = list(GroupKFold(n_splits=c_folds).split(scaled, labels[rows], speakers[rows]))
                classifier = LogisticRegressionCV(Cs=C_GRID, l1_ratios=(0,), cv=splits, scoring="roc_auc",
                                                  max_iter=MAX_ITER, use_legacy_attributes=False)
            else:
                classifier = LogisticRegression(C=c_by_tone[tone], max_iter=MAX_ITER)
            classifier.fit(scaled, labels[rows])
            models[tone] = ToneModel(mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_[0],
                                     intercept=float(classifier.intercept_[0]),
                                     c=float(np.ravel(classifier.C_)[0]) if c_by_tone is None else c_by_tone[tone])
        return cls(models)

    def chosen_c(self) -> dict[int, float]:
        return {tone: model.c for tone, model in self.models.items()}

    def score(self, features: np.ndarray, tones: np.ndarray) -> np.ndarray:
        """p(tono mal dicho) de cada fila; NaN si su tono esperado no tiene modelo."""
        scores = np.full(len(features), np.nan)
        for tone, model in self.models.items():
            rows = tones == tone
            if rows.any():
                scores[rows] = model.error_probability(features[rows])
        return scores

    def state_dict(self) -> dict:
        return {str(tone): {"mean": model.mean.tolist(), "scale": model.scale.tolist(), "coef": model.coef.tolist(),
                            "intercept": model.intercept, "c": model.c}
                for tone, model in self.models.items()}

    @classmethod
    def from_state_dict(cls, state: dict) -> ToneErrorDetector:
        return cls({int(tone): ToneModel(mean=np.array(values["mean"]), scale=np.array(values["scale"]),
                                         coef=np.array(values["coef"]), intercept=float(values["intercept"]),
                                         c=float(values["c"]))
                    for tone, values in state.items()})


def out_of_fold_scores(features: np.ndarray, tones: np.ndarray, labels: np.ndarray, speakers: np.ndarray,
                       n_folds: int) -> np.ndarray:
    """p(tono mal dicho) de cada fila con un detector que no vio a su hablante. Cada fold elige su
    propia C, para que ni el ajuste ni la elección de C hayan visto las filas que puntúa."""
    scores = np.full(len(features), np.nan)
    for fit_rows, held_rows in GroupKFold(n_splits=n_folds).split(features, labels, speakers):
        detector = ToneErrorDetector.fit(features[fit_rows], tones[fit_rows], labels[fit_rows], speakers[fit_rows])
        scores[held_rows] = detector.score(features[held_rows], tones[held_rows])
    return scores


def false_alarm_threshold(scores: np.ndarray, labels: np.ndarray, false_alarm: float) -> float:
    """Umbral de p(tono mal dicho) por encima del cual se avisa a ``false_alarm`` de las sílabas bien dichas."""
    return float(np.quantile(scores[~labels], 1 - false_alarm))
