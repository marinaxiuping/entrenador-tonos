"""Temperature scaling sobre las probabilidades out-of-fold del baseline de Tone Perfect.

Evaluación sin fuga: la temperatura de cada hablante se ajusta con las probabilidades
out-of-fold de los otros cinco, igual que en el baseline. La temperatura final, ajustada
con todos, es la que se usará en la app y se guarda en temperature.json.

Salidas en outputs/calibracion_toneperfect/<method>/<set>/: metrics.json, temperature.json,
oof_calibrado.parquet y fiabilidad.png (diagrama de fiabilidad antes y después).

Uso: uv run python scripts/22_calibrar_toneperfect.py --method praat --set base
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tonos.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    probabilities_to_logits,
    reliability_bins,
)

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="praat")
ap.add_argument("--set", dest="feature_set", default="base")
ap.add_argument("--baseline-dir", default="outputs/baseline_toneperfect")
ap.add_argument("--out", default="outputs/calibracion_toneperfect")
ap.add_argument("--bins", type=int, default=15)
args = ap.parse_args()

oof = pd.read_parquet(Path(args.baseline_dir) / args.method / f"oof_{args.feature_set}.parquet")
out = Path(args.out) / args.method / args.feature_set
out.mkdir(parents=True, exist_ok=True)

proba_cols = sorted(c for c in oof.columns if c.startswith("p") and c[1:].isdigit())
tones = [int(c[1:]) for c in proba_cols]
labels = oof.tone.map({t: i for i, t in enumerate(tones)}).to_numpy()
proba = oof[proba_cols].to_numpy()
logits = probabilities_to_logits(proba)


def summarize(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "nll": negative_log_likelihood(p, y),
        "ece": expected_calibration_error(p, y, args.bins),
        "brier": brier_score(p, y),
        "confianza_media": float(p.max(axis=1).mean()),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


calibrated = np.zeros_like(proba)
speaker_temperatures = {}
for speaker in sorted(oof.speaker.unique()):
    test = (oof.speaker == speaker).to_numpy()
    speaker_temperatures[speaker] = fit_temperature(logits[~test], labels[~test])
    calibrated[test] = apply_temperature(logits[test], speaker_temperatures[speaker])

per_speaker = {}
for speaker in speaker_temperatures:
    mask = (oof.speaker == speaker).to_numpy()
    per_speaker[speaker] = {"sin_calibrar": summarize(proba[mask], labels[mask]),
                            "calibrado": summarize(calibrated[mask], labels[mask])}

final_temperature = fit_temperature(logits, labels)
results = {
    "sin_calibrar": summarize(proba, labels),
    "calibrado": summarize(calibrated, labels),
    "temperatura_por_hablante": {s: round(t, 4) for s, t in speaker_temperatures.items()},
    "temperatura_final": final_temperature,
    "por_hablante": per_speaker,
}

print(f"=== {args.method} | {args.feature_set} ===")
print("temperatura por hablante:", results["temperatura_por_hablante"])
print(f"temperatura final: {final_temperature:.3f}")
for stage in ("sin_calibrar", "calibrado"):
    r = results[stage]
    print(f"{stage:13s} NLL {r['nll']:.4f} | ECE {r['ece']:.4f} | Brier {r['brier']:.4f} | "
          f"confianza media {r['confianza_media']:.3f} | accuracy {r['accuracy']:.3f}")

(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
(out / "temperature.json").write_text(
    json.dumps({"temperature": final_temperature, "tones": tones, "feature_set": args.feature_set,
                "method": args.method}, indent=2), encoding="utf-8")
calibrated_oof = oof.drop(columns=proba_cols)
for i, column in enumerate(proba_cols):
    calibrated_oof[column] = calibrated[:, i]
calibrated_oof.to_parquet(out / "oof_calibrado.parquet", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, (stage, p) in zip(axes, [("sin calibrar", proba), ("calibrado", calibrated)]):
    bins = reliability_bins(p, labels, args.bins)
    filled = bins["count"] > 0
    centers = (np.arange(args.bins) + 0.5) / args.bins
    ax.bar(centers[filled], bins["accuracy"][filled], width=1 / args.bins, edgecolor="k", alpha=0.7,
           label="accuracy del bin")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibración perfecta")
    ax.plot(centers[filled], bins["confidence"][filled], "o", color="tab:red", ms=4,
            label="confianza media del bin")
    for x, n in zip(centers[filled], bins["count"][filled]):
        ax.text(x, 0.02, str(n), ha="center", fontsize=6, rotation=90)
    ece = expected_calibration_error(p, labels, args.bins)
    ax.set_title(f"{stage} · ECE {ece:.4f}")
    ax.set_xlabel("confianza (clase predicha)")
axes[0].set_ylabel("accuracy")
axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.9))
fig.text(0.5, 0.005, "nº de muestras por bin en la base de cada barra", ha="center", fontsize=7)
fig.suptitle(f"Diagrama de fiabilidad – {args.method} · {args.feature_set}")
fig.tight_layout()
fig.savefig(out / "fiabilidad.png", dpi=120)
print(f"\nResultados en {out}/")
