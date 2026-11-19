"""Baseline sobre Tone Perfect: features de F0 + LightGBM, dejando fuera un hablante por fold.

Con 6 hablantes cada fold es un hablante, así que la media entre folds es la exactitud
esperada con una voz nueva y su dispersión importa tanto como la media.

Conjuntos de features:
  - base:         contour_features sin la duración de la sílaba. Es el de referencia: la
                  duración separa por sí sola T3 (larga) de T4 (corta) en Tone Perfect y no
                  se transfiere a hablantes con otro ritmo (con ella, el 34 % de los T3 de
                  MV1 sale como T4).
  - con_duracion: base + duración, solo como comparativa del atajo.

Salidas en outputs/baseline_toneperfect/<method>/: metrics.json, oof_<set>.parquet
(probabilidades out-of-fold, para calibración), confusion_<set>.png, importancia_<set>.csv.

Uso: uv run python scripts/21_baseline_toneperfect.py --method praat
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tonos.evaluation import save_confusion, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="praat")
ap.add_argument("--feats", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/baseline_toneperfect")
args = ap.parse_args()

df = pd.read_parquet(args.feats or f"data/processed/feats_toneperfect_{args.method}.parquet")
out = Path(args.out) / args.method
out.mkdir(parents=True, exist_ok=True)

META = {"syllable", "tone", "speaker", "gender"}
all_features = [c for c in df.columns if c not in META]
FEATURE_SETS = {
    "base": [c for c in all_features if c != "dur"],
    "con_duracion": all_features,
}
TONES = [1, 2, 3, 4]
SPEAKERS = sorted(df.speaker.unique())
y = df.tone.to_numpy()

# Los mismos hiperparámetros que el baseline de bisílabos de AISHELL-3 (script 20, ya retirado).
params = dict(objective="multiclass", n_estimators=600, learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=args.seed, verbose=-1, n_jobs=-1)


results = {}
for name, cols in FEATURE_SETS.items():
    proba = np.zeros((len(df), len(TONES)))
    importances = np.zeros(len(cols))
    for speaker in SPEAKERS:
        test = (df.speaker == speaker).to_numpy()
        model = lgb.LGBMClassifier(**params).fit(df.loc[~test, cols], y[~test])
        proba[test] = model.predict_proba(df.loc[test, cols])
        importances += model.booster_.feature_importance("gain")
    pred = np.array(TONES)[proba.argmax(1)]

    per_speaker = {s: summarize(y[df.speaker == s], pred[df.speaker == s], TONES) for s in SPEAKERS}
    speaker_acc = [r["accuracy"] for r in per_speaker.values()]
    res = {
        "global": summarize(y, pred, TONES),
        "accuracy_hablante_media_std": [float(np.mean(speaker_acc)), float(np.std(speaker_acc))],
        "por_hablante": per_speaker,
        "por_genero": {g: summarize(y[df.gender == g], pred[df.gender == g], TONES) for g in sorted(df.gender.unique())},
        "n_features": len(cols),
    }
    results[name] = res
    g = res["global"]
    print(f"\n=== {args.method} | {name} ({len(cols)} features) ===")
    print(f"accuracy {g['accuracy']:.3f} (hablantes {np.mean(speaker_acc):.3f} ± {np.std(speaker_acc):.3f}) | "
          f"macro-F1 {g['macro_f1']:.3f}")
    print("recall por tono:", g["recall_por_tono"])
    print("por hablante:", {s: round(r["accuracy"], 3) for s, r in per_speaker.items()})

    oof = df[["syllable", "speaker", "gender", "tone"]].copy()
    for i, t in enumerate(TONES):
        oof[f"p{t}"] = proba[:, i]
    oof.to_parquet(out / f"oof_{name}.parquet", index=False)
    pd.Series(importances / len(SPEAKERS), index=cols).sort_values(ascending=False).to_csv(
        out / f"importancia_{name}.csv")
    save_confusion(y, pred, TONES, f"{args.method} · {name} · acc {g['accuracy']:.3f}", out / f"confusion_{name}.png")

(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nResultados en {out}/")
