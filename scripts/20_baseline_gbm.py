"""Baseline: features de F0 + LightGBM, validación cruzada por hablante (5 folds equilibrados por género).

Compara dos conjuntos de features:
  - propia:   solo la sílaba (+ posición en la palabra y en la frase)
  - contexto: + la otra sílaba del bisílabo y diferencias entre ambas

Salidas en outputs/baseline/<method>/: metrics.json, oof_<set>.parquet (probabilidades
out-of-fold, para calibración), confusion_<set>.png, importancia_<set>.csv.

Uso: uv run python scripts/20_baseline_gbm.py --method pyin
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from tonos.splits import speaker_folds

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="pyin")
ap.add_argument("--feats", default=None)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/baseline")
args = ap.parse_args()

df = pd.read_parquet(args.feats or f"data/processed/feats_{args.method}.parquet")
out = Path(args.out) / args.method
out.mkdir(parents=True, exist_ok=True)

folds = speaker_folds(df[["speaker", "gender"]].drop_duplicates(), args.folds, args.seed)
df["fold"] = df.speaker.map(folds)

META = {"speaker", "gender", "utt", "word_idx", "hanzi", "pinyin", "word_tones", "tone", "score", "fold"}
ctx_cols = [c for c in df.columns if c.startswith("ctx_") or c.startswith("diff_")]
FEATURE_SETS = {
    "propia": [c for c in df.columns if c not in META and c not in ctx_cols],
    "contexto": [c for c in df.columns if c not in META],
}
TONES = [1, 2, 3, 4, 5]
y = df.tone.to_numpy()

params = dict(objective="multiclass", n_estimators=600, learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=args.seed, verbose=-1, n_jobs=-1)


def metrics(y_true, y_pred, mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    full = y_true < 5
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1_5": f1_score(y_true, y_pred, labels=TONES, average="macro", zero_division=0),
        "macro_f1_1a4": f1_score(y_true[full], y_pred[full], labels=TONES[:4], average="macro", zero_division=0),
        "recall_por_tono": dict(zip(map(str, TONES),
                                    recall_score(y_true, y_pred, labels=TONES, average=None, zero_division=0).round(4).tolist())),
    }


results = {}
for name, cols in FEATURE_SETS.items():
    proba = np.zeros((len(df), len(TONES)))
    importances = np.zeros(len(cols))
    for k in range(args.folds):
        tr, te = df.fold != k, df.fold == k
        m = lgb.LGBMClassifier(**params).fit(df.loc[tr, cols], y[tr])
        proba[te.to_numpy()] = m.predict_proba(df.loc[te, cols])
        importances += m.booster_.feature_importance("gain")
    pred = np.array(TONES)[proba.argmax(1)]

    # Exactitud por palabra: las dos sílabas bien
    word_ok = (pd.Series(pred == y).groupby([df.utt, df.word_idx]).all())
    per_fold = [accuracy_score(y[df.fold == k], pred[df.fold == k]) for k in range(args.folds)]
    res = {
        "global": metrics(y, pred),
        "accuracy_palabra": float(word_ok.mean()),
        "accuracy_por_fold": [round(a, 4) for a in per_fold],
        "accuracy_fold_media_std": [float(np.mean(per_fold)), float(np.std(per_fold))],
        "por_genero": {g: metrics(y, pred, (df.gender == g).to_numpy()) for g in sorted(df.gender.unique())},
        "por_posicion": {f"σ{p + 1}": metrics(y, pred, (df.pos == p).to_numpy()) for p in (0, 1)},
        "n_features": len(cols),
    }
    results[name] = res
    g = res["global"]
    print(f"\n=== {args.method} | {name} ({len(cols)} features) ===")
    print(f"accuracy {g['accuracy']:.3f} (folds {np.mean(per_fold):.3f} ± {np.std(per_fold):.3f}) | "
          f"macro-F1 {g['macro_f1_5']:.3f} | macro-F1 T1–T4 {g['macro_f1_1a4']:.3f} | palabra {res['accuracy_palabra']:.3f}")
    print("recall por tono:", g["recall_por_tono"])
    print("por género:", {k: round(v["accuracy"], 3) for k, v in res["por_genero"].items()},
          "| por posición:", {k: round(v["accuracy"], 3) for k, v in res["por_posicion"].items()})

    oof = df[["speaker", "gender", "utt", "word_idx", "pos", "tone", "fold"]].copy()
    for i, t in enumerate(TONES):
        oof[f"p{t}"] = proba[:, i]
    oof.to_parquet(out / f"oof_{name}.parquet", index=False)
    pd.Series(importances / args.folds, index=cols).sort_values(ascending=False).to_csv(out / f"importancia_{name}.csv")

    cm = confusion_matrix(y, pred, labels=TONES, normalize="true")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black", fontsize=9)
    ax.set_xticks(range(5), [f"T{t}" for t in TONES])
    ax.set_yticks(range(5), [f"T{t}" for t in TONES])
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(f"{args.method} · {name} · acc {g['accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{name}.png", dpi=120)
    plt.close(fig)

(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nResultados en {out}/")
