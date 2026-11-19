"""Análisis de las etiquetas de AISHELL-3 para la decisión de bisílabos.

- Palabras prosódicas (límites % y $ en label_train-set.txt): cuántas son bisílabas.
- Balance de pares de tonos en los bisílabos y bisílabos por hablante.
- Tono realizado (etiqueta) frente a canónico (pypinyin): ¿las etiquetas recogen el sandhi?

Uso: uv run python scripts/11_analizar_etiquetas_aishell3.py
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from pypinyin import Style, pinyin

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="data/raw/aishell3/train/label_train-set.txt")
ap.add_argument("--out", default="outputs/aishell3")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

SPLIT = re.compile(r"\s*[%$]\s*")


def tone(syl: str) -> int:
    return int(syl[-1]) if syl[-1].isdigit() else 5


rows, bad = [], 0
for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    utt, py, hz = line.split("|")
    py_words = [w.split() for w in SPLIT.split(py) if w.strip()]
    hz_words = [w.strip() for w in SPLIT.split(hz) if w.strip()]
    if len(py_words) != len(hz_words) or any(len(p) != len(h) for p, h in zip(py_words, hz_words)):
        bad += 1
        continue
    for i, (p, h) in enumerate(zip(py_words, hz_words)):
        rows.append({"utt": utt, "speaker": utt[:7], "idx": i, "hanzi": h, "pinyin": " ".join(p),
                     "n_syl": len(p), "tones": "".join(str(tone(s)) for s in p)})

w = pd.DataFrame(rows)
print(f"Frases descartadas por desalineación pinyin/hanzi: {bad}")
print(f"Palabras prosódicas: {len(w)} | hablantes: {w.speaker.nunique()}")
print("\nPalabras por nº de sílabas (%):")
print((w.n_syl.value_counts(normalize=True).sort_index().head(6) * 100).round(1))

bi = w[w.n_syl == 2].copy()
print(f"\nBisílabos: {len(bi)}  | tipos (hanzi) distintos: {bi.hanzi.nunique()}")
print("\nPares de tonos realizados en bisílabos (cuenta):")
pairs = bi.tones.value_counts()
grid = pd.DataFrame(0, index=list("12345"), columns=list("12345"))
for k, v in pairs.items():
    grid.loc[k[0], k[1]] = v
print(grid.rename_axis(index="σ1", columns="σ2"))

per_spk = bi.groupby("speaker").size()
print(f"\nBisílabos por hablante: mediana {per_spk.median():.0f}, min {per_spk.min()}, max {per_spk.max()}")

# --- Realizado vs canónico ---
canon = []
for h in bi.hanzi:
    c = [s[0] for s in pinyin(h, style=Style.TONE3, neutral_tone_with_five=True)]
    canon.append("".join(str(tone(s)) for s in c) if len(c) == 2 else None)
bi["canon"] = canon
bi = bi.dropna(subset=["canon"])


def category(r):
    if r.tones == r.canon:
        return "igual"
    if r.canon == "33" and r.tones == "23":
        return "sandhi 3+3→2+3"
    if r.hanzi[0] in "一不" and r.tones[0] != r.canon[0]:
        return "sandhi 一/不"
    if "5" in r.tones and r.tones.replace("5", "") != r.canon.replace("5", ""):
        return "neutro en la etiqueta"
    if "5" in r.canon:
        return "neutro canónico, tono pleno en la etiqueta"
    return "otro"


bi["cat"] = bi.apply(category, axis=1)
print("\nRealizado (etiqueta) vs canónico (pypinyin) en bisílabos (%):")
print((bi.cat.value_counts(normalize=True) * 100).round(2))
c33 = (bi.canon == "33").sum()
r33 = ((bi.canon == "33") & (bi.tones == "33")).sum()
print(f"\nCanónicos 3+3: {c33} → etiquetados 3+3 sin sandhi: {r33} ({100 * r33 / max(c33, 1):.1f}%)")
print("\nEjemplos de 'otro':")
print(bi[bi.cat == "otro"].sample(min(15, (bi.cat == "otro").sum()), random_state=0)[["hanzi", "pinyin", "canon"]].to_string(index=False))

w.to_csv(out / "palabras_prosodicas.csv", index=False)
bi.to_csv(out / "bisilabos.csv", index=False)
grid.to_csv(out / "pares_tonos.csv")
per_spk.sort_values(ascending=False).to_csv(out / "bisilabos_por_hablante.csv")
print(f"\nGuardado en {out}/")
