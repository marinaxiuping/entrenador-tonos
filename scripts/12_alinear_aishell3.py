"""Alinea a nivel de sílaba las frases de los hablantes descargados y extrae los bisílabos.

Salidas (en --out):
  alineamiento_silabas.csv : una fila por sílaba (utt, palabra, tiempos, score CTC, tono)
  bisilabos_alineados.csv  : una fila por bisílabo prosódico (tiempos de cada sílaba)
  revision/                : N bisílabos al azar (wav con 150 ms de contexto) + figura para revisarlos a mano
  contornos_por_tono.png   : contorno medio de F0 por tono de las sílabas alineadas (control automático)

Uso: uv run python scripts/12_alinear_aishell3.py --speakers SSB0544 SSB0502 --n-review 30
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from tonos.align import SR, SyllableAligner, fill_gaps, load_16k
from tonos.f0 import f0_pyin, fixed_length_contour, hz_to_st, speaker_reference_hz

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # hanzi en las figuras

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--speakers", nargs="*", default=[])
ap.add_argument("--speakers-file", help="TSV con columna 'speaker'")
ap.add_argument("--out", default="outputs/aishell3/alineamiento")
ap.add_argument("--n-review", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-fill", action="store_true", help="usa los tramos CTC crudos (sin rellenar huecos)")
args = ap.parse_args()
if args.speakers_file:
    args.speakers += pd.read_csv(args.speakers_file, sep="\t").speaker.tolist()
root, out = Path(args.root), Path(args.out)
(out / "revision").mkdir(parents=True, exist_ok=True)

SPLIT = re.compile(r"\s*[%$]\s*")

# --- Etiquetas con límites de palabra prosódica ---
utts = []
for line in (root / "train/label_train-set.txt").read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    utt, py, hz = line.split("|")
    if utt[:7] not in args.speakers:
        continue
    py_words = [w.split() for w in SPLIT.split(py) if w.strip()]
    hz_words = [w.strip() for w in SPLIT.split(hz) if w.strip()]
    if len(py_words) != len(hz_words) or any(len(p) != len(h) for p, h in zip(py_words, hz_words)):
        continue
    wav = root / "train/wav" / utt[:7] / f"{utt}.wav"
    if wav.exists():
        utts.append((utt, py_words, hz_words, wav))
print(f"Frases a alinear: {len(utts)}")

# --- Alineamiento ---
al = SyllableAligner()
syl_rows = []
for utt, py_words, hz_words, wav in tqdm(utts):
    y = load_16k(str(wav))
    sylls = [s for w in py_words for s in w]
    raw = al.align(y, sylls)
    spans = raw if args.no_fill else fill_gaps(raw)
    k = 0
    for wi, (pw, hw) in enumerate(zip(py_words, hz_words)):
        for si, (s, h) in enumerate(zip(pw, hw)):
            sp = spans[k]
            k += 1
            syl_rows.append({"utt": utt, "speaker": utt[:7], "path": str(wav), "word_idx": wi, "syl_idx": si,
                             "word_len": len(pw), "hanzi": h, "pinyin": s,
                             "tone": int(s[-1]) if s[-1].isdigit() else 5,
                             "start": sp.start, "end": sp.end, "score": sp.score})
syl = pd.DataFrame(syl_rows)
syl["dur"] = syl.end - syl.start
syl.to_csv(out / "alineamiento_silabas.csv", index=False)

print("\nDuración de sílaba (s):", syl.dur.describe().round(3).to_dict())
print("Score CTC por sílaba: mediana %.3f | p10 %.3f" % (syl.score.median(), syl.score.quantile(0.1)))
print(f"Sílabas < 50 ms (sospechosas): {(syl.dur < 0.05).mean() * 100:.1f}%")

# --- Bisílabos ---
b = syl[syl.word_len == 2]
bi = (b.sort_values(["utt", "word_idx", "syl_idx"])
      .groupby(["utt", "word_idx"])
      .agg(speaker=("speaker", "first"), path=("path", "first"), hanzi=("hanzi", "".join),
           pinyin=("pinyin", " ".join), tones=("tone", lambda t: "".join(map(str, t))),
           start1=("start", "first"), end1=("end", "first"), start2=("start", "last"), end2=("end", "last"),
           min_score=("score", "min"))
      .reset_index())
bi.to_csv(out / "bisilabos_alineados.csv", index=False)
print(f"\nBisílabos alineados: {len(bi)}")

# --- Control automático: contorno medio de F0 por tono (sílabas no neutras) ---
rng = np.random.default_rng(args.seed)
sample = (syl[syl.tone < 5].sample(frac=1, random_state=args.seed)
          .groupby("tone").head(300).sort_values(["path", "start"]))
cache: dict[str, np.ndarray] = {}
f0s = []
for r in tqdm(sample.itertuples(), total=len(sample), desc="F0"):
    if r.path not in cache:
        cache = {r.path: load_16k(r.path)}
    seg = cache[r.path][int(r.start * SR): int(r.end * SR)]
    f0s.append(f0_pyin(seg)[0] if seg.size > 1024 else np.array([np.nan]))
sample = sample.assign(i=range(len(sample)))
refs = {s: speaker_reference_hz([f0s[i] for i in g.i]) for s, g in sample.groupby("speaker")}
C = np.stack([fixed_length_contour(hz_to_st(f, refs[s])) for f, s in zip(f0s, sample.speaker)])
ok = ~np.isnan(C).all(1)
half = C.shape[1] // 2
slope = pd.DataFrame({"tone": sample.tone.to_numpy()[ok],
                      "inicio→fin (st)": C[ok, -1] - C[ok, 0],
                      "1ª mitad (st)": C[ok, half] - C[ok, 0],
                      "2ª mitad (st)": C[ok, -1] - C[ok, half]})
print("\nMovimiento medio de F0 por tono (semitonos):")
print(slope.groupby("tone").mean().round(2))

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
t = np.linspace(0, 1, C.shape[1])
for tone, ax in zip(range(1, 5), axes):
    m = (sample.tone == tone).to_numpy() & ~np.isnan(C).all(1)
    for c in C[m][:80]:
        ax.plot(t, c, color="tab:gray", alpha=0.15, lw=0.7)
    ax.plot(t, np.nanmean(C[m], 0), "k", lw=2.5)
    ax.set_title(f"Tono {tone} (n={m.sum()})")
axes[0].set_ylabel("semitonos")
fig.suptitle("Sílabas de AISHELL-3 alineadas con MMS_FA: contorno F0 por tono realizado")
fig.tight_layout()
fig.savefig(out / "contornos_por_tono.png", dpi=120)

# --- Material de revisión manual ---
rev = bi.sample(min(args.n_review, len(bi)), random_state=args.seed).reset_index(drop=True)
n = len(rev)
fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 2.2 * ((n + 2) // 3)))
for i, (r, ax) in enumerate(zip(rev.itertuples(), axes.flat)):
    y = load_16k(r.path)
    a, z = max(0, r.start1 - 0.15), min(len(y) / SR, r.end2 + 0.15)
    seg = y[int(a * SR): int(z * SR)]
    sf.write(out / "revision" / f"{i:02d}_{r.utt}_{r.tones}.wav", seg, SR)
    tt = np.arange(seg.size) / SR + a
    ax.plot(tt, seg, lw=0.4, color="0.6")
    for x0, x1, c in [(r.start1, r.end1, "tab:blue"), (r.start2, r.end2, "tab:orange")]:
        ax.axvspan(x0, x1, color=c, alpha=0.25)
    ax2 = ax.twinx()
    f0, ft = f0_pyin(seg)
    ax2.plot(ft + a, f0, "k.", ms=2)
    ax.set_title(f"{i:02d} {r.hanzi} {r.pinyin} (min score {r.min_score:.2f})", fontsize=9)
    ax.set_yticks([])
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(out / "revision" / "revision_bisilabos.png", dpi=110)
rev.to_csv(out / "revision" / "revision.csv", index=False)
print(f"\nSalidas en {out}/  ->  escucha revision/*.wav y mira revision_bisilabos.png")
