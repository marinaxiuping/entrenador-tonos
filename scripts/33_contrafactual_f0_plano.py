"""Contrafactual: ¿HuBERT clasifica el tono por el pitch o por el léxico?

Se resintetizan con WORLD frases de AISHELL-3 en dos condiciones y se pasan por los mismos
clasificadores lineales de HuBERT del script 31 (entrenados con audio original de los
otros folds, así que cada frase la clasifica un modelo que no ha visto a su hablante):
  - resintesis: WORLD sin tocar el F0. Control de los artefactos del vocoder.
  - f0_plano: WORLD con el F0 de los frames con voz fijado a la media de la frase. No queda
    información tonal; el contenido segmental (qué se dice) se conserva.
Si con F0 plano HuBERT sigue cerca de la línea base léxica, acierta por lo que se dice; si
cae hacia el azar (0,25), depende del pitch.

Comprobación de la manipulación: desviación típica del F0 (Praat, en semitonos) de cada
frase en las tres condiciones; con F0 plano tiene que ser casi cero.

Salidas en outputs/escalera_dominio/contrafactual_f0_plano/: metrics.json,
comparacion.csv, predicciones.parquet y ejemplos/ (wav de las tres condiciones).

Uso: uv run python scripts/33_contrafactual_f0_plano.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from tonos.aishell3 import TONES, lexical_tone_baseline, load_syllables, speaker_fold_ids
from tonos.align import SR, load_16k
from tonos.embeddings import SpeechEncoder
from tonos.evaluation import speaker_bootstrap_accuracy
from tonos.f0 import FMAX, FMIN, f0_praat, hz_to_st
from tonos.linear_probe import LinearProbe
from tonos.resynthesis import flatten_voiced_f0, resynthesize_with_f0

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--embeddings", default="data/interim/embeddings/aishell3_chinese-hubert-base")
ap.add_argument("--hubert-metrics", default="outputs/escalera_dominio/aishell3_hubert/metrics.json",
                help="capa elegida por fold y C del script 31")
ap.add_argument("--model", default="TencentGameMate/chinese-hubert-base")
ap.add_argument("--revision", default="fce0375452b1dd6c080ac3248d423d4d037bc831")
ap.add_argument("--utterances-per-speaker", type=int, default=60)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--examples", type=int, default=3)
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/escalera_dominio/contrafactual_f0_plano")
args = ap.parse_args()

CONDITIONS = ["original", "resintesis", "f0_plano"]
TRANSFORMS = {"resintesis": lambda f0: f0, "f0_plano": flatten_voiced_f0}
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
(out / "ejemplos").mkdir(parents=True, exist_ok=True)

syllables = load_syllables(args.syllables, args.contours)
table, labels, evaluated = syllables.table, syllables.labels, syllables.evaluated
fold_of = speaker_fold_ids(table, args.folds, args.seed)
hubert = json.loads(Path(args.hubert_metrics).read_text(encoding="utf-8"))
layer_of_fold = {int(fold): layer for fold, layer in hubert["capa_elegida_por_fold"].items()}

rng = np.random.default_rng(args.seed)
sampled_utterances = sorted(
    utt for _, utts in table.groupby("speaker").utt.unique().items()
    for utt in rng.choice(utts, size=min(args.utterances_per_speaker, len(utts)), replace=False))
subset_rows = np.flatnonzero(table.utt.isin(sampled_utterances).to_numpy() & evaluated)
print(f"{len(sampled_utterances)} frases, {subset_rows.size} sílabas evaluadas | dispositivo {device}")


def wav_path(utt: str) -> Path:
    return Path(args.root) / "train" / "wav" / utt[:7] / f"{utt}.wav"


def counterfactual_waveforms(utt: str) -> dict[str, np.ndarray]:
    waveform = load_16k(str(wav_path(utt)))
    return {"original": waveform, **{name: resynthesize_with_f0(waveform, SR, FMIN, FMAX, transform)
                                     for name, transform in TRANSFORMS.items()}}


def f0_spread_st(waveform: np.ndarray) -> float:
    """Desviación típica del F0 de la frase en semitonos (Praat, rango genérico)."""
    f0 = f0_praat(waveform)[0]
    voiced = f0[np.isfinite(f0)]
    return float(np.std(hz_to_st(voiced, np.exp(np.mean(np.log(voiced)))))) if voiced.size > 1 else np.nan


print("Resíntesis con WORLD...")
waveforms = dict(zip(sampled_utterances, Parallel(n_jobs=args.jobs)(
    delayed(counterfactual_waveforms)(utt) for utt in sampled_utterances)))
spread = pd.DataFrame([{condition: f0_spread_st(w[condition]) for condition in CONDITIONS}
                       for w in waveforms.values()])
for utt in sampled_utterances[:args.examples]:
    for condition in CONDITIONS:
        sf.write(out / "ejemplos" / f"{utt}_{condition}.wav", waveforms[utt][condition], SR)

# --- Embeddings de cada condición, solo de las capas elegidas ---
needed_layers = sorted(set(layer_of_fold.values()))
encoder = SpeechEncoder(args.model, args.revision, device)
subset_table = table.loc[subset_rows]
embeddings = {condition: {layer: np.zeros((subset_rows.size, encoder.model.config.hidden_size), np.float32)
                          for layer in needed_layers} for condition in CONDITIONS}
positions = pd.Series(np.arange(subset_rows.size), index=subset_rows)
for utt, rows in tqdm(subset_table.groupby("utt"), desc="embeddings"):
    spans = list(zip(rows.start, rows.end))
    for condition in CONDITIONS:
        means = encoder.segment_means(waveforms[utt][condition], spans)
        for layer in needed_layers:
            embeddings[condition][layer][positions[rows.index].to_numpy()] = means[:, layer]

# --- Clasificadores del script 31, reentrenados con audio original de los otros folds ---
predicted = {condition: np.zeros(subset_rows.size, dtype=int) for condition in CONDITIONS}
for fold, layer in layer_of_fold.items():
    train_rows = np.flatnonzero(evaluated & (fold_of != fold))
    x_train = torch.from_numpy(np.load(Path(args.embeddings) / f"layer_{layer:02d}.npy",
                                       mmap_mode="r")[train_rows].astype(np.float32)).to(device)
    probe = LinearProbe(len(TONES), hubert["C"]).fit(x_train, torch.from_numpy(labels[train_rows]).long().to(device))
    in_fold = fold_of[subset_rows] == fold
    for condition in CONDITIONS:
        x_test = torch.from_numpy(embeddings[condition][layer][in_fold]).to(device)
        predicted[condition][in_fold] = np.array(TONES)[probe.logits(x_test).argmax(1).cpu().numpy()]
    del x_train

evaluated_table = table[evaluated].reset_index(drop=True)
lexical = pd.Series(lexical_tone_baseline(evaluated_table, fold_of[evaluated]), index=np.flatnonzero(evaluated))
predicted["lexico_sin_audio"] = lexical[subset_rows].to_numpy()

truth = subset_table.tone.to_numpy()
correct = pd.DataFrame({name: pred == truth for name, pred in predicted.items()})
boot = speaker_bootstrap_accuracy(correct, subset_table.speaker.to_numpy(), args.resamples, args.seed)
original_column = correct.columns.get_loc("original")
comparison = pd.DataFrame([{
    "condicion": name,
    "accuracy": round(float(correct[name].mean()), 4),
    "ic95_inf": round(float(np.percentile(boot[:, i], 2.5)), 4),
    "ic95_sup": round(float(np.percentile(boot[:, i], 97.5)), 4),
    "dif_vs_original": round(float(correct[name].mean() - correct.original.mean()), 4),
    "dif_ic95_inf": round(float(np.percentile(boot[:, i] - boot[:, original_column], 2.5)), 4),
    "dif_ic95_sup": round(float(np.percentile(boot[:, i] - boot[:, original_column], 97.5)), 4),
} for i, name in enumerate(correct.columns)])
comparison.to_csv(out / "comparacion.csv", index=False)

by_tone = correct.groupby(truth).mean().round(4)
agreement_with_lexicon = {condition: round(float((predicted[condition] == predicted["lexico_sin_audio"]).mean()), 4)
                          for condition in CONDITIONS}
results = {
    "n_frases": len(sampled_utterances),
    "n_silabas": int(subset_rows.size),
    "capa_por_fold": layer_of_fold,
    "accuracy_por_condicion": comparison.set_index("condicion").accuracy.to_dict(),
    "accuracy_por_tono": {str(t): row.to_dict() for t, row in by_tone.iterrows()},
    "coincidencia_con_lexico": agreement_with_lexicon,
    "dispersion_f0_st_mediana": spread.median().round(3).to_dict(),
}
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
predictions = subset_table[["speaker", "utt", "word_idx", "syl_idx", "hanzi", "pinyin", "tone"]].reset_index(drop=True)
for name, pred in predicted.items():
    predictions[f"pred_{name}"] = pred
predictions.to_parquet(out / "predicciones.parquet", index=False)

print("\nDispersión del F0 por frase (st, mediana):", results["dispersion_f0_st_mediana"])
print(comparison.to_string(index=False))
print("\nAccuracy por tono:\n", by_tone.to_string())
print("\nCoincidencia de la predicción con la línea base léxica:", agreement_with_lexicon)
print(f"\nResultados en {out}/")
