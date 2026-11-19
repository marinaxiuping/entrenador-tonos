"""Contrafactual del caso de uso de la app: se cambia el tono de UNA sílaba, ¿lo detecta el modelo?

En cada frase (las mismas que el script 33) se elige una sílaba al azar y se le impone con
WORLD el contorno de otro tono; el resto de la frase no se toca. El contorno impuesto es
el de una sílaba real de ese tono elegida al azar entre los hablantes de los otros folds,
no la media del tono: la media es un prototipo limpio que favorecería a los modelos de F0
frente a la variabilidad del habla natural. Es lo que pasa cuando un alumno dice bien una frase salvo un tono: un buen juez debe seguir al
tono impuesto y no al que corresponde a la palabra.

Condiciones: original, resintesis (WORLD sin cambios, control del vocoder) y tono_cambiado.
Modelos, cada uno entrenado sin el fold del hablante y con la misma configuración que en
los scripts 29-31: CNN por sílaba, frase (CNN + BiGRU) y HuBERT + lineal. Más la línea
base léxica, que por construcción se queda siempre con el tono original.

El audio manipulado pasa por el mismo pipeline de F0 que el de entrenamiento, con el
perfil del hablante (rango y referencia) estimado sobre su audio original, como hace la
app con la calibración. Comprobación de la manipulación: distancia (RMSE, st) del contorno
reextraído de la sílaba cambiada al contorno impuesto y al de la resíntesis sin cambios.

Salidas en outputs/escalera_dominio/contrafactual_cambio_tono/: metrics.json,
comparacion.csv, por_par.csv, predicciones.parquet y ejemplos/.

Uso: uv run python scripts/34_contrafactual_cambio_tono.py
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

from tonos.aishell3 import TONES, lexical_tone_baseline, load_syllables, speaker_fold_ids, utterance_bounds
from tonos.align import SR, load_16k
from tonos.cnn import TrainingConfig, contour_inputs, predict_logits, train_cnn
from tonos.context import TaggerConfig, predict_tagger_logits, syllable_inputs, train_tagger
from tonos.embeddings import SpeechEncoder
from tonos.evaluation import speaker_bootstrap_accuracy
from tonos.linear_probe import LinearProbe
from tonos.resynthesis import impose_contour_on_span, resynthesize_with_f0
from tonos.utterance_contours import estimate_speaker_profile, syllable_contours

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--embeddings", default="data/interim/embeddings/aishell3_chinese-hubert-base")
ap.add_argument("--hubert-metrics", default="outputs/escalera_dominio/aishell3_hubert/metrics.json",
                help="capa elegida por fold y C del script 31")
ap.add_argument("--model", default="TencentGameMate/chinese-hubert-base")
ap.add_argument("--revision", default="fce0375452b1dd6c080ac3248d423d4d037bc831")
ap.add_argument("--utterances-per-speaker", type=int, default=60, help="igual que el script 33")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--examples", type=int, default=3)
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/escalera_dominio/contrafactual_cambio_tono")
args = ap.parse_args()

CONDITIONS = ["original", "resintesis", "tono_cambiado"]
MODELS = ["cnn_silaba", "frase", "hubert", "lexico_sin_audio"]
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
(out / "ejemplos").mkdir(parents=True, exist_ok=True)

syllables = load_syllables(args.syllables, args.contours)
table, labels, evaluated = syllables.table, syllables.labels, syllables.evaluated
fold_of = speaker_fold_ids(table, args.folds, args.seed)
donor_pool = {(fold, t): np.flatnonzero(evaluated & (fold_of != fold) & (table.tone == t).to_numpy())
              for fold in range(args.folds) for t in TONES}
n_points = syllables.contours.shape[1]
hubert = json.loads(Path(args.hubert_metrics).read_text(encoding="utf-8"))
layer_of_fold = {int(fold): layer for fold, layer in hubert["capa_elegida_por_fold"].items()}

# --- Mismas frases que el script 33 y, en cada una, la sílaba que se cambia ---
rng = np.random.default_rng(args.seed)
sampled_utterances = sorted(
    utt for _, utts in table.groupby("speaker").utt.unique().items()
    for utt in rng.choice(utts, size=min(args.utterances_per_speaker, len(utts)), replace=False))
sample = table[table.utt.isin(sampled_utterances)]
targets = []
for utt, rows in sample.groupby("utt", sort=False):
    candidates = rows.index[evaluated[rows.index]]
    if candidates.empty:
        continue
    row = int(rng.choice(candidates))
    original_tone = int(table.tone[row])
    imposed_tone = int(rng.choice([t for t in TONES if t != original_tone]))
    targets.append({"utt": utt, "speaker": table.speaker[row], "row": row, "tono_original": original_tone,
                    "tono_impuesto": imposed_tone,
                    "fila_donante": int(rng.choice(donor_pool[int(fold_of[row]), imposed_tone])),
                    "start": table.start[row], "end": table.end[row]})
targets = pd.DataFrame(targets)
sample = sample[sample.utt.isin(targets.utt)]
sample_position = pd.Series(np.arange(len(sample)), index=sample.index)
imposed_contour_of = dict(zip(targets.utt, syllables.contours[targets.fila_donante.to_numpy()]))
print(f"{len(targets)} frases con una sílaba cambiada | dispositivo {device}")


def process_speaker(rows: pd.DataFrame, speaker_targets: pd.DataFrame,
                    imposed_contours: dict[str, np.ndarray]) -> dict:
    """Audio y contornos de las tres condiciones para las frases de un hablante."""
    utterances = list(rows.utt.unique())
    spans = [list(zip(g.start, g.end)) for _, g in rows.groupby("utt", sort=False)]
    original = [load_16k(str(Path(args.root) / "train" / "wav" / utt[:7] / f"{utt}.wav")) for utt in utterances]
    profile = estimate_speaker_profile(original, spans)
    by_utt = speaker_targets.set_index("utt")

    def with_imposed_tone(waveform: np.ndarray, utt: str) -> np.ndarray:
        target = by_utt.loc[utt]
        return resynthesize_with_f0(waveform, SR, profile.fmin, profile.fmax,
                                    lambda f0: impose_contour_on_span(f0, target.start, target.end,
                                                                      imposed_contours[utt], profile.reference_hz))

    waveforms = {
        "original": original,
        "resintesis": [resynthesize_with_f0(y, SR, profile.fmin, profile.fmax, lambda f0: f0) for y in original],
        "tono_cambiado": [with_imposed_tone(y, utt) for y, utt in zip(original, utterances)],
    }
    return {"utterances": utterances, "waveforms": waveforms,
            "contours": {c: syllable_contours(w, spans, profile, n_points) for c, w in waveforms.items()}}


print("Resíntesis con WORLD y contornos...")
speakers = sorted(sample.speaker.unique())
per_speaker = Parallel(n_jobs=args.jobs)(
    delayed(process_speaker)(sample[sample.speaker == s], targets[targets.speaker == s],
                             {utt: imposed_contour_of[utt] for utt in targets.utt[targets.speaker == s]})
    for s in speakers)
waveform_of = {c: {utt: y for r in per_speaker for utt, y in zip(r["utterances"], r["waveforms"][c])}
               for c in CONDITIONS}
sample_contours = {c: np.concatenate([r["contours"][c] for r in per_speaker]) for c in CONDITIONS}
target_positions = sample_position[targets.row].to_numpy()
target_folds = fold_of[targets.row.to_numpy()]

for _, target in targets.head(args.examples).iterrows():
    for condition in ["resintesis", "tono_cambiado"]:
        sf.write(out / "ejemplos" / f"{target.utt}_{condition}.wav", waveform_of[condition][target.utt], SR)
targets.head(args.examples).to_csv(out / "ejemplos" / "silabas_cambiadas.csv", index=False)

# --- Comprobación de la manipulación ---
def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean((a - b) ** 2, axis=1))


imposed_contours = np.stack([imposed_contour_of[utt] for utt in targets.utt])
changed = sample_contours["tono_cambiado"][target_positions]
manipulation = {
    "rmse_st_a_contorno_impuesto": float(np.nanmedian(rmse(changed, imposed_contours))),
    "rmse_st_a_resintesis_sin_cambios": float(np.nanmedian(rmse(changed, sample_contours["resintesis"][target_positions]))),
    "silabas_cambiadas_sin_f0": int(np.isnan(changed).all(axis=1).sum()),
}
print("Manipulación:", manipulation)

# --- Embeddings de HuBERT de la sílaba cambiada en cada condición ---
encoder = SpeechEncoder(args.model, args.revision, device)
target_embeddings = {c: np.zeros((len(targets), encoder.model.config.hidden_size), np.float32) for c in CONDITIONS}
for i, target in tqdm(targets.iterrows(), total=len(targets), desc="embeddings"):
    layer = layer_of_fold[int(target_folds[i])]
    for condition in CONDITIONS:
        target_embeddings[condition][i] = encoder.segment_means(waveform_of[condition][target.utt],
                                                                [(target.start, target.end)])[0, layer]

# --- Modelos entrenados sin el fold de cada hablante ---
sample_bounds = utterance_bounds(sample.utt.to_numpy())
all_bounds = syllables.utterance_bounds()
utterance_fold = fold_of[all_bounds[:, 0]]
x_syllable_train = contour_inputs(syllables.contours)
x_utterance_train = syllable_inputs(syllables.contours)
predicted = {(m, c): np.full(len(targets), -1) for m in MODELS[:-1] for c in CONDITIONS}


def to_tones(logits: np.ndarray) -> np.ndarray:
    return np.array(TONES)[logits.argmax(1)]


for fold in range(args.folds):
    in_fold = target_folds == fold
    if not in_fold.any():
        continue
    train = evaluated & (fold_of != fold)
    cnn = train_cnn(x_syllable_train[train], labels[train], len(TONES), TrainingConfig(), device, args.seed)
    tagger = train_tagger(x_utterance_train, labels, all_bounds[utterance_fold != fold], len(TONES),
                          TaggerConfig(), device, args.seed)
    layer = layer_of_fold[fold]
    x_train = torch.from_numpy(np.load(Path(args.embeddings) / f"layer_{layer:02d}.npy",
                                       mmap_mode="r")[train].astype(np.float32)).to(device)
    probe = LinearProbe(len(TONES), hubert["C"]).fit(x_train, torch.from_numpy(labels[train]).long().to(device))
    del x_train
    for condition in CONDITIONS:
        contours = sample_contours[condition]
        target_contours = contours[target_positions[in_fold]]
        has_f0 = np.isfinite(target_contours).all(axis=1)
        cnn_pred = np.full(in_fold.sum(), -1)
        cnn_pred[has_f0] = to_tones(predict_logits(cnn, contour_inputs(target_contours[has_f0]), device))
        predicted["cnn_silaba", condition][in_fold] = cnn_pred
        utterance_logits = predict_tagger_logits(tagger, syllable_inputs(contours), sample_bounds, device)
        predicted["frase", condition][in_fold] = to_tones(utterance_logits[target_positions[in_fold]])
        hubert_logits = probe.logits(torch.from_numpy(target_embeddings[condition][in_fold]).to(device))
        predicted["hubert", condition][in_fold] = to_tones(hubert_logits.cpu().numpy())
    print(f"  fold {fold}: {in_fold.sum()} sílabas cambiadas evaluadas")

lexical = pd.Series(lexical_tone_baseline(table[evaluated].reset_index(drop=True), fold_of[evaluated]),
                    index=np.flatnonzero(evaluated))[targets.row].to_numpy()
for condition in CONDITIONS:
    predicted["lexico_sin_audio", condition] = lexical

# --- Métricas: ¿sigue al tono original o al impuesto? ---
original_tone, imposed_tone = targets.tono_original.to_numpy(), targets.tono_impuesto.to_numpy()
outcomes = {}
for model in MODELS:
    outcomes[f"{model}|acierto_original"] = predicted[model, "original"] == original_tone
    outcomes[f"{model}|acierto_resintesis"] = predicted[model, "resintesis"] == original_tone
    outcomes[f"{model}|detecta_cambio"] = predicted[model, "tono_cambiado"] == imposed_tone
    outcomes[f"{model}|acepta_error"] = predicted[model, "tono_cambiado"] == original_tone
outcomes = pd.DataFrame(outcomes)
boot = speaker_bootstrap_accuracy(outcomes, targets.speaker.to_numpy(), args.resamples, args.seed)
rows = []
for i, column in enumerate(outcomes.columns):
    model, metric = column.split("|")
    rows.append({"modelo": model, "metrica": metric, "tasa": round(float(outcomes[column].mean()), 4),
                 "ic95_inf": round(float(np.percentile(boot[:, i], 2.5)), 4),
                 "ic95_sup": round(float(np.percentile(boot[:, i], 97.5)), 4)})
comparison = pd.DataFrame(rows)
comparison.to_csv(out / "comparacion.csv", index=False)

pair = "T" + targets.tono_original.astype(str) + " -> T" + targets.tono_impuesto.astype(str)
by_pair = (outcomes[[f"{m}|detecta_cambio" for m in MODELS[:-1]]]
           .groupby(pair.to_numpy()).mean().round(4)
           .rename(columns=lambda c: c.split("|")[0]))
by_pair.insert(0, "n", pair.value_counts().sort_index())
by_pair.to_csv(out / "por_par.csv")

predictions = targets.copy()
for (model, condition), pred in predicted.items():
    predictions[f"pred_{model}_{condition}"] = pred
predictions.to_parquet(out / "predicciones.parquet", index=False)
results = {"n_silabas_cambiadas": len(targets), "capa_por_fold": layer_of_fold, "manipulacion": manipulation,
           "tasas": comparison.pivot(index="modelo", columns="metrica", values="tasa").to_dict(orient="index")}
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + comparison.pivot(index="modelo", columns="metrica", values="tasa").to_string())
print("\nDetecta el cambio, por par original -> impuesto:\n" + by_pair.to_string())
print(f"\nResultados en {out}/")
