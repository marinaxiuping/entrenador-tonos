"""Paquete de la demo "¿quién te corrige?" (tonos.lab): modelos, frases y perfiles.

Los tres modelos se entrenan con la misma configuración que en los scripts 29-31 pero
sin los hablantes del fold --holdout-fold, y las frases de la demo salen solo de esos
hablantes: lo que se ve en la demo es rendimiento con voces no vistas, no memoria.
Las frases se eligen al azar (sin mirar qué aciertan los modelos) entre las que tienen
una longitud cómoda y F0 en todas sus sílabas de tono pleno.

El contorno que se impone para cada tono es el de la sílaba real de entrenamiento más
cercana a la media de ese tono: típico, pero natural (no un promedio suavizado).

Salidas en --out (por defecto models/laboratorio/): laboratorio.pt y audio/<frase>.wav.
Requiere los scripts 15, 16 y 31 (de este último sale la capa de HuBERT).

Uso: uv run python scripts/35_preparar_laboratorio.py
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pypinyin.contrib.tone_convert import to_tone

from tonos.aishell3 import TONES, load_syllables, speaker_fold_ids
from tonos.align import SR, load_16k
from tonos.cnn import TrainingConfig, contour_inputs, train_cnn
from tonos.context import TaggerConfig, syllable_inputs, train_tagger
from tonos.lab import BUNDLE_FILE
from tonos.linear_probe import LinearProbe
from tonos.utterance_contours import estimate_speaker_profile

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/aishell3")
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--embeddings", default="data/interim/embeddings/aishell3_chinese-hubert-base")
ap.add_argument("--model", default="TencentGameMate/chinese-hubert-base")
ap.add_argument("--revision", default="fce0375452b1dd6c080ac3248d423d4d037bc831")
ap.add_argument("--hubert-metrics", default="outputs/escalera_dominio/aishell3_hubert/metrics.json",
                help="métricas del script 31, con la capa elegida para cada fold")
ap.add_argument("--layer", type=int, help="capa de HuBERT; por defecto, la que eligió el script 31 para el fold reservado")
ap.add_argument("--C", type=float, default=1.0)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--holdout-fold", type=int, default=0)
ap.add_argument("--utterances-per-speaker", type=int, default=3)
ap.add_argument("--min-syllables", type=int, default=5)
ap.add_argument("--max-syllables", type=int, default=12)
ap.add_argument("--profile-utterances", type=int, default=60, help="frases para estimar el perfil de cada hablante")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="models/laboratorio")
args = ap.parse_args()


def chosen_hubert_layer(metrics_path: str, holdout_fold: int) -> int:
    """Capa que el script 31 eligió con validación interna sin ver ``holdout_fold``: la misma
    selección sin fuga que la de la evaluación."""
    chosen = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["capa_elegida_por_fold"]
    return int(chosen[str(holdout_fold)])


hubert_layer = args.layer if args.layer is not None else chosen_hubert_layer(args.hubert_metrics, args.holdout_fold)

device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
(out / "audio").mkdir(parents=True, exist_ok=True)
wav_root = Path(args.root) / "train" / "wav"

syllables = load_syllables(args.syllables, args.contours)
table, labels, evaluated, contours = syllables.table, syllables.labels, syllables.evaluated, syllables.contours
fold_of = speaker_fold_ids(table, args.folds, args.seed)
train = evaluated & (fold_of != args.holdout_fold)
bounds = syllables.utterance_bounds()
print(f"Entrenando sin el fold {args.holdout_fold} ({train.sum()} sílabas) en {device}, HuBERT capa {hubert_layer}")

cnn = train_cnn(contour_inputs(contours[train]), labels[train], len(TONES), TrainingConfig(), device, args.seed)
tagger = train_tagger(syllable_inputs(contours), labels, bounds[fold_of[bounds[:, 0]] != args.holdout_fold],
                      len(TONES), TaggerConfig(), device, args.seed)
x_train = torch.from_numpy(np.load(Path(args.embeddings) / f"layer_{hubert_layer:02d}.npy",
                                   mmap_mode="r")[train].astype(np.float32)).to(device)
probe = LinearProbe(len(TONES), args.C).fit(x_train, torch.from_numpy(labels[train]).long().to(device))
del x_train

training = table[train]
base_syllable = table.pinyin.str.replace(r"\d", "", regex=True)
lexicon = {
    "characters": training.groupby("hanzi").tone.agg(lambda t: int(t.mode().iloc[0])).to_dict(),
    "syllables": training.groupby(base_syllable[train]).tone.agg(lambda t: int(t.mode().iloc[0])).to_dict(),
}

donor_contours = {}
for tone in TONES:
    candidates = np.flatnonzero(train & (table.tone == tone).to_numpy())
    mean_contour = contours[candidates].mean(axis=0)
    nearest = candidates[np.argmin(((contours[candidates] - mean_contour) ** 2).sum(axis=1))]
    donor_contours[str(tone)] = contours[nearest].round(3).tolist()

# --- Frases de la demo: hablantes del fold reservado ---
rng = np.random.default_rng(args.seed)
holdout = table[fold_of == args.holdout_fold]
utterance_size = holdout.groupby("utt").size()
tonal = holdout.tone.isin(TONES)
complete = (~tonal | pd.Series(evaluated, index=table.index)[holdout.index]).groupby(holdout.utt).all()
eligible = set(utterance_size.index[(utterance_size >= args.min_syllables) & (utterance_size <= args.max_syllables)
                                    & complete[utterance_size.index]])
utterances, profiles = [], {}
for speaker, speaker_rows in holdout.groupby("speaker"):
    speaker_utts = speaker_rows.utt.unique()
    profile_utts = rng.choice(speaker_utts, size=min(args.profile_utterances, len(speaker_utts)), replace=False)
    groups = {utt: g for utt, g in speaker_rows[speaker_rows.utt.isin(profile_utts)].groupby("utt")}
    profile = estimate_speaker_profile([load_16k(str(wav_root / speaker / f"{u}.wav")) for u in groups],
                                       [list(zip(g.start, g.end)) for g in groups.values()])
    # float(): los escalares de numpy no se pueden cargar con torch.load(weights_only=True).
    profiles[speaker] = {k: float(v) for k, v in asdict(profile).items()}
    candidates = [u for u in speaker_utts if u in eligible]
    for utt in sorted(rng.choice(candidates, size=min(args.utterances_per_speaker, len(candidates)), replace=False)):
        rows = speaker_rows[speaker_rows.utt == utt]
        sf.write(out / "audio" / f"{utt}.wav", load_16k(str(wav_root / speaker / f"{utt}.wav")), SR, subtype="PCM_16")
        utterances.append({
            "id": str(utt), "speaker": speaker, "gender": rows.gender.iloc[0],
            "text": "".join(rows.hanzi),
            "syllables": [{"hanzi": r.hanzi, "pinyin": to_tone(r.pinyin), "base": base_syllable[i],
                           "tone": int(r.tone), "start": float(r.start), "end": float(r.end),
                           "evaluable": bool(evaluated[i])} for i, r in rows.iterrows()],
        })

artifact = {
    "tones": TONES,
    "n_points": int(contours.shape[1]),
    "cnn": {k: v.cpu() for k, v in cnn.state_dict().items()},
    "tagger": {k: v.cpu() for k, v in tagger.state_dict().items()},
    "hubert": {"model": args.model, "revision": args.revision, "layer": hubert_layer, "probe": probe.state_dict()},
    "lexicon": lexicon,
    "donor_contours": donor_contours,
    "profiles": profiles,
    "utterances": utterances,
    "training": {"holdout_fold": args.holdout_fold, "seed": args.seed, "n_samples": int(train.sum()),
                 "cnn": asdict(TrainingConfig()), "tagger": asdict(TaggerConfig()), "C": args.C},
}
torch.save(artifact, out / BUNDLE_FILE)
print(f"{len(utterances)} frases de {len(profiles)} hablantes no vistos | guardado en {out}/")
