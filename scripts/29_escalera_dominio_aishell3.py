"""Escalera de cambio de dominio, peldaño 1: de Tone Perfect (sílabas aisladas) a AISHELL-3 (habla continua).

Misma CNN y mismas sílabas de AISHELL-3 (tonos 1-4) en tres condiciones:
  - cero disparos: el modelo de la app (models/cnn_toneperfect_praat.pt), sin tocar.
  - en dominio, tamaño de Tone Perfect: CNN entrenada con AISHELL-3 pero con tantas
    sílabas como Tone Perfect. Es el control que separa el efecto del dominio del efecto
    de tener 14 veces más datos.
  - en dominio, completo: CNN entrenada con todas las sílabas de AISHELL-3.
Las condiciones en dominio usan folds por hablante equilibrados por género. La calibración
se evalúa con la temperatura de Tone Perfect (lo que haría la app) y con una temperatura
reajustada en AISHELL-3 para cada fold con los demás folds, sin fuga.

El desglose por factores del habla continua (duración, posición, tonos vecinos...) dice
dónde se pierde la accuracy al cambiar de dominio.

Salidas en outputs/escalera_dominio/aishell3/: metrics.json, desglose.csv,
predicciones.parquet y confusion_<condición>.png.

Uso: uv run python scripts/29_escalera_dominio_aishell3.py
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tonos.aishell3 import TONES, load_syllables, speaker_fold_ids
from tonos.calibration import apply_temperature, calibrate_by_group
from tonos.cnn import TrainingConfig, contour_inputs, predict_logits, train_cnn
from tonos.evaluation import accuracy_breakdown, continuous_speech_factors, domain_metrics, save_confusion
from tonos.inference import ToneClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/cnn_toneperfect_praat.pt")
ap.add_argument("--syllables", default="data/processed/silabas_aishell3_praat.parquet")
ap.add_argument("--contours", default="data/processed/contornos_aishell3_praat.npz")
ap.add_argument("--tp-metrics", default="outputs/cnn_toneperfect/praat/metrics.json",
                help="accuracy de Tone Perfect con hablante no visto (peldaño 0)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/escalera_dominio/aishell3")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# --- Datos: solo tonos plenos con contorno, que es lo que sabe clasificar el modelo ---
syllables = load_syllables(args.syllables, args.contours)
meta = syllables.table[syllables.evaluated].reset_index(drop=True)
x = contour_inputs(syllables.contours[syllables.evaluated])
labels = syllables.labels[syllables.evaluated]
excluded_tonal = (syllables.table.tone.isin(TONES) & ~syllables.evaluated).sum()
print(f"{len(meta)} sílabas de {meta.speaker.nunique()} hablantes "
      f"(fuera: {(syllables.table.tone == 5).sum()} neutras, {excluded_tonal} sin F0 suficiente) | "
      f"dispositivo {device}")

classifier = ToneClassifier(args.model, device=device)
if classifier.n_points != x.shape[-1] or classifier.tones != TONES:
    raise SystemExit("Los contornos no tienen la longitud o los tonos del modelo de Tone Perfect")
tone_perfect_size = torch.load(args.model, weights_only=True)["training"]["n_samples"]

fold_of = speaker_fold_ids(meta, args.folds, args.seed)
config = TrainingConfig()


def out_of_fold_logits(train_size: int | None) -> np.ndarray:
    """Logits de cada fold con una CNN entrenada en los demás (submuestreados si train_size)."""
    rng = np.random.default_rng(args.seed)
    logits = np.zeros((len(meta), len(TONES)), dtype=np.float32)
    for fold in range(args.folds):
        test = fold_of == fold
        train_ids = np.flatnonzero(~test)
        if train_size is not None:
            train_ids = rng.choice(train_ids, size=min(train_size, train_ids.size), replace=False)
        model = train_cnn(x[train_ids], labels[train_ids], len(TONES), config, device, args.seed)
        logits[test] = predict_logits(model, x[test], device)
        print(f"  fold {fold}: {train_ids.size} sílabas de entrenamiento, "
              f"accuracy {(logits[test].argmax(1) == labels[test]).mean():.3f}")
    return logits


print("\nCero disparos (modelo de Tone Perfect)")
zero_shot_logits = predict_logits(classifier.model, x, device)
print("\nEn dominio, tamaño de Tone Perfect")
matched_logits = out_of_fold_logits(tone_perfect_size)
print("\nEn dominio, completo")
full_logits = out_of_fold_logits(None)

probabilities = {
    "cero_disparos": apply_temperature(zero_shot_logits, classifier.temperature),
    "cero_disparos_T_reajustada": calibrate_by_group(zero_shot_logits, labels, fold_of),
    "en_dominio_tamano_tp": calibrate_by_group(matched_logits, labels, fold_of),
    "en_dominio_completo": calibrate_by_group(full_logits, labels, fold_of),
}
tone_perfect_accuracy = json.loads(Path(args.tp_metrics).read_text(encoding="utf-8"))["global"]["accuracy"]
results = {
    "peldano_0_tone_perfect_hablante_no_visto": tone_perfect_accuracy,
    "condiciones": {name: domain_metrics(p, labels, meta, TONES) for name, p in probabilities.items()},
    "temperatura_tone_perfect": classifier.temperature,
    "n_silabas": int(len(meta)),
    "n_entrenamiento_tamano_tp": int(tone_perfect_size),
    "entrenamiento": {**asdict(config), "seed": args.seed, "folds": args.folds},
}

# Las dos variantes de cero disparos solo difieren en calibración: una basta en el desglose.
compared = ["cero_disparos", "en_dominio_tamano_tp", "en_dominio_completo"]
correct = pd.DataFrame({name: probabilities[name].argmax(1) == labels for name in compared})
accuracy_breakdown(correct, continuous_speech_factors(meta)).to_csv(out / "desglose.csv", index=False)

predictions = meta[["speaker", "gender", "utt", "word_idx", "syl_idx", "pinyin", "tone"]].copy()
for name in compared:
    predictions[f"pred_{name}"] = np.array(TONES)[probabilities[name].argmax(1)]
    predictions[f"conf_{name}"] = probabilities[name].max(1).round(4)
    save_confusion(meta.tone.to_numpy(), predictions[f"pred_{name}"].to_numpy(), TONES,
                   f"AISHELL-3 · {name} · acc {results['condiciones'][name]['global']['accuracy']:.3f}",
                   out / f"confusion_{name}.png")
predictions.to_parquet(out / "predicciones.parquet", index=False)
(out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nPeldaño 0, Tone Perfect (hablante no visto): accuracy {tone_perfect_accuracy:.3f}")
for name, r in results["condiciones"].items():
    mean, std = r["accuracy_hablante_media_std"]
    print(f"{name:28s} accuracy {r['global']['accuracy']:.3f} (hablantes {mean:.3f} ± {std:.3f}) | "
          f"macro-F1 {r['global']['macro_f1']:.3f} | ECE {r['ece']:.3f} | recall {r['global']['recall_por_tono']}")
print(f"\nResultados en {out}/")
