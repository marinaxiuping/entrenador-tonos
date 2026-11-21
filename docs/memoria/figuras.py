"""Figuras de la memoria, leídas de las salidas de los experimentos y de los contornos procesados
(nada copiado a mano).

Genera SVG con el texto como texto (no trazado), para que el PDF lo imprima nítido con las
fuentes del sistema. Los colores de los jueces se validaron con el validador de paletas del
proyecto de visualización (luminosidad, croma, separación con daltonismo y contraste).

Uso: uv run python docs/memoria/figuras.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tonos.aishell3 import load_syllables  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--experiments", default=str(ROOT / "outputs" / "escalera_dominio"))
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figuras"))
args = ap.parse_args()

EXPERIMENTS = Path(args.experiments)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

INK, INK_SOFT, MUTED, HAIRLINE = "#3a2a1f", "#6f5847", "#8f7a68", "#dccab0"
JUDGE_COLORS = {"cnn_silaba": "#a8613a", "frase": "#d29a2e", "hubert": "#b02a3c", "detector": "#1f82b0"}
JUDGE_NAMES = {"cnn_silaba": "CNN por sílaba", "frase": "CNN + BiGRU", "hubert": "HuBERT",
               "detector": "HuBERT + alumnos"}

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": HAIRLINE,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent_axis(ax, limits=(0, 1), ticks=(0, 0.25, 0.5, 0.75, 1)):
    ax.set_xlim(*limits)
    ax.set_xticks(ticks, [f"{t:.0%}".replace("%", " %") for t in ticks])
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def ladder_figure() -> None:
    """Accuracy en cada peldaño: del laboratorio al habla continua, y qué recupera el contexto."""
    rung = load_json(EXPERIMENTS / "aishell3" / "metrics.json")
    conditions = rung["condiciones"]
    context = pd.read_csv(EXPERIMENTS / "comparacion_contexto" / "comparacion.csv").set_index("modelo").accuracy
    rows = [
        ("Tone Perfect, hablante no visto", rung["peldano_0_tone_perfect_hablante_no_visto"], "laboratorio"),
        ("AISHELL-3, modelo de la app sin tocar", conditions["cero_disparos"]["global"]["accuracy"], "continua"),
        ("AISHELL-3, CNN con el tamaño de Tone Perfect", conditions["en_dominio_tamano_tp"]["global"]["accuracy"], "continua"),
        ("AISHELL-3, CNN con todo (14 veces más)", conditions["en_dominio_completo"]["global"]["accuracy"], "continua"),
        ("+ sílaba anterior y siguiente", context["ventana"], "contexto"),
        ("+ frase completa (BiGRU)", context["frase"], "contexto"),
        ("HuBERT congelado + lineal", context["hubert"], "contexto"),
        ("Léxico, sin oír el audio", context["lexico_sin_audio"], "lexico"),
    ]
    colors = {"laboratorio": INK_SOFT, "continua": JUDGE_COLORS["cnn_silaba"], "contexto": JUDGE_COLORS["cnn_silaba"],
              "lexico": MUTED}
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for position, (label, value, kind) in enumerate(rows):
        hatch = "////" if kind == "lexico" else None
        ax.barh(position, value, height=0.62, color="white" if hatch else colors[kind],
                edgecolor=colors[kind], hatch=hatch, linewidth=0.8)
        ax.text(value + 0.012, position, f"{value:.3f}".replace(".", ","), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    for boundary in (0.5, 3.5, 6.5):
        ax.axhline(boundary, color=HAIRLINE, linewidth=0.8, linestyle=(0, (2, 2)))
    percent_axis(ax, limits=(0, 1.08), ticks=(0, 0.25, 0.5, 0.75, 1))
    ax.set_xlabel("accuracy (hablantes no vistos)")
    save(fig, "escalera")


def three_evaluations_figure() -> None:
    """Los mismos jueces medidos de tres formas: el orden cambia en cada panel."""
    swap = pd.read_csv(EXPERIMENTS / "contrafactual_cambio_tono" / "comparacion.csv").pivot(
        index="modelo", columns="metrica", values="tasa")
    detector_runs = pd.read_csv(EXPERIMENTS / "ompal" / "detector" / "particiones.csv")
    real_auc = detector_runs.groupby("modelo").auc.mean()
    real_auc.index = real_auc.index.str.replace("juez_", "").str.replace("detector_hubert$", "detector", regex=True)
    panels = [
        ("Accuracy con nativos", swap["acierto_original"], (0.7, 1.0)),
        ("Detecta errores fabricados", swap["detecta_cambio"], (0.2, 0.75)),
        ("AUC con errores reales (OMPAL)", real_auc, (0.8, 0.9)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.15), sharey=True)
    judges = ["cnn_silaba", "frase", "hubert", "detector"]
    for ax, (title, values, limits) in zip(axes, panels):
        for position, judge in enumerate(judges):
            if judge not in values.index:
                ax.text(limits[0] + 0.01 * (limits[1] - limits[0]), position, "no medido", va="center",
                        fontsize=7.5, color=MUTED, style="italic")
                continue
            value = values[judge]
            ax.plot([limits[0], value], [position, position], color=HAIRLINE, linewidth=1, zorder=1)
            ax.scatter(value, position, s=46, color=JUDGE_COLORS[judge], edgecolor="white", linewidth=1.5, zorder=2)
            ax.text(value, position - 0.32, f"{value:.3f}".replace(".", ","), ha="center", fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlim(*limits)
        ax.set_xticks([limits[0], limits[1]], [f"{limits[0]:.2f}".replace(".", ","), f"{limits[1]:.2f}".replace(".", ",")])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(judges)), [JUDGE_NAMES[j] for j in judges])
    axes[0].set_ylim(len(judges) - 0.5, -0.7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "tres_evaluaciones")


def detector_figure() -> None:
    """Detección al 10 % de falsas alarmas en el test de cada partición: juez sin entrenar y detector."""
    runs = pd.read_csv(EXPERIMENTS / "ompal" / "detector" / "particiones.csv").pivot(
        index="particion", columns="modelo", values="detecta_0.10")
    fig, ax = plt.subplots(figsize=(6.3, 1.6))
    for position, (partition, row) in enumerate(runs.iterrows()):
        before, after = row["juez_hubert"], row["detector_hubert"]
        ax.plot([before, after], [position, position], color=HAIRLINE, linewidth=2, zorder=1)
        ax.scatter(before, position, s=46, color=JUDGE_COLORS["hubert"], edgecolor="white", linewidth=1.5, zorder=2,
                   label="HuBERT sin entrenar con alumnos" if position == 0 else None)
        ax.scatter(after, position, s=46, color=JUDGE_COLORS["detector"], edgecolor="white", linewidth=1.5, zorder=2,
                   label="HuBERT + alumnos (detector)" if position == 0 else None)
        ax.text(after + 0.012, position, f"+{(after - before) * 100:.0f} puntos", va="center", fontsize=7.5, color=INK_SOFT)
    ax.set_yticks(range(len(runs)), [f"partición {p}" for p in runs.index])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    percent_axis(ax, limits=(0.35, 0.9), ticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    ax.set_xlabel("errores de tono detectados con un 10 % de falsas alarmas (alumnos de test)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.0), ncols=2)
    save(fig, "detector_particiones")


def contour_compression_figure() -> None:
    """Contorno medio de cada tono en sílabas de estudio y en habla continua (mismo pipeline de F0)."""
    tone_perfect = np.load(ROOT / "data" / "processed" / "contornos_toneperfect_praat.npz")
    continuous = load_syllables(str(ROOT / "data" / "processed" / "silabas_aishell3_praat.parquet"),
                                str(ROOT / "data" / "processed" / "contornos_aishell3_praat.npz"))
    continuous_tones = continuous.table.tone.to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.55), sharey=True)
    for ax, tone in zip(axes, [1, 2, 3, 4]):
        studio = np.nanmean(tone_perfect["contours"][tone_perfect["tone"] == tone], axis=0)
        speech = np.nanmean(continuous.contours[continuous.evaluated & (continuous_tones == tone)], axis=0)
        x = np.linspace(0, 1, len(studio))
        ax.axhline(0, color=HAIRLINE, linewidth=0.8)
        ax.plot(x, studio, color=INK_SOFT, linewidth=1.6, linestyle=(0, (4, 2)),
                label="sílabas de estudio (Tone Perfect)")
        ax.plot(x, speech, color=JUDGE_COLORS["hubert"], linewidth=2, label="habla continua (AISHELL-3)")
        ax.set_title(f"T{tone}", fontsize=9, color=INK, loc="left")
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("semitonos")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncols=2, frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "compresion_contornos")


ladder_figure()
three_evaluations_figure()
detector_figure()
contour_compression_figure()
print(f"Figuras en {OUT}")
