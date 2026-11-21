"""Genera notebooks/02_escalera_dominio.ipynb sin ejecutar.

El texto del notebook cita cifras de los scripts 06, 15, 16, 29-37 y 39-49: si cambian los
resultados, hay que revisarlo aquí y regenerarlo. Las celdas se ejecutan después con
nbconvert, que es lo que deja las salidas en el notebook.

Uso:
  uv run python scripts/38_generar_notebook_02.py
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_escalera_dominio.ipynb
"""

import argparse

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="notebooks/02_escalera_dominio.ipynb")
args = ap.parse_args()

cells = []


def add_markdown(text):
    cells.append(new_markdown_cell(text.strip()))


def add_code(text):
    cells.append(new_code_cell(text.strip()))


add_markdown("""
# Escalera de dominio: de las sílabas de laboratorio al juez de pronunciación

Continúa [01_tone_perfect_pipeline.ipynb](01_tone_perfect_pipeline.ipynb). Allí los modelos llegan al 98,8–99,0 % con seis nativos que leen sílabas aisladas en estudio, una cifra que no dice cuánto valen en habla real ni si sirven para corregir a un alumno. Este notebook sube esa escalera paso a paso:

1. **Peldaño 1: habla continua de nativos (AISHELL-3).** ¿Cuánto cae el modelo de la app y por qué?
2. **Contexto y modelos preentrenados.** ¿Se recupera con contexto de frase o con HuBERT?
3. **Dos contrafactuales.** ¿Los modelos deciden por el tono que oyen o por la palabra que reconocen?
4. **Peldaño 2: errores reales de alumnos (OMPAL).** ¿Se sostiene lo que dicen los contrafactuales cuando los errores son de verdad?
5. **La tesis.** Tres formas de evaluar a los mismos jueces y tres órdenes distintos.
6. **De los resultados a la app.** La demo «¿Quién te corrige?», la pestaña de resultados y la guía tarareada.

Los alumnos de OMPAL son francófonos. El peldaño con hispanohablantes depende de grabaciones que no se han podido reunir a tiempo.

Igual que el 01, el notebook **no vuelve a ejecutar los pasos pesados** (F0 de 17.549 frases, embeddings, entrenamientos, resíntesis). Lee lo que dejan los scripts en `data/` y `outputs/`. Para regenerarlo todo (requiere el alineamiento de AISHELL-3 de los scripts 09-12 y el pipeline de Tone Perfect del 01):

```powershell
uv run python scripts/15_contornos_aishell3.py
uv run python scripts/16_embeddings_aishell3.py
uv run python scripts/29_escalera_dominio_aishell3.py
uv run python scripts/37_curva_hablantes_aishell3.py
uv run python scripts/30_contexto_f0_aishell3.py
uv run python scripts/31_hubert_aishell3.py
uv run python scripts/32_comparar_contexto_aishell3.py
uv run python scripts/33_contrafactual_f0_plano.py
uv run python scripts/34_contrafactual_cambio_tono.py
# semillas 1 y 2 de 29, 30 y 34 con --train-seed y --out outputs/escalera_dominio/semillas/<carpeta>/semilla_<n>
uv run python scripts/39_resumir_semillas.py
uv run python scripts/40_alinear_ompal.py
uv run python scripts/41_jueces_ompal.py
uv run python scripts/42_revisar_alineamiento_ompal.py   # audios para revisar el alineamiento a oído
uv run python scripts/43_analizar_ompal.py
uv run python scripts/44_umbral_ompal.py
uv run python scripts/45_embeddings_ompal.py
uv run python scripts/46_detector_errores_ompal.py
uv run python scripts/49_juez_practicar_ompal.py
uv run python scripts/35_preparar_laboratorio.py
uv run python scripts/47_detector_laboratorio.py
uv run python scripts/48_detector_errores_fabricados.py
uv run python scripts/36_exportar_resultados.py
uv run python scripts/06_exportar_guia_tonos.py
```

Los entrenamientos de los scripts 29, 30 y 34 se repitieron con **tres semillas de entrenamiento** (mismos folds y datos, otra inicialización; script 39): las cifras se mueven en milésimas y el orden de los jueces no cambia (sección 7.1). El resto de experimentos usa una semilla.
""")

add_code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
OUT, DATA = ROOT / "outputs" / "escalera_dominio", ROOT / "data"
TONES = [1, 2, 3, 4]
pd.set_option("display.precision", 3)


def show(path, width=420):
    display(Image(filename=str(path), width=width))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
""")

add_markdown("""
## 1. Por qué una escalera

Un 0,988 con hablante no visto en Tone Perfect solo dice que el modelo distingue cuatro formas de contorno muy marcadas, producidas con cuidado y aisladas. En una app real hay dos saltos: el habla deja de ser de laboratorio y quien habla deja de ser nativo. Mezclarlos impide saber qué rompe qué, así que se miden por separado. El primer peldaño cambia solo el estilo de habla (habla continua leída por nativos) y mantiene a hablantes que pronuncian bien. Así cualquier caída se debe al dominio y no a errores de pronunciación.

## 2. El mismo pipeline sobre AISHELL-3

Para atribuir la caída al dominio, la entrada del modelo tiene que construirse **exactamente igual** en los dos corpus (`15_contornos_aishell3.py`):

- **Sílabas del alineamiento de AISHELL-3 (scripts 09-12).** 40 hablantes (20 mujeres y 20 hombres) alineados con MMS_FA y `fill_gaps` (ver `docs/corpus-bisilabos.md`). Las etiquetas son tonos *realizados*, con el sandhi incluido.
- **F0 con Praat sobre la frase completa** y después cortado por sílaba. Praat necesita contexto en los bordes; cortar antes estropearía el principio y el final de cada sílaba, que es justo donde está el movimiento tonal.
- **Rango por hablante en dos pasadas, `fix_octave_jumps` por sílaba, semitonos respecto a la media del hablante y 64 puntos**, como en Tone Perfect.
- **Un error que habría invalidado todo.** Praat no empieza sus frames en t = 0 sino a media ventana del inicio, y la ventana depende del suelo de F0. Sin corregirlo, los tiempos del alineamiento caían 2-3 frames desplazados sobre el F0, del orden del 10 % de una sílaba de 220 ms. `f0_on_frame_grid` coloca el F0 en la rejilla de 10 ms que empieza en t = 0.

Hay una diferencia que no se puede evitar: la referencia del hablante sale de su habla natural, con la distribución de tonos del chino, y no del 25 % por tono de Tone Perfect.
""")

add_code("""
syllables = pd.read_parquet(DATA / "processed" / "silabas_aishell3_praat.parquet")
contours_ai = np.load(DATA / "processed" / "contornos_aishell3_praat.npz")["contours"]
tp = np.load(DATA / "processed" / "contornos_toneperfect_praat.npz", allow_pickle=True)
contours_tp, tones_tp = tp["contours"], tp["tone"].astype(int)

no_f0 = np.isnan(contours_ai).all(axis=1)
print(f"{len(syllables)} sílabas alineadas de {syllables.speaker.nunique()} hablantes y "
      f"{syllables.utt.nunique()} frases; sin F0 suficiente: {no_f0.mean():.1%}")
print("sílabas por tono (5 = neutro):", syllables.tone.value_counts().sort_index().to_dict())

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
t = np.linspace(0, 1, contours_ai.shape[1])
for tone, ax in zip(TONES, axes):
    ax.plot(t, np.nanmean(contours_tp[tones_tp == tone], axis=0), "k--", lw=2, label="Tone Perfect (aislada)")
    ax.plot(t, np.nanmean(contours_ai[syllables.tone.to_numpy() == tone], axis=0), color="tab:red", lw=2.5,
            label="AISHELL-3 (habla continua)")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_title(f"Tono {tone}")
axes[0].set_ylabel("semitonos respecto a la media del hablante")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Contorno medio por tono: la misma forma, comprimida")
fig.tight_layout()

for tone in TONES:
    a = np.nanmean(contours_tp[tones_tp == tone], axis=0)
    b = np.nanmean(contours_ai[syllables.tone.to_numpy() == tone], axis=0)
    print(f"T{tone}: recorrido (máx - mín) Tone Perfect {a.max() - a.min():.1f} st | AISHELL-3 {b.max() - b.min():.1f} st")
""")

add_markdown("""
La figura resume el problema antes de entrenar nada. En habla continua las formas se mantienen, pero muy comprimidas: el recorrido del T4 pasa de 10,7 a 2,8 st y el del T2 de 6,7 a 1,0 st (apenas sube), y el T3 pierde la subida final y se queda en un tono bajo descendente (el «medio tercer tono» de los manuales). Un modelo que aprendió las formas exageradas de las sílabas aisladas va a confundir sobre todo lo que se ha aplanado.

## 3. Peldaño 1: el modelo de la app en habla continua

`29_escalera_dominio_aishell3.py` evalúa la misma CNN sobre las 188.743 sílabas de tono pleno con contorno. Quedan fuera las neutras, porque el modelo no las conoce, y el 1 % sin F0. Hay tres condiciones:

- **Cero disparos:** el modelo de la app tal cual, con su temperatura de Tone Perfect.
- **En dominio, completo:** la misma arquitectura entrenada con AISHELL-3, en 5 folds por hablante equilibrados por género.
- **En dominio, tamaño de Tone Perfect:** igual, pero con solo 9.839 sílabas de entrenamiento. Es el control que haría cualquier revisor: AISHELL-3 tiene 14 veces más datos, y sin él no se sabría si la mejora en dominio viene del dominio o del volumen.

La calibración se mide con la temperatura de Tone Perfect (lo que haría la app) y con una temperatura reajustada en AISHELL-3 para cada fold usando solo los demás folds.
""")

add_code("""
rung = load_json(OUT / "aishell3" / "metrics.json")
rows = {"Tone Perfect, hablante no visto (peldaño 0)": {"accuracy": rung["peldano_0_tone_perfect_hablante_no_visto"]}}
labels = {"cero_disparos": "AISHELL-3, cero disparos (T de Tone Perfect)",
          "cero_disparos_T_reajustada": "AISHELL-3, cero disparos (T reajustada)",
          "en_dominio_tamano_tp": "AISHELL-3, en dominio, tamaño de Tone Perfect",
          "en_dominio_completo": "AISHELL-3, en dominio, completo"}
for key, label in labels.items():
    r = rung["condiciones"][key]
    rows[label] = {"accuracy": r["global"]["accuracy"], "media entre hablantes": r["accuracy_hablante_media_std"][0],
                   "desv. entre hablantes": r["accuracy_hablante_media_std"][1], "macro-F1": r["global"]["macro_f1"],
                   "ECE": r["ece"], **{f"recall T{t}": r["global"]["recall_por_tono"][str(t)] for t in TONES}}
pd.DataFrame(rows).T
""")

add_code("""
for name in ["cero_disparos", "en_dominio_completo"]:
    show(OUT / "aishell3" / f"confusion_{name}.png", width=380)
""")

add_markdown("""
Hay tres lecturas, de más a menos evidente:

1. **El 98,8 % se queda en 56,6 %.** El tono que más sufre es el T2 (recall 0,19): comprimido, su subida no se distingue de un T1 o de un T3.
2. **El modelo se equivoca con seguridad.** Con la temperatura de Tone Perfect, el ECE es 0,32: de media, su confianza se aleja 32 puntos de lo que acierta, casi siempre por exceso. En una app eso significa decirle a un alumno, con total confianza, que lo ha hecho mal cuando no es así. Reajustar la temperatura lo deja en 0,017, así que **la calibración no viaja entre dominios y hay que rehacerla en cada uno**.
3. **Más sílabas no es la solución.** Entrenar en el dominio recupera 18 puntos (0,748 con el tamaño de Tone Perfect), y multiplicar las sílabas por 14 con los mismos hablantes solo añade 2 más (0,768). La sección 3.1 mira la otra variable, el número de voces.

### 3.1 ¿Y más voces?

`37_curva_hablantes_aishell3.py` entrena con 4, 8, 16 y 32 hablantes de los otros folds (mitad mujeres, mitad hombres, con todas sus sílabas) y evalúa en los hablantes del fold. Los subconjuntos son anidados, para que la diferencia entre dos puntos no dependa de qué hablantes tocan. Entrena dos modelos: la CNN por sílaba de la sección 3 y la CNN + BiGRU sobre la frase de la sección 4, que se adelanta aquí.
""")

add_code("""
curve = pd.read_csv(OUT / "curva_hablantes" / "resumen.csv")
display(curve.pivot(index="hablantes", columns="modelo", values=["accuracy_media", "accuracy_desv"]))
show(OUT / "curva_hablantes" / "curva_hablantes.png", width=520)
domain = load_json(OUT / "aishell3" / "metrics.json")["condiciones"]
print(f"Control del script 29 (32 hablantes, sílabas recortadas al tamaño de Tone Perfect): "
      f"{domain['en_dominio_tamano_tp']['global']['accuracy']:.3f}")
""")

add_markdown("""
La curva no se aplana en el rango medido. Cada vez que se duplican los hablantes, la CNN por sílaba gana entre 1,1 y 1,5 puntos (0,730 / 0,745 / 0,756 / 0,768) y la frase entre 1,5 y 2,5 (0,772 / 0,797 / 0,812 / 0,833). La desviación entre folds es de 1-2 puntos, del mismo orden que cada salto, así que un punto aislado no es concluyente, pero la tendencia sí, porque se repite en los cinco folds y en los dos modelos.

Más hablantes también trae más sílabas (de 18.000 a 151.000), así que la curva sola no separa voces de volumen. El control del script 29 sí ayuda a separarlos. Con 32 hablantes y solo 9.839 sílabas se llega a 0,748, más que con 4 hablantes y 18.000 sílabas (0,730). **Con menos sílabas pero más voces se acierta más**: lo que generaliza a un hablante nuevo es la variedad de voces, no la cantidad de sílabas de las mismas voces.

Esto matiza la conclusión de la sección 3. El techo de ~0,77 de la CNN por sílaba no es de volumen de sílabas, pero sí podría subir algo con más voces, al ritmo de ~1,2 puntos por duplicación. Para llegar a lo que logra la frase completa (0,833) harían falta unas 5 duplicaciones más, unas 40 veces más hablantes, si la tendencia se mantuviera, cosa que no se ha medido. El contexto sigue siendo la palanca grande. La curva de la frase es más empinada, así que el modelo con contexto aprovecha mejor cada voz nueva.

### 3.2 Dónde falla
""")

add_code("""
breakdown = pd.read_csv(OUT / "aishell3" / "desglose.csv")
columns = ["n", "cero_disparos", "en_dominio_completo"]
for factor in ["tono", "duracion_ms", "posicion_en_frase", "tono_siguiente"]:
    display(breakdown[breakdown.factor == factor].set_index("nivel")[columns].rename_axis(factor))
""")

add_markdown("""
Incluso entrenado en el dominio, el modelo falla más en la **última sílaba de la frase** (0,63 sin sílaba siguiente), en las **sílabas cortas** (0,67 por debajo de 181 ms) y en el **T3** (0,63). Son los tres sitios donde el contexto ayuda a un oyente humano: al final de la frase la declinación y la voz rasposa deforman el contorno, en una sílaba corta apenas hay contorno, y el T3 realizado depende de lo que viene detrás. Eso lleva directamente al siguiente experimento.

## 4. Contexto sobre el F0

`30_contexto_f0_aishell3.py` mantiene la entrada (solo F0) y amplía lo que ve el modelo. Usa las mismas sílabas evaluadas, los mismos folds y la misma calibración sin fuga que el script 29:

- **Ventana ±1:** la CNN recibe 9 canales: contorno, derivada y un canal de presencia de la sílaba anterior, la actual y la siguiente. El canal de presencia distingue un borde de frase o una sílaba sin F0 de un contorno plano a cero.
- **Frase (CNN + BiGRU):** la misma CNN codifica cada sílaba y una BiGRU de dos capas recorre la frase entera. Así cada predicción puede usar la declinación, el sandhi y la coarticulación con cualquier vecina. Las sílabas neutras y sin F0 entran como contexto pero no cuentan en la pérdida. Solo se codifican las sílabas reales y no el relleno, para que BatchNorm no vea ceros artificiales.

Ninguno de los dos ve la duración, igual que la CNN por sílaba.
""")

add_code("""
context = load_json(OUT / "aishell3_contexto" / "metrics.json")
pd.DataFrame({name: {"accuracy": r["global"]["accuracy"], "media entre hablantes": r["accuracy_hablante_media_std"][0],
                     "macro-F1": r["global"]["macro_f1"], "ECE": r["ece"],
                     **{f"recall T{t}": r["global"]["recall_por_tono"][str(t)] for t in TONES}}
              for name, r in context["modelos"].items()}).T
""")

add_markdown("""
El contexto ayuda y la ayuda va donde hacía falta. La frase completa sube a 0,833, con el T3 de 0,63 a 0,78, las sílabas cortas de 0,67 a 0,76 y la última de la frase de 0,63 a 0,77 (desglose en la sección 5). Sigue lejos del 98,8 % de laboratorio.

## 5. HuBERT y una línea base que no oye nada

`16_embeddings_aishell3.py` pasa cada frase **completa** por `chinese-hubert-base`, el mismo modelo y commit que en el 01, y promedia cada capa dentro de los tiempos de cada sílaba. Por la autoatención, cada sílaba lleva ya información de toda la frase, así que es un modelo con contexto, comparable a la BiGRU. `31_hubert_aishell3.py` entrena una regresión logística por capa y elige la capa dentro de cada fold, por validación interna sobre los demás folds.

Con 150.000 filas por 768 dimensiones y 325 ajustes, la regresión logística de sklearn era inviable, así que se reimplementó en GPU (`tonos.linear_probe`: mismo objetivo, L-BFGS). Con 300 iteraciones la capa 0, mal condicionada, tardaba unos 20 minutos por capa; con 100 iteraciones la accuracy es la misma (0,9598 frente a 0,9594 en una prueba) y cada ajuste tarda unos 2 s.
""")

add_code("""
hubert = load_json(OUT / "aishell3_hubert" / "metrics.json")
layers = pd.read_csv(OUT / "aishell3_hubert" / "accuracy_por_capa.csv", index_col=0).accuracy
print("capa elegida por fold:", hubert["capa_elegida_por_fold"])
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(layers.index, layers.values, "o-", color="k")
ax.set_xlabel("capa (0 = salida de la CNN de entrada)")
ax.set_ylabel("accuracy en el fold de test")
ax.set_title("Diagnóstico por capa (no se usa para elegir)")
fig.tight_layout()
""")

add_markdown("""
Con la capa 10, elegida en los cinco folds, HuBERT llega a **0,966**. Antes de creérselo hace falta una comparación que el resto de la literatura suele omitir. HuBERT codifica sobre todo **qué se dice**, y en chino el carácter casi determina el tono. Así que `32_comparar_contexto_aishell3.py` añade una **línea base léxica sin audio**: el tono más frecuente de cada carácter en los folds de entrenamiento (si el carácter no aparece, el de la sílaba sin tono).

Todas las diferencias llevan intervalos de confianza al 95 % por **bootstrap de hablantes**. La unidad independiente es el hablante, no la sílaba: remuestrear 188.743 sílabas daría intervalos falsamente estrechos. Las diferencias son pareadas, con los mismos hablantes en cada remuestreo.
""")

add_code("""
pd.read_csv(OUT / "comparacion_contexto" / "comparacion.csv").set_index("modelo")
""")

add_code("""
print("Solapamiento de errores frase / HuBERT:", load_json(OUT / "comparacion_contexto" / "solapamiento_errores.json"))
context_breakdown = pd.read_csv(OUT / "comparacion_contexto" / "desglose.csv")
models = ["lexico_sin_audio", "cnn_silaba", "frase", "hubert"]
for factor in ["tono", "duracion_ms", "tono_siguiente"]:
    display(context_breakdown[context_breakdown.factor == factor].set_index("nivel")[["n", *models]].rename_axis(factor))
""")

add_markdown("""
**Sin oír nada se acierta el 93,4 %.** HuBERT queda solo 3 puntos por encima del léxico. La combinación frase + HuBERT no mejora a HuBERT solo, aunque sus errores son bastante distintos (un oráculo que eligiera siempre al que acierta llegaría al 97,9 %).

Hay un indicio de que HuBERT oye algo más que el léxico. Donde el léxico falla por el sandhi (T3 ante T3, realizado como T2), HuBERT se mantiene. Pero no basta: la regla del sandhi también se deduce del carácter siguiente, que HuBERT reconoce. La accuracy no puede separar «oye el tono» de «sabe qué palabra es». Hace falta intervenir sobre el audio.

## 6. Contrafactual 1: borrar el tono

`33_contrafactual_f0_plano.py` toma 2.400 frases (60 por hablante) y las resintetiza con WORLD de dos formas:

- **Resíntesis sin tocar el F0:** el control de los artefactos del vocoder.
- **F0 plano:** el F0 de los frames con voz se fija a la media geométrica de la frase. Desaparece toda la información tonal y se conserva el contenido segmental, la duración, la intensidad y la calidad de voz.

Cada frase la clasifica el clasificador de HuBERT del fold que no ha visto a su hablante. La **comprobación de la manipulación** mide con Praat la dispersión del F0 de cada frase en las tres condiciones.
""")

add_code("""
flat = load_json(OUT / "contrafactual_f0_plano" / "metrics.json")
print("dispersión del F0 por frase (st, mediana):", flat["dispersion_f0_st_mediana"])
print("coincidencia de la predicción con el léxico:", flat["coincidencia_con_lexico"])
display(pd.read_csv(OUT / "contrafactual_f0_plano" / "comparacion.csv").set_index("condicion"))
pd.DataFrame(flat["accuracy_por_tono"]).T.rename_axis("tono")
""")

add_markdown("""
La manipulación funciona: la dispersión del F0 baja de 2,96 a 0,12 st. El vocoder solo cuesta 1,1 puntos. **Con el tono borrado, HuBERT cae 27 puntos, hasta 0,693.** Hay dos conclusiones:

- **HuBERT sí usa el pitch:** 27 puntos dependen de él, así que no es puramente léxico.
- **Sin ningún tono sigue acertando el 69 %.** Parte de eso es léxico. Otra parte son pistas que no son pitch y que WORLD conserva (duración, intensidad, voz rasposa del T3). Y otra es un sesgo: un F0 plano *suena* a T1, por eso el T1 se queda en 0,88 y el T3 cae a 0,44. Este experimento no separa esas tres partes.

Para una app el dato incómodo es el segundo. Un alumno que leyera una frase sin ningún tono recibiría el visto bueno de HuBERT en el 69 % de las sílabas. Pero el F0 plano es un caso extremo. El caso de uso real es más sutil: una frase bien dicha salvo un tono.

## 7. Contrafactual 2: cambiar el tono de una sola sílaba

`34_contrafactual_cambio_tono.py` reproduce el error de un alumno. Usa las mismas 2.400 frases: en cada una se elige una sílaba al azar y se le impone con WORLD el contorno de otro tono, también al azar, sin tocar el resto de la frase. Un buen juez debe **seguir al tono impuesto** y no al que corresponde a la palabra. Las decisiones de diseño:

- **El contorno impuesto es el de una sílaba real de ese tono, de un hablante de otro fold,** y no la media del tono. La primera versión imponía la media, y la prueba de humo (1 época, 80 sílabas, no guardada) dio a la CNN un 96 % de detección. Ese número era sospechoso: la media es un prototipo limpio que favorece a los modelos de F0 frente a la variabilidad del habla real.
- **El audio manipulado pasa por el mismo pipeline de F0 que el de entrenamiento,** con el perfil del hablante (rango y referencia) estimado sobre su audio original, igual que hace la app con la calibración.
- **Los tres modelos se reentrenan por fold** con la misma configuración que en los scripts 29-31, y cada frase la juzga el modelo que no ha visto a su hablante.
- **Comprobación de la manipulación:** distancia del contorno reextraído al impuesto frente a la distancia al de la resíntesis sin cambios.
""")

add_code("""
swap = load_json(OUT / "contrafactual_cambio_tono" / "metrics.json")
print("manipulación:", swap["manipulacion"])
rates = pd.read_csv(OUT / "contrafactual_cambio_tono" / "comparacion.csv")
table = rates.pivot(index="modelo", columns="metrica", values="tasa")[
    ["acierto_original", "acierto_resintesis", "detecta_cambio", "acepta_error"]]
table.loc[["cnn_silaba", "frase", "hubert", "lexico_sin_audio"]]
""")

add_code("""
display(rates[rates.metrica == "detecta_cambio"].set_index("modelo")[["tasa", "ic95_inf", "ic95_sup"]])
pd.read_csv(OUT / "contrafactual_cambio_tono" / "por_par.csv", index_col=0).rename_axis("original -> impuesto")
""")

add_code("""
order = ["cnn_silaba", "frase", "hubert"]
names = {"cnn_silaba": "CNN por sílaba", "frase": "CNN + BiGRU", "hubert": "HuBERT"}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for model in order:
    x, y = table.loc[model, "acierto_original"], table.loc[model, "detecta_cambio"]
    ax.scatter(x, y, s=60, color="k")
    ax.annotate(names[model], (x, y), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("accuracy con habla nativa")
ax.set_ylabel("detecta un tono mal dicho")
ax.set_xlim(0.7, 1.0)
ax.set_ylim(0.2, 0.75)
ax.set_title("Cuanto más acierta con nativos, menos errores detecta")
fig.tight_layout()
""")

add_markdown("""
**El orden de los modelos se invierte.** La manipulación funciona: el contorno reextraído queda a 0,44 st del impuesto y a 3,04 st del original, y solo 13 sílabas se quedan sin F0.

- **CNN por sílaba:** acierta el 76 % con nativos y detecta el 67,5 % de los cambios. Solo deja pasar el 10 %.
- **CNN + BiGRU:** acierta más (83 %) pero detecta menos (59,8 %) y deja pasar el 14 %. Sobre la misma señal de F0, el contexto aprende qué tono es *probable* ahí, y eso lo sesga hacia lo esperado.
- **HuBERT:** acierta el 97 % pero solo detecta el 31 %, y en el **47 %** de los casos responde con el tono que «toca» según la palabra.
- **Léxico:** por construcción se queda siempre con el tono original (el 2 % restante son sus propios errores).

En todos los pares de tonos, la CNN por sílaba detecta más que la BiGRU y esta más que HuBERT. La debilidad compartida es imponer un T3 (lo comprimido y bajo se confunde fácilmente).

Con estos datos, la conclusión parecía clara: la accuracy con nativos no mide la validez del juez, y el más simple, la CNN por sílaba, es el mejor para corregir. La primera mitad se sostiene. La segunda no: la sección 8 pone a los mismos jueces ante errores reales de alumnos y el orden vuelve a cambiar.

""")

add_markdown("""
### 7.1 ¿Depende de la semilla?

Una crítica obvia a los números de este notebook es que salen de un solo entrenamiento. `29`, `30` y `34` se repitieron con `--train-seed` 1 y 2: mismos folds, mismas frases y mismos contornos donantes, y solo cambian la inicialización y el orden de entrenamiento de las redes. HuBERT no se repite: su clasificador es una regresión logística con L-BFGS, que da lo mismo con cualquier semilla.
""")

add_code("""
seeds = pd.read_csv(OUT / "semillas" / "resumen.csv").set_index("metrica")
display(seeds)
by_seed = pd.read_csv(OUT / "semillas" / "semillas.csv")
detection = by_seed[by_seed.metrica.str.endswith("detecta_cambio")].assign(juez=lambda d: d.metrica.str.split(" · ").str[1])
detection.pivot(index="semilla", columns="juez", values="valor")[["cnn_silaba", "frase", "hubert"]]
""")

add_markdown("""
La desviación entre semillas es de milésimas: 0,001 en la accuracy de la CNN por sílaba y de la frase, y como mucho 0,007 en las tasas del contrafactual (lo que deja pasar la frase). Las diferencias entre jueces son de decenas de puntos, así que el orden CNN > frase > HuBERT en detección se repite en las tres semillas. La semilla no explica ningún resultado de este notebook.
""")

add_markdown("""
## 8. Peldaño 2: errores reales de alumnos (OMPAL)

Los contrafactuales fabrican el error: cambian solo el F0 de la sílaba de un nativo. Un alumno que se equivoca no hace eso. Puede dudar, alargar la sílaba o pronunciarla a medias. **OMPAL** (Hsieh et al., Interspeech 2025, CC BY 4.0) son 1.768 frases leídas por 46 alumnos francófonos de mandarín y 82 de 3 nativos. Tres profesores marcan cada carácter de los alumnos con «tono correcto» o «tono incorrecto». Los caracteres son tradicionales, así que probablemente es mandarín de Taiwán.

### 8.1 Preparación

- **Tono esperado** (`40`): pypinyin sobre la frase entera, con el sandhi de 一, 不 y del tercer tono. Un T3 seguido de otro T3 se marca como ambiguo y se excluye (526 sílabas de alumnos): no se sabe si lo correcto era T2 o T3.
- **Tiempos:** OMPAL no trae dónde está cada sílaba. Se alinean con el mismo MMS_FA que AISHELL-3. Con alumnos que se equivocan, el alineamiento puede fallar, y entonces el juez juzgaría un trozo que no es la sílaba anotada. `42` exporta 20 frases (223 sílabas), con las 5 de peor score CTC, cada sílaba recortada y separada por silencios. **La revisión a oído dio los cortes por buenos.** Fue un veredicto global, sin anotar sílaba a sílaba, así que no hay un porcentaje de cortes correctos.
- **Etiqueta:** el tono está mal si lo dicen al menos 2 de los 3 profesores.
- **Jueces:** los de la demo (paquete de `35`), entrenados con AISHELL-3. Ningún hablante de OMPAL se ha visto al entrenar.
""")

add_code("""
ompal = load_json(OUT / "ompal" / "metrics.json")
print(ompal["n"])
judges_real = pd.DataFrame(ompal["jueces"]).T.drop(index="lexico_sin_audio")
judges_real[["detecta", "falsa_alarma", "auc", "falsa_alarma_nativos"]].astype(float)
""")

add_markdown("""
### 8.2 Los jueces ante errores reales

Se evalúan 17.001 sílabas de alumnos, de las que 2.389 (14,1 %) tienen el tono mal según la mayoría. Un juez «avisa» cuando el tono que oye no es el esperado.

**El resultado contradice al contrafactual.** HuBERT es el que mejor separa los tonos bien dichos de los mal dichos (AUC 0,859 frente a 0,831 de la CNN y 0,830 de la frase). Detecta casi lo mismo que la CNN (0,85 frente a 0,89) y da muchas menos falsas alarmas: 25 % de las sílabas bien dichas por alumnos frente a 42 %, y 10 % frente a 38 % con nativos. Lo que en `34` parecía su gran defecto, dejar pasar el error porque reconoce la palabra, con errores reales no aparece. La ventaja de AUC de HuBERT sobre la CNN es pequeña pero no es ruido: +0,028, con intervalo pareado por hablantes [0,016; 0,040] (sección 8.3).

### 8.3 Cuánto coinciden los profesores y qué pesa la palabra

`43` compara a cada profesor y a cada juez con los otros dos profesores cuando estos coinciden, y calcula el kappa de Cohen. También da la AUC de cada juez y la diferencia entre cada par con los mismos remuestreos de hablantes (intervalo pareado).
""")

add_code("""
analysis = load_json(OUT / "ompal" / "metrics_analisis.json")
display(pd.DataFrame(analysis["techo_humano_medio"]).T)
print("kappa medio:", analysis["kappa_medio"])
segments = pd.DataFrame(analysis["segmentos"])
display(segments.pivot_table(index=["tono", "segmentos", "n"], columns="juez", values="tasa"))
pd.read_csv(OUT / "ompal" / "auc_pareada.csv").set_index("comparacion")
""")

add_markdown("""
- **La etiqueta es ruidosa.** Entre profesores, el kappa medio es 0,45. HuBERT llega a 0,35 con cada profesor y la CNN a 0,22. Un profesor frente a los otros dos detecta solo el 58 % de los errores, con un 5 % de falsas alarmas: los profesores avisan mucho menos que cualquier juez. Supuesto sin verificar: que la posición de cada voto en el JSON sea siempre el mismo profesor.
- **La ventaja de HuBERT en AUC es real pero pequeña:** +0,028 sobre la CNN y +0,029 sobre la frase, con intervalos que no incluyen el 0. La CNN y la frase son indistinguibles (−0,001 [−0,012; 0,011]).
- **HuBERT no parece decidir por la palabra.** Si lo hiciera, dejaría pasar más errores de tono cuando la consonante y la vocal están bien, porque la palabra se reconoce. Con los segmentos bien detecta el 86 % y con los segmentos mal el 78 %: no se ve ese patrón.

### 8.4 El umbral: con la misma falsa alarma, los jueces se parecen

Con argmax, cada juez trabaja en un punto distinto: la CNN avisa mucho y HuBERT poco. `44` fija para cada juez el umbral de p(tono esperado) que da una falsa alarma objetivo en el train de cada una de las **5 particiones oficiales** de OMPAL, y lo mide en su test. Las particiones separan hablantes (42 en train, 5 en test), pero sus test se solapan entre sí: se resume con la media por partición.
""")

add_code("""
threshold = pd.read_csv(OUT / "ompal" / "umbral" / "particiones.csv")
threshold.groupby(["juez", "regla"], sort=False)[["umbral", "detecta", "falsa_alarma", "falsa_alarma_nativos"]].mean()
""")

add_markdown("""
Al 10 % de falsa alarma, la CNN detecta 0,52, la frase 0,49 y HuBERT 0,53. Al 20 %, 0,71, 0,69 y 0,78. **Buena parte de la ventaja de HuBERT con argmax es de punto de trabajo, no de separar mejor.** Además, pedir pocas falsas alarmas cuesta caro: al 10 %, los tres dejan pasar casi la mitad de los errores. Los umbrales salen muy bajos (0,005 a 0,025): los jueces dan una probabilidad casi nula al tono esperado en muchas sílabas bien dichas.

### 8.5 Entrenar con errores reales

Ningún juez ha oído a un alumno al entrenar. `45` y `46` entrenan directamente la pregunta de la app, «¿la sílaba lleva bien el tono esperado?», con la etiqueta de los profesores. Es una regresión logística por tono esperado (`tonos.error_detector`), con C elegido por validación cruzada agrupada por hablante. El umbral se fija con predicciones fuera de fold del train, en las que cada fold elige también su propia C, para que no salga optimista. Se prueban tres entradas: la probabilidad del tono esperado de los tres jueces, el embedding de HuBERT de la sílaba (la misma capa que el juez) y ese embedding más el contorno de F0.
""")

add_code("""
detector = pd.read_csv(OUT / "ompal" / "detector" / "particiones.csv")
display(detector.drop(columns="particion").groupby("modelo", sort=False).mean())
detector.pivot(index="particion", columns="modelo", values="detecta_0.10")
""")

add_markdown("""
**Entrenar con alumnos es lo que más mejora.** El detector sobre HuBERT detecta 0,68 al 10 % de falsa alarma, frente a 0,53 del juez HuBERT sin entrenar, y su AUC pasa de 0,851 a 0,880. Gana en las 5 particiones. Recombinar los tres jueces existentes da la mitad de la mejora (0,61). El contorno de F0 no añade nada al embedding (0,66). La falsa alarma medida en test va de 0,07 a 0,16 para un objetivo de 0,10: con 4 alumnos por test, varía mucho entre particiones. Entrenado con los 46 alumnos (`47`), fuera de fold da AUC 0,890 y detecta el 69 % con el 10 % de falsas alarmas.

Se usa una logística sobre el embedding congelado y no un ajuste fino de HuBERT: 46 alumnos y unos 2.400 errores no dan para ajustar 95 millones de parámetros sin sobreajustar.

### 8.6 El mejor juez con errores reales, el peor con errores fabricados

`48` pasa a los cuatro jueces por los errores fabricados de la demo: las 24 frases de `35`, cada sílaba tal cual y con cada uno de los otros tres tonos impuestos con WORLD.
""")

add_code("""
fabricated = load_json(OUT / "ompal" / "detector" / "demo_fabricados.json")
print(fabricated["n"])
pd.DataFrame(fabricated["jueces"]).T
""")

add_markdown("""
**El detector entrenado con alumnos es el que menos cambios detecta (30 %).** Es justo el que más acierta con errores reales. Aprendió cómo suena un alumno que se equivoca, y un tono trasplantado sobre la voz de un nativo no se parece a eso. Esta explicación es una hipótesis sin verificar.

Las tasas de los jueces de tono en la demo no coinciden con las de `34` (CNN 0,88 frente a 0,675). La demo imponía antes el contorno real más cercano a la media de cada tono, el prototipo que `34` había descartado por inflar la detección, y con él la CNN llegaba al 96 %. Desde el 21 de noviembre usa contornos reales al azar, como `34`, y la diferencia se reduce pero no desaparece. No la explican ni el criterio de selección de frases ni los 8 hablantes de la demo: en `34`, con esas mismas condiciones, la CNN detecta el 66 % y el 69 %. **Sirve para comparar el orden de los jueces, no como estimación.** La cifra de referencia es la de `34`.

### 8.7 El juez de Practicar: solo una cota

Practicar juzga sílabas sueltas con la CNN de Tone Perfect, y OMPAL no tiene sílabas sueltas: su frase más corta tiene 5 sílabas. No hay una validación limpia. `49` aplica el mismo modelo y la misma regla que la app (avisa si el tono más probable no es el esperado) a las sílabas de OMPAL. Es una cota inferior: mezcla el error de juzgar a alumnos con el salto de sílaba aislada a habla continua, que en AISHELL-3 ya bajaba su accuracy de 0,988 a 0,566.
""")

add_code("""
practice = load_json(OUT / "ompal" / "practicar" / "metrics.json")
display(pd.DataFrame({k: v["tasa"] if isinstance(v, dict) else v for k, v in practice["regla_de_la_app"].items()}, index=["practicar"]))
print("AUC:", practice["auc"], "| diferencia pareada:", practice["auc_practicar_menos_cnn_demo"])
pd.DataFrame(practice["detecta_con_falsa_alarma_0.10"]).T
""")

add_markdown("""
**El juez de Practicar no sirve para habla continua, y ahí es peor que la CNN de la demo.** Con los nativos de OMPAL solo acierta el 52 % y marca como error el 48 % de sus sílabas, así que el salto de dominio domina el resultado. Con alumnos da un 47 % de falsas alarmas y AUC 0,780, frente a 0,831 de la CNN entrenada con AISHELL-3 (diferencia pareada −0,052 [−0,063; −0,041]). Con un 10 % de falsas alarmas detecta el 39 % de los errores, frente al 52 %.

Esto no dice cómo se porta con una sílaba suelta, que es lo que graba Practicar. Lo que sí dice es que el juez de la app no tiene ninguna validación con errores reales en su propio formato, y que la pieza que más se ha estudiado en este trabajo (los jueces de frase) no es la que usa Practicar.

## 9. La tesis: tres evaluaciones, tres órdenes

Los mismos jueces, medidos de tres formas:
""")

add_code("""
swap_rates = rates.pivot(index="modelo", columns="metrica", values="tasa")
threshold_10 = threshold[threshold.regla == "fa_0.10"].groupby("juez").detecta.mean()
detector_means = detector.drop(columns="particion").groupby("modelo").mean()
detector_means.index = detector_means.index.str.replace("juez_", "").str.replace("detector_hubert$", "detector", regex=True)
demo_detection = pd.Series({judge: rates["detecta"] for judge, rates in fabricated["jueces"].items()})
judges = ["cnn_silaba", "frase", "hubert", "detector"]
# Cada columna sale de una sola fuente, para no comparar cifras de experimentos distintos.
pd.DataFrame({
    "accuracy con nativos (34)": swap_rates["acierto_original"],
    "detecta errores fabricados (34)": swap_rates["detecta_cambio"],
    "detecta errores fabricados en la demo (48)": demo_detection,
    "AUC con errores reales, media por partición (46)": detector_means["auc"],
    "detecta errores reales al 10 % de falsa alarma (46)": detector_means["detecta_0.10"],
}).reindex(judges)
""")

add_markdown("""
| Evaluación | Orden |
|---|---|
| Accuracy con nativos | HuBERT > frase > CNN |
| Errores fabricados | CNN > frase > HuBERT |
| Errores reales (AUC) | HuBERT > CNN ≈ frase |
| Errores reales, con la misma falsa alarma | empate entre los tres; el detector entrenado, muy por encima |

El detector no está en `34` (sus frases no se pueden volver a juzgar sin repetir la resíntesis): su comparación con errores fabricados es dentro de la demo (`48`), donde todos los jueces detectan más que en `34`.

**Tesis: cómo se evalúa a un juez de tonos decide qué juez parece el mejor.** Con nativos, con errores fabricados y con errores reales de alumnos, los mismos jueces quedan en órdenes distintos. Solo los errores reales miden lo que la app necesita, y entrenar con ellos es lo que más mejora al juez.

- **Se conserva** de la versión anterior: la accuracy con nativos no mide la validez de un juez.
- **Se cae:** «HuBERT deja pasar los errores del alumno porque usa el léxico». Era un efecto de los errores fabricados.
- **Se añade:** con errores reales, buena parte de la diferencia entre jueces depende del umbral. Lo que de verdad mejora es entrenar con errores de alumnos, y el juez entrenado así es el peor con errores fabricados: los errores fabricados no sustituyen a los reales.

Hipótesis sin verificar: el orden se invierte porque WORLD solo cambia el F0 y deja intactos la duración, la intensidad y la articulación de nativo, mientras que un alumno que se equivoca de tono probablemente también duda o alarga la sílaba, y HuBERT puede captarlo.
""")

add_markdown("""
## 10. De los resultados a la app

Los hallazgos se han llevado a dos pestañas nuevas del frontend:

- **«¿Quién te corrige?»** (`src/tonos/lab.py`, `/api/lab/*`, paquete de `35_preparar_laboratorio.py`). Eliges una sílaba de una frase real, le cambias el tono y ves en vivo el veredicto de cinco jueces: los tres de este notebook, el detector entrenado con alumnos de OMPAL (sección 8.5) y el léxico.
  - **Sin trampas:** los modelos se entrenan sin el fold 0 y las 24 frases de la demo salen solo de sus 8 hablantes, elegidas al azar sin mirar qué aciertan los modelos.
  - **Contorno impuesto:** una de 200 sílabas reales de entrenamiento de ese tono, elegidas al azar como en `34`, fija para cada sílaba y tono. Hasta el 21 de noviembre era la más cercana a la media del tono, que inflaba la detección de los jueces de F0 (sección 8.6).
  - **Coste:** un juicio tarda ~0,55 s en CPU. Es opcional en la API porque añade HuBERT (~380 MB) al despliegue.
- **«Resultados»** (`36_exportar_resultados.py`). Página estática con las cifras de esta escalera, leídas de `outputs/` y no copiadas a mano, más 4 ejemplos al azar para escuchar.

En la práctica se añadió además una **guía tarareada** (`06_exportar_guia_tonos.py`, `frontend/src/audio/toneGuide.ts`). Antes de grabar se puede escuchar la melodía del tono objetivo a la altura de la propia voz. Usa el mismo contorno que dibuja la gráfica y la duración mediana de cada tono en Tone Perfect (T1 0,64 s, T2 0,71 s, T3 0,84 s, T4 0,50 s). La duración no entra en el modelo porque no transfiere entre hablantes, pero ayuda a imitar.

Se verificó capturando el audio en Edge y midiendo su F0 con Praat (18 de noviembre, no guardado en `outputs/`): error frente al contorno esperado de 0,02 a 0,66 st. El primer timbre (diente de sierra) sonaba a trompeta y se cambió por armónicos que caen como 1/n², que suenan a «mmm» tarareado. La voz nativa de Tone Perfect se sirve con su licencia, CC BY-NC 4.0, confirmada el 18 de noviembre: con atribución y sin uso comercial.
""")

add_code("""
guide = load_json(ROOT / "frontend" / "src" / "data" / "tone_guide.json")
print("duración de la guía por tono (s):", {t: g["duration_s"] for t, g in guide.items()})
lab_results = load_json(ROOT / "frontend" / "src" / "data" / "resultados.json")
pd.DataFrame([{"texto": e["frase"], "sílaba": f"{e['silaba']} {e['pinyin']}",
               "cambio": f"T{e['tono_original']} -> T{e['tono_impuesto']}",
               **{v["juez"]: v["tono"] if v.get("avisa") is None else ("avisa" if v["avisa"] else "no avisa")
                  for v in e["veredictos"]}} for e in lab_results["ejemplos"]])
""")

add_markdown("""
Los cuatro ejemplos de la pestaña Resultados se eligen al azar con semilla, sin mirar qué jueces aciertan. Muestran casos distintos de quién detecta el cambio y quién no.

## 11. Decisiones y límites

| Decisión | Motivo | Alternativa descartada |
|---|---|---|
| Medir el cambio de dominio por peldaños (estilo de habla primero, hablantes no nativos después) | Saber qué rompe qué | Saltar directamente a hispanohablantes |
| Mismo pipeline de F0 en los dos corpus, F0 sobre la frase completa y rejilla desde t = 0 | Que la caída no se pueda atribuir a una entrada construida de otra forma; Praat necesita contexto en los bordes | Cortar la sílaba antes de calcular el F0; ignorar el desfase de Praat (2-3 frames) |
| Control en dominio con el tamaño de Tone Perfect | Separar dominio de volumen: los datos aportan 2 puntos | Comparar solo con el modelo entrenado con todo |
| Curva por número de hablantes con subconjuntos anidados (script 37) | Separar voces de sílabas: con más voces se gana más que con más sílabas | Subconjuntos al azar en cada punto (la diferencia dependería de qué hablantes tocan) |
| Temperatura reajustada por fold sin fuga | La calibración no viaja: ECE 0,32 → 0,017 | Reutilizar la temperatura de Tone Perfect |
| IC por bootstrap de hablantes, pareados | La unidad independiente es el hablante | Bootstrap de sílabas (intervalos falsamente estrechos) |
| Línea base léxica sin audio | Es el atajo que un modelo de habla puede explotar | Comparar modelos solo entre sí |
| Regresión logística de HuBERT en GPU (L-BFGS, 100 iteraciones) | 325 ajustes viables; misma accuracy que con 300 | sklearn (horas) |
| Contrafactuales con control de resíntesis y comprobación de la manipulación | Separar el efecto de la manipulación de los artefactos de WORLD y verificar que se hizo | Fiarse de que la resíntesis cambia lo que se pretende |
| Contorno impuesto de una sílaba real de otro fold | La media del tono es un prototipo que infla la detección de los modelos de F0 | Imponer la media del tono |
| La CNN de Tone Perfect sigue siendo el juez de Practicar | Es el único juez de sílaba suelta. Se eligió por el contrafactual (detecta el 67,5 %), que OMPAL no confirma. En habla continua de OMPAL es peor que la CNN de la demo (AUC 0,780 frente a 0,831), una cota que no mide su formato | Cambiar de juez sin datos de sílaba suelta con alumnos |
| Validar con errores reales de OMPAL, con sus 5 particiones oficiales | Es la única forma de saber si un juez sirve para corregir; las particiones separan hablantes | Seguir solo con contrafactuales |
| Umbral fijado por falsa alarma objetivo con alumnos | Compara a los jueces en el mismo punto de trabajo y prioriza no frustrar al alumno | Argmax (cada juez en un punto distinto) o máximo de Youden |
| Detector: logística por tono sobre el embedding congelado, umbral fuera de fold | 46 alumnos no dan para ajustar HuBERT; el umbral fuera de fold no sale optimista | Ajuste fino de HuBERT; umbral sobre las predicciones del propio train |
| Contornos donantes de la demo al azar, no el prototipo del tono | El prototipo inflaba la detección de la CNN (96 % frente a 67,5 %) | Mantener el prototipo |

**Límites:**
- **Semillas:** tres semillas de entrenamiento en 29, 30 y 34, con los mismos folds y datos. No se ha variado el reparto de hablantes en folds ni la selección de frases del contrafactual.
- **Etiquetas y alineamiento:** AISHELL-3 tiene un ~1-2 % de ruido de etiqueta y el alineamiento solo se revisó a mano en 30 bisílabos.
- **Los contrafactuales no son errores reales, y OMPAL lo confirma:** ordenan a los jueces al revés que los errores de alumnos. WORLD introduce además artefactos pequeños (−1 a −2 puntos en el control). El F0 plano no separa léxico, pistas no tonales y sesgo hacia T1.
- **AISHELL-3 es habla leída de nativos,** no de aprendices: mide lo frágil que es el modelo y si su decisión depende del pitch, no su rendimiento con alumnos.
- **OMPAL no son hispanohablantes.** Son 46 francófonos que leen frases, probablemente en mandarín de Taiwán. Que los resultados valgan para hispanohablantes es una suposición.
- **El tamaño efectivo de OMPAL son los hablantes.** Cada test de partición tiene 4 alumnos y 1 nativo, casi siempre el mismo (001002), y los test se solapan. Varios análisis (41, 43, 44 y 46) usan la misma muestra. La ventaja de AUC de HuBERT sobre la CNN tiene intervalo pareado y no incluye el 0, pero es pequeña (+0,028); diferencias de ese orden dependen de estos 46 alumnos.
- **La etiqueta de error es ruidosa** (kappa 0,45 entre profesores) y **el alineamiento se revisó con un veredicto global**, no sílaba a sílaba.
- **El juez de Practicar no se ha validado con errores reales en su formato:** OMPAL solo tiene frases. En ellas, como cota inferior, da un 47 % de falsas alarmas a alumnos (sección 8.7).
- **Las tasas de la demo no son una evaluación** (sección 8.6).
- **Hispanohablantes pendientes:** la validación definitiva serían sus errores reales, y no se han podido grabar a tiempo.
""")

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}})
nbformat.write(nb, args.out)
print(f"{len(cells)} celdas -> {args.out}")
