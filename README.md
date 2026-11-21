# ¿Quién corrige el tono? Entrenador de tonos del mandarín

Trabajo de Fin de Máster (Máster en Data Science & IA, Evolve Academy). Una aplicación para practicar los cuatro tonos del mandarín y un estudio de cómo se evalúa a un juez automático de pronunciación.

- **Memoria:** [docs/memoria/memoria.pdf](docs/memoria/memoria.pdf) (fuente en [docs/memoria/memoria.html](docs/memoria/memoria.html)).
- **App desplegada:** Space de Hugging Face [Eskarcho/entrenador-tonos](https://huggingface.co/spaces/Eskarcho/entrenador-tonos).
- **Notebooks:** [01, Tone Perfect](notebooks/01_tone_perfect_pipeline.ipynb) y [02, escalera de dominio](notebooks/02_escalera_dominio.ipynb), con cada paso y cada decisión.

## Resultado en una frase

Cómo se evalúa a un juez de tonos decide qué juez parece el mejor. Con nativos gana HuBERT (0,966 en habla continua); con errores fabricados cambiando el tono con WORLD gana una CNN sobre el contorno de F0 (detecta el 67,5 % de los cambios frente al 31 %); con errores reales de 46 alumnos (OMPAL), HuBERT vuelve a quedar primero (AUC +0,028 [0,016; 0,040]) y con la misma tasa de falsas alarmas los jueces casi empatan. Entrenar con los errores de los alumnos sube la detección del 53 % al 68 % con un 10 % de falsas alarmas.

## La aplicación

- **Practicar:** calibras tu voz con *ma* en los cuatro tonos, eliges una sílaba, grabas y ves tu curva de tono frente a la objetivo, el veredicto de una CNN calibrada, la voz de un nativo y tu propia voz corregida con WORLD.
- **¿Quién corrige?:** cambias el tono de una sílaba en una frase real de AISHELL-3 y ves qué dicen cinco jueces: CNN por sílaba, CNN + BiGRU, HuBERT, un detector entrenado con errores reales de alumnos y una línea base léxica que no oye el audio.
- **Resultados:** las cifras del trabajo, leídas de las salidas de los experimentos.

## Estructura

```
src/tonos/        paquete: F0, modelos, evaluación, inferencia, demo y API
scripts/          experimentos numerados (01-49): cada uno lee datos y escribe en outputs/
notebooks/        explicación de cada paso (el 02 lo genera scripts/38)
frontend/         React + Vite (la API sirve el build en /)
models/           modelo de Practicar y paquete de la demo (versionados)
tests/            pytest (sin modelos ni descargas)
docs/memoria/     memoria (HTML → PDF), figuras y capturas
docs/planificacion/  alcance inicial, decisión de corpus y plan de reclutamiento
deploy/           Space de Hugging Face (SDK Gradio, ZeroGPU)
sources/          fuentes del proyecto: licencias, corpus y referencias (con INDEX.md)
.claude/context/  contexto de trabajo: estado, decisiones con fecha y motivo
```

`data/` y `outputs/` no se versionan: los genera cada script.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y, para el frontend, Node 24.

```powershell
uv sync                     # versiones exactas de uv.lock (torch con CUDA)
copy .env.example .env      # configuración de la API (ver comentarios del fichero)
uv run pytest               # 47 tests, sin modelos ni descargas
cd frontend; npm ci; npm run build; cd ..
uv run --env-file .env uvicorn tonos.api:app --port 8000   # http://localhost:8000
```

La app usa los modelos versionados en `models/`. La demo descarga `chinese-hubert-base` (~380 MB) la primera vez. La voz nativa de Practicar necesita las grabaciones de Tone Perfect (opcional, `TONOS_TONE_PERFECT_DIR`).

## Datos

| Corpus | Uso | Dónde | Licencia |
|---|---|---|---|
| [Tone Perfect](https://tone.lib.msu.edu) (MSU) | Peldaño 0, modelo de Practicar, voz nativa | Descarga manual en `data/raw/toneperfect/` | CC BY-NC 4.0 |
| [AISHELL-3](https://www.openslr.org/93/) | Peldaño 1, contrafactuales, demo | `scripts/09`, `10` (descarga parcial de 40 hablantes desde Hugging Face) | Apache 2.0 |
| [OMPAL](https://github.com/phantomhsieh/OMPAL-corpus) | Peldaño 2: errores reales de alumnos | Clonar en `data/raw/ompal/OMPAL-corpus-main/` | CC BY 4.0 |

Las fichas de licencia y cita están en [sources/](sources/INDEX.md). Tone Perfect se redistribuye en la app (voz nativa): el despliegue tiene que ser sin uso comercial y con atribución.

## Reproducir los experimentos

Cada cifra de la memoria sale de un script; el anexo de la memoria dice cuál. En orden:

```powershell
# Peldaño 0: Tone Perfect
uv run python scripts/01_explorar_toneperfect.py
uv run python scripts/02_contornos_f0.py --method praat --n-per-cell 0
uv run python scripts/03_features_toneperfect.py --method praat
uv run python scripts/21_baseline_toneperfect.py --method praat
uv run python scripts/22_calibrar_toneperfect.py --method praat --set base
uv run python scripts/23_cnn_toneperfect.py --method praat
uv run python scripts/04_embeddings_toneperfect.py
uv run python scripts/24_lineal_embeddings_toneperfect.py
uv run python scripts/25_entrenar_cnn_final.py --method praat        # modelo de Practicar
uv run python scripts/26_evaluar_voz_corregida.py --per-cell 25
uv run python scripts/27_contorno_objetivo_personalizado.py
uv run python scripts/28_evaluar_calibracion.py

# Peldaño 1: AISHELL-3
uv run python scripts/09_seleccionar_hablantes.py --per-gender 20
uv run python scripts/10_descargar_aishell3.py --speakers-file data/aishell3_hablantes.txt
uv run python scripts/12_alinear_aishell3.py --speakers-file data/aishell3_hablantes.txt --out outputs/aishell3/alineamiento_40
uv run python scripts/15_contornos_aishell3.py
uv run python scripts/16_embeddings_aishell3.py
uv run python scripts/29_escalera_dominio_aishell3.py
uv run python scripts/30_contexto_f0_aishell3.py
uv run python scripts/31_hubert_aishell3.py
uv run python scripts/32_comparar_contexto_aishell3.py
uv run python scripts/37_curva_hablantes_aishell3.py
uv run python scripts/33_contrafactual_f0_plano.py
uv run python scripts/34_contrafactual_cambio_tono.py
# semillas 1 y 2 de 29, 30 y 34 con --train-seed y --out outputs/escalera_dominio/semillas/<carpeta>/semilla_<n>
uv run python scripts/39_resumir_semillas.py

# Peldaño 2: OMPAL
uv run python scripts/40_alinear_ompal.py
uv run python scripts/41_jueces_ompal.py
uv run python scripts/42_revisar_alineamiento_ompal.py   # audios para revisar el alineamiento a oído
uv run python scripts/43_analizar_ompal.py
uv run python scripts/44_umbral_ompal.py
uv run python scripts/45_embeddings_ompal.py
uv run python scripts/46_detector_errores_ompal.py
uv run python scripts/49_juez_practicar_ompal.py

# App: demo, detector, resultados y datos del frontend
uv run python scripts/35_preparar_laboratorio.py
uv run python scripts/47_detector_laboratorio.py
uv run python scripts/48_detector_errores_fabricados.py
uv run python scripts/36_exportar_resultados.py
uv run python scripts/05_exportar_silabas_toneperfect.py
uv run python scripts/06_exportar_guia_tonos.py
uv run python scripts/07_exportar_voces_referencia.py

# Notebooks, figuras y memoria
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_tone_perfect_pipeline.ipynb
uv run python scripts/38_generar_notebook_02.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_escalera_dominio.ipynb
uv run python docs/memoria/figuras.py
uv run --with playwright==1.48.0 python docs/memoria/capturas.py   # con la app arrancada
```

La memoria se imprime a PDF con Edge: `msedge --headless=new --no-pdf-header-footer --print-to-pdf=docs/memoria/memoria.pdf docs/memoria/memoria.html` (con la ruta completa del HTML como `file:///…`).

## API

| Ruta | Qué hace |
|---|---|
| `POST /api/calibrate` | Perfil de voz (`reference_hz`, `fmin`, `fmax`) a partir de una sílaba en los cuatro tonos |
| `POST /api/analyze` | Tono, probabilidades calibradas y contornos del usuario y objetivo (64 puntos, semitonos) |
| `POST /api/resynthesize` | La grabación con el contorno del tono objetivo (WORLD), en WAV |
| `GET /api/reference-audio/{sílaba}/{tono}/{hablante}` | Grabación nativa de Tone Perfect |
| `GET /api/lab/utterances`, `POST /api/lab/judge` | Frases de la demo y juicio de los cinco jueces con el tono cambiado |

Documentación interactiva en `/docs`. Formatos: wav, flac, ogg o mp3. Las rutas pesadas (`resynthesize`, `lab/judge`) admiten `TONOS_HEAVY_REQUESTS_PER_MINUTE` peticiones por minuto y cliente (30 por defecto). La API precalienta la demo al arrancar (`TONOS_LAB_WARM_UP`). Para desarrollar el frontend con recarga en caliente: `cd frontend; npm run dev` (Vite reenvía `/api`, ver `frontend/.env.example`).

## Despliegue

**Docker (CPU).** El `Dockerfile` compila el frontend, instala las versiones de `uv.lock` con torch de CPU y descarga HuBERT al construir. Necesita las grabaciones de Tone Perfect en `data/raw/toneperfect/tone_perfect/`. La imagen ocupa 4,3 GB.

```powershell
docker build -t tonos-app .
docker run --rm -p 7860:7860 tonos-app   # http://localhost:7860
```

**Hugging Face Spaces (gratuito).** Los Spaces con Docker son de pago, y el SDK Gradio gratuito solo ofrece ZeroGPU, que construye con Python 3.10 y exige al menos una función con `@spaces.GPU`. `deploy/preparar_space.py` monta en `deploy/hf_space/build/` (ignorada por git) un Space cuyo `app.py` arranca la misma API sin interfaz de Gradio y ejecuta en la GPU el juicio de la demo. Sus dependencias están en `deploy/hf_space/requirements.txt`, resuelto para Python 3.10 y regenerable con `--update-requirements`.

```powershell
cd frontend; npm run build; cd ..
uv run python deploy/preparar_space.py
uv run hf upload <usuario>/<space> deploy/hf_space/build . --repo-type space
```

## Código

Módulos de `src/tonos/`, de los datos a la app:

| Módulo | Qué contiene |
|---|---|
| `data.py` | Índice de Tone Perfect a partir de los nombres de fichero |
| `f0.py` | Recorte de silencios, F0 (pYIN, Praat o CREPE), corrección de octava, semitonos y contorno de longitud fija |
| `features.py`, `splits.py` | Rasgos de F0 para LightGBM; folds por hablante equilibrados por género |
| `align.py`, `aishell3.py`, `utterance_contours.py` | Alineamiento con MMS_FA, sílabas de AISHELL-3, línea base léxica y contornos de una frase |
| `ompal.py` | Particiones oficiales y etiqueta de error de OMPAL |
| `cnn.py`, `context.py` | CNN sobre el contorno y clasificadores con contexto de frase (ventana, BiGRU) |
| `embeddings.py`, `linear_probe.py` | Embeddings congelados de HuBERT por capa y sonda lineal en GPU |
| `error_detector.py` | Detector de errores de tono: una regresión logística por tono esperado, guardada como arrays |
| `calibration.py`, `evaluation.py` | Temperature scaling y métricas; bootstrap por hablantes, también de AUC pareada |
| `resynthesis.py` | Resíntesis con WORLD cambiando solo el F0 |
| `inference.py`, `reference_audio.py` | Tono de una grabación nueva, perfil de voz y voz nativa |
| `lab.py` | La demo «¿Quién corrige?»: cambio de tono y juicio de los cinco jueces |
| `rate_limit.py`, `api.py` | Límite de peticiones y API FastAPI, capa fina sobre lo anterior |

Los tests (`tests/`) cubren el F0, AISHELL-3, la resíntesis, la evaluación, el detector y la capa HTTP de la API con dobles de los modelos. `.github/workflows/ci.yml` los ejecuta en Linux con torch de CPU y compila el frontend.
