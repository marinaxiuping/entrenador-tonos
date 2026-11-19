# TFM — Entrenador de tonos del mandarín

Alcance y calendario: [alcance-tfm.md](alcance-tfm.md).

## Setup
```powershell
uv sync
uv run pytest   # tests de las funciones del pipeline de F0, de AISHELL-3 y de la resíntesis
```

## Datos
- Tone Perfect: descargar manualmente de https://listings.lib.msu.edu/toneperfect/ y descomprimir en `data/raw/toneperfect/` (en cualquier estructura de subcarpetas).
  - Licencia: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (atribución obligatoria, sin uso comercial). La voz nativa de Practicar reproduce estas grabaciones y muestra la atribución.
  - Cita: Tone Perfect MP3 Dataset. Michigan State University Libraries. East Lansing, MI. https://listings.lib.msu.edu/toneperfect/ (consultado el 18 de noviembre de 2026).

## Scripts
| Script | Qué hace |
|---|---|
| `scripts/01_explorar_toneperfect.py` | Índice del corpus: hablantes, muestras, balance de tonos, duraciones → `outputs/exploracion/` |
| `scripts/02_contornos_f0.py --method pyin\|praat` | F0 en dos pasadas (rango adaptado a cada hablante contra saltos de octava) → semitonos respecto a la media del hablante → contornos por tono → `outputs/contornos/` |
| `scripts/03_features_toneperfect.py --method praat` | F0 corregido + features por sílaba y contornos de longitud fija → `data/processed/feats_toneperfect_<method>.parquet`, `contornos_toneperfect_<method>.npz` |
| `scripts/05_exportar_silabas_toneperfect.py` | Sílabas de práctica (solo combinaciones que son palabras reales, con pinyin y caracteres) → `frontend/src/data/syllables.json` |
| `scripts/06_exportar_guia_tonos.py` | Guía tarareada de la práctica: contorno objetivo (el mismo del modelo) y duración mediana de cada tono → `frontend/src/data/tone_guide.json` |
| `scripts/07_exportar_voces_referencia.py` | Altura de voz de cada hablante de Tone Perfect, medida igual que el perfil del usuario (ma1-ma4), para ofrecer primero la voz nativa más parecida → `frontend/src/data/reference_voices.json` |
| `scripts/21_baseline_toneperfect.py --method praat` | LightGBM dejando fuera un hablante (6 folds), con y sin duración → `outputs/baseline_toneperfect/<method>/` |
| `scripts/22_calibrar_toneperfect.py --method praat --set base` | Temperature scaling sobre las probabilidades out-of-fold (T de cada hablante ajustada con los otros cinco) → `outputs/calibracion_toneperfect/<method>/<set>/` |
| `scripts/23_cnn_toneperfect.py --method praat` | CNN 1D sobre el contorno de F0 (64 puntos + derivada), mismos folds que el baseline → `outputs/cnn_toneperfect/<method>/` |
| `scripts/04_embeddings_toneperfect.py` | Embeddings congelados por capa (media temporal) de `chinese-hubert-base`, fijado a un commit → `data/interim/embeddings/` |
| `scripts/24_lineal_embeddings_toneperfect.py` | Regresión logística sobre los embeddings; capa elegida con validación interna por hablante → `outputs/lineal_toneperfect/<modelo>/` |
| `scripts/25_entrenar_cnn_final.py --method praat` | CNN final con todos los hablantes + temperatura + contornos objetivo → `models/cnn_toneperfect_<method>.pt` |
| `scripts/26_evaluar_voz_corregida.py --per-cell 25` | Voz corregida (WORLD) hacia los 4 tonos, evaluada sin fuga por hablante + ejemplos para escuchar → `outputs/voz_corregida/` |
| `scripts/27_contorno_objetivo_personalizado.py` | Plantilla fija frente a Ridge sobre duración y rango para el contorno objetivo → `outputs/contorno_objetivo/` |
| `scripts/28_evaluar_calibracion.py` | Conjuntos de sílabas de calibración: desviación de la referencia y accuracy sin fuga → `outputs/calibracion_perfil/` |
| `scripts/15_contornos_aishell3.py` | Contornos de F0 por sílaba de AISHELL-3 con el mismo pipeline que Tone Perfect (F0 por frase, cortado con el alineamiento de `12`) → `data/processed/silabas_aishell3_praat.parquet`, `contornos_aishell3_praat.npz` |
| `scripts/16_embeddings_aishell3.py` | Embeddings congelados de `chinese-hubert-base` sobre la frase completa, media por capa dentro de cada sílaba → `data/interim/embeddings/aishell3_chinese-hubert-base/layer_NN.npy` |
| `scripts/29_escalera_dominio_aishell3.py` | Cambio de dominio Tone Perfect → AISHELL-3: cero disparos, en dominio con tamaño de Tone Perfect y completo, calibración y desglose por factores → `outputs/escalera_dominio/aishell3/` |
| `scripts/30_contexto_f0_aishell3.py` | Contexto sobre el F0: CNN con ventana ±1 y CNN por sílaba + BiGRU sobre la frase → `outputs/escalera_dominio/aishell3_contexto/` |
| `scripts/31_hubert_aishell3.py` | HuBERT congelado + regresión logística (GPU), capa elegida por validación interna → `outputs/escalera_dominio/aishell3_hubert/` |
| `scripts/32_comparar_contexto_aishell3.py` | Comparación de 29-31 + línea base léxica sin audio, IC 95 % por bootstrap de hablantes, desglose y solapamiento de errores → `outputs/escalera_dominio/comparacion_contexto/` |
| `scripts/33_contrafactual_f0_plano.py` | Contrafactual: HuBERT con el F0 aplanado con WORLD (+ control de resíntesis y comprobación de la manipulación) frente al léxico sin audio → `outputs/escalera_dominio/contrafactual_f0_plano/` |
| `scripts/34_contrafactual_cambio_tono.py` | Contrafactual del caso de la app: se impone a una sílaba el contorno real de otro tono y se mide si cada modelo (CNN, frase, HuBERT, léxico) detecta el cambio → `outputs/escalera_dominio/contrafactual_cambio_tono/` |
| `scripts/35_preparar_laboratorio.py` | Paquete de la demo «¿quién te corrige?»: CNN, BiGRU y lineal de HuBERT entrenados sin un fold, 24 frases de sus hablantes, perfiles, contornos donantes y léxico → `models/laboratorio/` |
| `scripts/36_exportar_resultados.py` | Cifras de los scripts 29 y 32-34 y ejemplos al azar con el juez de la demo para la pestaña Resultados → `frontend/src/data/resultados.json`, `frontend/public/resultados/` |
| `scripts/37_curva_hablantes_aishell3.py` | Curva de aprendizaje con 4, 8, 16 y 32 hablantes de entrenamiento (CNN por sílaba y frase), subconjuntos anidados → `outputs/escalera_dominio/curva_hablantes/`. Solo se ha ejecutado el fold 0 y no se reanuda |
| `scripts/38_generar_notebook_02.py` | Genera `notebooks/02_escalera_dominio.ipynb` sin ejecutar; su texto cita cifras de los scripts, así que hay que revisarlo si cambian los resultados |
| `scripts/39_resumir_semillas.py` | Media y desviación de 29, 30 y 34 entre semillas de entrenamiento (`--train-seed`: mismos folds y datos, otra inicialización) y si el orden de los jueces se mantiene → `outputs/escalera_dominio/semillas/resumen.csv` |
| `scripts/09_seleccionar_hablantes.py --per-gender 20` | Elige hablantes de AISHELL-3 equilibrados por género → `data/aishell3_hablantes.txt` |
| `scripts/10_descargar_aishell3.py [--speakers ...]` | Descarga parcial de AISHELL-3 desde HF (metadatos y los hablantes indicados) |
| `scripts/11_analizar_etiquetas_aishell3.py` | Palabras prosódicas, bisílabos, pares de tonos y comparación realizado vs canónico |
| `scripts/12_alinear_aishell3.py --speakers ...` | Alineamiento silábico MMS_FA + `fill_gaps` → bisílabos con tiempos + material de revisión |

### Datos de AISHELL-3 (40 hablantes)
Preparan el alineamiento del que parten los scripts 15 y siguientes.
```powershell
uv run python scripts/09_seleccionar_hablantes.py --per-gender 20
uv run python scripts/10_descargar_aishell3.py --speakers-file data/aishell3_hablantes.txt
uv run python scripts/12_alinear_aishell3.py --speakers-file data/aishell3_hablantes.txt --out outputs/aishell3/alineamiento_40
```

## Notebook
`notebooks/01_tone_perfect_pipeline.ipynb` explica cada paso y decisión del pipeline de Tone Perfect (de los datos al MVP). `notebooks/02_escalera_dominio.ipynb` lo continúa: cambio de dominio a AISHELL-3, contexto, HuBERT frente a una línea base léxica, los contrafactuales de F0 plano y de cambio de tono, y su paso a la app (scripts 06, 15, 16 y 29-36). Lee los resultados de `data/` y `outputs/`, así que antes hay que ejecutar los scripts. Para volver a ejecutarlo (unos 30 s; las dependencias de desarrollo se instalan con `uv sync`):
```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_tone_perfect_pipeline.ipynb
uv run python scripts/38_generar_notebook_02.py   # solo si se ha cambiado su texto
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_escalera_dominio.ipynb
```

## App (MVP)
Requiere Node 24 además de `uv`.
```powershell
uv run python scripts/25_entrenar_cnn_final.py --method praat   # genera models/cnn_toneperfect_praat.pt
copy .env.example .env
cd frontend; npm ci; npm run build; cd ..
uv run --env-file .env uvicorn tonos.api:app --port 8000          # app en http://localhost:8000
```
El backend sirve el frontend compilado (`frontend/dist`) en `/` y la API en `/api`, desde el mismo origen. Para desarrollar el frontend con recarga en caliente, con el backend arrancado: `cd frontend; npm run dev` (Vite reenvía `/api` a `API_PROXY_TARGET`, ver `frontend/.env.example`).

API:
1. `POST /api/calibrate` con `files` = una sílaba en los cuatro tonos (p. ej. mā má mǎ mà) → `reference_hz`, `fmin`, `fmax`. El cliente guarda este perfil.
2. `POST /api/analyze` con `file` + el perfil (+ `target_tone` opcional) → tono, probabilidades calibradas con Tone Perfect (en habla continua no lo están), contorno del usuario y contorno objetivo (64 puntos, semitonos respecto a su media).
3. `POST /api/resynthesize` con `file` + el perfil + `target_tone` → WAV con la grabación del usuario y el contorno del tono objetivo (WORLD).

Formatos: wav, flac, ogg o mp3. El frontend graba con el formato nativo del navegador y lo convierte a wav antes de enviarlo. Documentación interactiva en `/docs`.

### Pestañas «¿Quién te corrige?» y «Resultados»
- **¿Quién te corrige?** (demo en vivo): eliges una sílaba de una frase real de AISHELL-3, le cambias el tono (WORLD) y ves qué dicen la CNN por sílaba, la CNN + BiGRU, HuBERT y el léxico. Los modelos se entrenan sin los hablantes de las frases de la demo. Necesita el paquete `models/laboratorio/` (`TONOS_LAB_DIR`); si no existe, la API arranca sin estas rutas. Al arrancar descarga `chinese-hubert-base` (~380 MB) la primera vez.
  - `GET /api/lab/utterances` → frases con sus sílabas.
  - `POST /api/lab/judge` con JSON `{utterance_id, syllable_index, imposed_tone}` (`null` = sin cambios) → audio WAV en base64, contornos y veredicto de cada juez.
- **Resultados**: estática (`frontend/src/data/resultados.json` y `frontend/public/resultados/`), no usa la API.

Para regenerarlas (requiere los scripts 15, 16 y 29-34):
```powershell
uv run python scripts/35_preparar_laboratorio.py   # models/laboratorio/
uv run python scripts/36_exportar_resultados.py    # datos y audios de Resultados
cd frontend; npm run build; cd ..
```
El audio de las frases de la demo y de los ejemplos es de AISHELL-3 (licencia Apache 2.0).
Las traducciones al castellano de esas frases están escritas a mano en `frontend/src/data/sentence_translations.json` (AISHELL-3 no trae traducción); si `35` elige otras frases, hay que añadirlas o se muestran sin traducir.

### Límite de uso
`/api/resynthesize` y `/api/lab/judge` admiten `TONOS_HEAVY_REQUESTS_PER_MINUTE` peticiones por minuto y cliente entre las dos (30 por defecto); por encima responden 429 con `Retry-After`. La cuenta vive en memoria, por proceso.

## Despliegue (Docker, CPU)
El `Dockerfile` construye una imagen para Hugging Face Spaces (Docker, puerto 7860): compila el frontend, instala las versiones de `uv.lock` con torch de CPU y descarga HuBERT al construir, así que arranca sin red. Entran la práctica, la voz nativa y la demo; necesita `models/laboratorio/` (script 35) y las grabaciones de Tone Perfect en `data/raw/toneperfect/tone_perfect/`, que no se versionan.
```powershell
docker build -t tonos-app .
docker run --rm -p 7860:7860 tonos-app   # http://localhost:7860
```
La imagen ocupa 1,3 GB. El primer juicio de la demo tarda ~10 s (carga de HuBERT en CPU); los siguientes, menos de 1 s. La imagen redistribuye las grabaciones de Tone Perfect: el Space tiene que ser sin uso comercial y mantener la atribución (CC BY-NC 4.0).

## Código
Módulos de `src/tonos/`, de los datos a la app:
- `data.py`: índice de Tone Perfect a partir de los nombres de fichero (`<sílaba><tono>_<hablante>_MP3.mp3`).
- `f0.py`: recorte de silencios, F0 (pYIN, Praat o CREPE), corrección de saltos de octava, semitonos por hablante y contorno de longitud fija.
- `features.py`: features de F0 por sílaba para el baseline de gradient boosting.
- `splits.py`: folds por hablante equilibrados por género.
- `align.py`: alineamiento forzado por sílaba de AISHELL-3 con MMS_FA.
- `aishell3.py`: sílabas alineadas de AISHELL-3, folds y línea base léxica sin audio.
- `utterance_contours.py`: contornos de F0 de las sílabas de una frase, con el pipeline de AISHELL-3.
- `cnn.py`: CNN 1D sobre el contorno de una sílaba.
- `context.py`: clasificadores que ven el contexto de la frase (ventana y CNN + BiGRU).
- `embeddings.py`: embeddings congelados por capa de un modelo de habla preentrenado (HuBERT).
- `linear_probe.py`: regresión logística multinomial en GPU para sondear esos embeddings.
- `calibration.py`: temperature scaling y métricas de calibración.
- `evaluation.py`: métricas, desgloses y figuras comunes.
- `resynthesis.py`: resíntesis con WORLD cambiando solo el F0 (voz corregida y contrafactuales).
- `inference.py`: tono de una grabación nueva y perfil de voz del usuario, sin depender del framework web.
- `reference_audio.py`: grabaciones de Tone Perfect que se sirven como voz nativa.
- `lab.py`: la demo «¿Quién te corrige?» (cambio de tono y juicio de los cuatro modelos).
- `rate_limit.py`: límite de peticiones por cliente para las rutas pesadas de la API.
- `api.py`: API FastAPI, capa fina sobre `inference`, `resynthesis`, `reference_audio` y `lab`.

## Docs
- [docs/corpus-bisilabos.md](docs/corpus-bisilabos.md): punto de control de los bisílabos.
- [docs/reclutamiento.md](docs/reclutamiento.md): mensaje, consentimiento y protocolo de grabación.
