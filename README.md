# TFM — Entrenador de tonos del mandarín

Alcance y calendario: [alcance-tfm.md](alcance-tfm.md).

## Setup
```powershell
uv sync
```

## Datos
- Tone Perfect: descargar manualmente y descomprimir en `data/raw/toneperfect/` (en cualquier estructura de subcarpetas).

## Scripts
| Script | Qué hace |
|---|---|
| `scripts/01_explorar_toneperfect.py` | Índice del corpus: hablantes, muestras, balance de tonos, duraciones → `outputs/exploracion/` |
| `scripts/02_contornos_f0.py --method pyin\|praat` | F0 en dos pasadas (rango adaptado a cada hablante contra saltos de octava) → semitonos respecto a la media del hablante → contornos por tono → `outputs/contornos/` |
| `scripts/03_features_toneperfect.py --method praat` | F0 corregido + features por sílaba y contornos de longitud fija → `data/processed/feats_toneperfect_<method>.parquet`, `contornos_toneperfect_<method>.npz` |
| `scripts/05_exportar_silabas_toneperfect.py` | Sílabas de práctica (solo combinaciones que son palabras reales, con pinyin y caracteres) → `frontend/src/data/syllables.json` |
| `scripts/06_exportar_guia_tonos.py` | Guía tarareada de la práctica: contorno objetivo (el mismo del modelo) y duración mediana de cada tono → `frontend/src/data/tone_guide.json` |
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
| `scripts/09_seleccionar_hablantes.py --per-gender 20` | Elige hablantes de AISHELL-3 equilibrados por género → `data/aishell3_hablantes.txt` |
| `scripts/10_descargar_aishell3.py [--speakers ...]` | Descarga parcial de AISHELL-3 desde HF (metadatos y los hablantes indicados) |
| `scripts/11_analizar_etiquetas_aishell3.py` | Palabras prosódicas, bisílabos, pares de tonos y comparación realizado vs canónico |
| `scripts/12_alinear_aishell3.py --speakers ...` | Alineamiento silábico MMS_FA + `fill_gaps` → bisílabos con tiempos + material de revisión |
| `scripts/13_extraer_f0_aishell3.py --method pyin\|crepe` | F0 por frase completa → `data/interim/f0/<method>/<speaker>.npz` |
| `scripts/14_features_aishell3.py --method ...` | Features de F0 por sílaba (+ contexto de la otra sílaba) → `data/processed/feats_<method>.parquet` |
| `scripts/20_baseline_gbm.py --method ...` | LightGBM con CV por hablante (5 folds equilibrados por género) → `outputs/baseline/<method>/` |

### Pipeline del baseline (40 hablantes)
```powershell
uv run python scripts/09_seleccionar_hablantes.py --per-gender 20
uv run python scripts/10_descargar_aishell3.py --speakers-file data/aishell3_hablantes.txt
uv run python scripts/12_alinear_aishell3.py --speakers-file data/aishell3_hablantes.txt --out outputs/aishell3/alineamiento_40
uv run python scripts/13_extraer_f0_aishell3.py --method pyin  --speakers-file data/aishell3_hablantes.txt
uv run python scripts/13_extraer_f0_aishell3.py --method crepe --speakers-file data/aishell3_hablantes.txt
uv run python scripts/14_features_aishell3.py --method pyin    # y --method crepe
uv run python scripts/20_baseline_gbm.py --method pyin         # y --method crepe
```

## Notebook
`notebooks/01_tone_perfect_pipeline.ipynb` explica cada paso y decisión del pipeline de Tone Perfect (de los datos al MVP). `notebooks/02_escalera_dominio.ipynb` lo continúa: cambio de dominio a AISHELL-3, contexto, HuBERT frente a una línea base léxica, los contrafactuales de F0 plano y de cambio de tono, y su paso a la app (scripts 06, 15, 16 y 29-36). Lee los resultados de `data/` y `outputs/`, así que antes hay que ejecutar los scripts. Para volver a ejecutarlo (unos 30 s; las dependencias de desarrollo se instalan con `uv sync`):
```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_tone_perfect_pipeline.ipynb
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

## Código
- `src/tonos/data.py`: parseo de los nombres de fichero de Tone Perfect (`<sílaba><tono>_<hablante>_MP3.mp3`).
- `src/tonos/f0.py`: recorte de silencios, F0 (pYIN / Praat; CREPE pendiente), normalización por hablante y contorno de longitud fija.

## Docs
- [docs/corpus-bisilabos.md](docs/corpus-bisilabos.md): punto de control de los bisílabos.
- [docs/reclutamiento.md](docs/reclutamiento.md): mensaje, consentimiento y protocolo de grabación.
