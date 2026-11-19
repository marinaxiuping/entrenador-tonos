# Contexto del proyecto

- Alcance y calendario: `alcance-tfm.md`. Setup y scripts: `README.md`.
- `progress.md`: estado actual, cómo retomar y siguientes pasos.
- `decisions.md`: decisiones técnicas con fecha y motivo.
- `notebooks/01_tone_perfect_pipeline.ipynb`: explicación de cada paso y decisión de Tone Perfect. Si cambian los resultados, regenerarlo y revisar las cifras que cita el texto.
- `notebooks/02_escalera_dominio.ipynb` (lo genera `scripts/38_generar_notebook_02.py`): escalera de dominio (AISHELL-3), contexto, HuBERT frente al léxico, contrafactuales y su paso a la app. Mismo criterio: si cambian los resultados, regenerarlo y revisar las cifras del texto.

## Datos clave
- Tone Perfect (MSU): 9.839 mp3 + xml de metadatos en `data/raw/toneperfect/` (no versionado). 6 hablantes (FV1-3, MV1-3), 410 sílabas x 4 tonos, balance 25 % por tono. Falta `jiao1_FV3`.
- Duración media por tono: T1 0,69 s, T2 0,74 s, T3 0,86 s, T4 0,55 s. Es un atajo que no transfiere entre hablantes: los modelos no la usan.
- Entorno Windows: `python`/`python3` sueltos abren el stub de la Store y se cuelgan; usar siempre `uv run python`. La consola es cp1252: evitar caracteres como `→` en los `print`.
- Tesis del TFM (2026-11-18): la accuracy con nativos no mide la validez de un juez de pronunciación. En el contrafactual de cambio de tono, HuBERT (0,97 con nativos) deja pasar el 47 % de los errores; la CNN por sílaba (0,76) solo el 10 %. Estable en 3 semillas. **En revisión (2026-11-19):** con errores reales de OMPAL sale al revés (HuBERT AUC 0,859 y 25 % de falsas alarmas frente a CNN 0,831 y 42 %), pendiente de validar el alineamiento.
- Licencias: Tone Perfect CC BY-NC 4.0 (atribución, sin uso comercial; fuente en `sources/`); AISHELL-3 Apache 2.0; OMPAL CC BY 4.0 (fuente en `sources/`).
- OMPAL: 46 alumnos francófonos + 3 nativos, caracteres tradicionales (probablemente mandarín de Taiwán), tono correcto/incorrecto por carácter con 3 votos; sin tiempos (alineado con MMS_FA en `40`). En `data/raw/ompal/`.
- Repo git en `main`, sin remoto. `/data/` y `/outputs/` ignorados solo en la raíz (los JSON de `frontend/src/data/` sí se versionan).
- `sources/`: fuentes del proyecto en .md con `INDEX.md` (regla de la usuaria).
