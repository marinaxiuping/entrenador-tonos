# Contexto del proyecto

- Alcance y calendario iniciales: `docs/planificacion/alcance-tfm.md`. Setup, estructura y orden de scripts: `README.md`.
- Memoria del TFM: `docs/memoria/` (`memoria.html` → `memoria.pdf`, figuras y capturas por script).
- `progress.md`: estado actual, cómo retomar y siguientes pasos.
- `decisions.md`: decisiones técnicas con fecha y motivo.
- `notebooks/01_tone_perfect_pipeline.ipynb`: explicación de cada paso y decisión de Tone Perfect. Si cambian los resultados, regenerarlo y revisar las cifras que cita el texto.
- `notebooks/02_escalera_dominio.ipynb` (lo genera `scripts/38_generar_notebook_02.py`): escalera de dominio (AISHELL-3), contexto, HuBERT frente al léxico, contrafactuales, OMPAL y la tesis. Mismo criterio: si cambian los resultados, regenerarlo y revisar las cifras del texto.

## Datos clave
- Tone Perfect (MSU): 9.839 mp3 + xml de metadatos en `data/raw/toneperfect/` (no versionado). 6 hablantes (FV1-3, MV1-3), 410 sílabas x 4 tonos, balance 25 % por tono. Falta `jiao1_FV3`.
- Duración media por tono: T1 0,69 s, T2 0,74 s, T3 0,86 s, T4 0,55 s. Es un atajo que no transfiere entre hablantes: los modelos no la usan.
- Entorno Windows: `python`/`python3` sueltos abren el stub de la Store y se cuelgan; usar siempre `uv run python`. La consola es cp1252: evitar caracteres como `→` en los `print`.
- Tesis del TFM (reformulada el 2026-11-21, en `decisions.md`): cómo se evalúa a un juez de tonos decide qué juez parece el mejor. Accuracy con nativos: HuBERT > frase > CNN; errores fabricados (`34`): CNN > frase > HuBERT; errores reales de OMPAL (AUC): HuBERT > CNN ≈ frase, y con la misma falsa alarma empatan. El detector entrenado con alumnos es el mejor con errores reales y el peor con fabricados. Está en el notebook 02, la pestaña Resultados y la memoria.
- Licencias: Tone Perfect CC BY-NC 4.0 (atribución, sin uso comercial; fuente en `sources/`); AISHELL-3 Apache 2.0; OMPAL CC BY 4.0 (fuente en `sources/`).
- OMPAL: 46 alumnos francófonos + 3 nativos, caracteres tradicionales (probablemente mandarín de Taiwán), tono correcto/incorrecto por carácter con 3 votos; sin tiempos (alineado con MMS_FA en `40`). En `data/raw/ompal/`.
- Repo git en `main`, sin remoto todavía. App también desplegada en un Space de Hugging Face con ZeroGPU (`deploy/`). `/data/` y `/outputs/` ignorados solo en la raíz (los JSON de `frontend/src/data/` sí se versionan).
- `sources/`: fuentes del proyecto en .md con `INDEX.md` (regla de la usuaria).
