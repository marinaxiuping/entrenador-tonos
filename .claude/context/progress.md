# Progreso

El detalle de cada paso y decisión está en `notebooks/01_tone_perfect_pipeline.ipynb` (Tone Perfect) y `notebooks/02_escalera_dominio.ipynb` (AISHELL-3, contexto, HuBERT, contrafactuales y su paso a la app). Las decisiones con fecha y motivo, en `decisions.md`.

## Estado (actualizado el 2026-11-18)
Pilar 1 (clasificación) hecho con Tone Perfect y MVP montado en local (backend + frontend). Falta probarlo con micrófono real, desplegarlo y todo lo que depende de las grabaciones de hispanohablantes.

- Datos: Tone Perfect en `data/raw/toneperfect/` (no versionado). Los zips originales, en `data/raw/` (ignorada).
- F0: Praat + corrección de octava (rango por hablante y `fix_octave_jumps` con bordes de 3 frames y huecos ≥ 9 st). Residuo: 4,4 % de T3 y 3,5 % de T4 con saltos > 6 st que no son de octava.
- Modelos (6 folds por hablante, sin duración):
  - LightGBM `base`: 0,983 ± 0,014.
  - CNN sobre contorno: 0,988 ± 0,012 (semillas 1-2: 0,987).
  - HuBERT congelado + lineal: 0,990 ± 0,011.
  - Errores de CNN y HuBERT casi disjuntos (2 comunes de 121 y 98).
- Calibración: LightGBM T 1,65 (ECE 0,011 → 0,002); CNN T 1,28 (0,004 → 0,001); HuBERT no lo necesita.
- MVP: `models/cnn_toneperfect_praat.pt` (`25`); API FastAPI en `/api` (`/calibrate`, `/analyze`, `/resynthesize`) que sirve también `frontend/dist` en `/`; frontend React + Vite en `frontend/`.

- 2026-11-18: práctica ampliada a las 408 sílabas de Tone Perfect (`05` → `frontend/src/data/syllables.json`): 1.243 combinaciones sílaba-tono que son palabras reales, con pinyin y caracteres del corpus; los 397 huecos léxicos no se ofrecen. Botón "Aleatoria".
- 2026-11-18: voz corregida (pilar A). `src/tonos/resynthesis.py` (WORLD, solo F0 de frames con voz, ~0,1 s por sílaba), `POST /api/resynthesize` y botón en el resultado. Evaluación objetiva (`26`, sin fuga): la CNN reconoce el tono objetivo en el 99,8 % de 2.400 resíntesis, error de F0 0,2 st (circular: no mide cómo suena). Ejemplos en `outputs/voz_corregida/ejemplos/`. Modelo personalizado del contorno objetivo (`27`): no supera a la plantilla (2,18 frente a 2,13 st) → la app usa la plantilla.
- 2026-11-18: calibración del perfil (`28`): con solo "ma" la referencia se desvía hasta 2 st pero cuesta 0,4 puntos (0,984 frente a 0,988; la cifra previa de 0,970 era ruido de una muestra de 100 por hablante). Añadir yi, wu o li empeora (0,971; MV1 0,886) porque las vocales altas suben la referencia → se mantiene "ma".

- 2026-11-18: escalera de dominio, peldaño 1 (`15`, `29`): mismas sílabas de AISHELL-3 (40 hablantes, 188.743 sílabas de tonos 1-4) y mismo pipeline de F0 que Tone Perfect. Cero disparos con el modelo de la app: 0,566 (Tone Perfect 0,988); T2 recall 0,19; ECE 0,322 con la T de Tone Perfect (0,017 reajustada). En dominio con el tamaño de Tone Perfect: 0,748; con todo (14 veces más): 0,768. Los contornos se comprimen ~4 veces (recorrido del T4 10,7 → 2,8 st; T2 6,7 → 1,0 st) y el T3 pierde la subida. Peores: sílaba final de frase (0,63 en dominio), las más cortas (<181 ms) y T3 (0,63). Lectura: el volumen de datos aporta 2 puntos; el techo de ~0,77 es de la representación (contorno de una sílaba sin contexto).

- 2026-11-18: contexto y HuBERT en AISHELL-3 (`16`, `30`, `31`, `32`; mismas 188.743 sílabas y folds, IC 95 % por bootstrap de hablantes). CNN por sílaba 0,768 · ventana ±1 0,813 (+4,4 [3,9; 4,9]) · frase CNN+BiGRU 0,833 (+6,5 [5,9; 7,0]) · HuBERT congelado + lineal (capa 10 en los 5 folds) 0,965 · léxico sin audio (tono más frecuente del carácter) 0,934. Sospecha: HuBERT acierta sobre todo por el léxico, no por el pitch. El léxico falla en T3 ante T3 (0,79, sandhi) y HuBERT no (0,96), pero el sandhi también se deduce del léxico vecino. El refactor de `29` (módulos compartidos) no se ha vuelto a ejecutar; `32` sí y da las mismas cifras.
- 2026-11-18: contrafactual de F0 plano (`33`, 2.400 frases, 25.825 sílabas, clasificadores de `31` con hablante no visto). Manipulación comprobada: dispersión de F0 2,96 → 0,12 st. HuBERT original 0,964 · resíntesis WORLD sin tocar 0,953 (−1,1) · F0 plano 0,693 (−27,1 [−28,7; −25,4]) · léxico 0,932. Con F0 plano: T1 0,88 (lo plano suena a T1), T3 0,44. Lectura: HuBERT sí usa el pitch (27 puntos), pero el 69 % sobrevive sin ninguna información tonal (léxico + duración/intensidad/calidad de voz + sesgo hacia T1). Como juez de un alumno no es fiable solo. Ejemplos en `outputs/escalera_dominio/contrafactual_f0_plano/ejemplos/`.

- 2026-11-18: contrafactual de cambio de tono de una sílaba (`34`, 2.400 frases de `33`, contorno real de un donante de otro fold impuesto con WORLD; manipulación comprobada: RMSE 0,44 st al impuesto frente a 3,04 al original). Detecta el cambio / acepta el error: CNN por sílaba 0,675 / 0,100 · frase 0,598 / 0,143 · HuBERT 0,310 / 0,474 · léxico 0,022 / 0,925. El ranking se invierte respecto a la accuracy (HuBERT 0,968 > frase 0,833 > CNN 0,761): cuanto más contexto o léxico usa el modelo, más errores del alumno deja pasar. Tesis del TFM: la accuracy no mide la validez de un juez de pronunciación.

- 2026-11-18: frontend con tres pestañas: Practicar, «¿Quién te corrige?» (demo en vivo: `src/tonos/lab.py`, `/api/lab/*`, paquete de `35` entrenado sin el fold 0 y 24 frases de sus 8 hablantes) y Resultados (estática, de `36`). Verificado: build del frontend sin errores de tipos, API probada con curl (juicio en ~0,55 s en CPU, 422 en peticiones inválidas, estáticos servidos). Probada en Edge con Playwright (móvil 400 px y escritorio): las tres pestañas, juicio con tono cambiado y original, los 8 audios de ejemplo responden 200 y no hay scroll horizontal. Único error de consola: falta `favicon.ico` (ya faltaba antes).

- 2026-11-18: guía tarareada en Practicar (`06`, `frontend/src/audio/toneGuide.ts`): Web Audio sigue el contorno objetivo a la altura del perfil del usuario, con la duración mediana de cada tono. Verificada en Edge capturando el audio y midiendo el F0 con Praat: RMSE frente al contorno esperado 0,03 / 0,12 / 0,17 / 0,65 st (T1-T4). Timbre cambiado de diente de sierra a armónicos 1/n² (sonaba a trompeta).
- 2026-11-18: licencia de Tone Perfect confirmada: CC BY-NC 4.0 (`sources/2026-11-18_brief_licencia-tone-perfect.md`). Voz nativa en Practicar: `07` → `frontend/src/data/reference_voices.json` (referencia de cada hablante con ma1-4, igual que el perfil del usuario: FV1 233, FV2 232, FV3 216, MV1 141, MV2 125, MV3 114 Hz), `src/tonos/reference_audio.py`, `GET /api/reference-audio/{sílaba}/{tono}/{hablante}` (opcional, `TONOS_TONE_PERFECT_DIR`), componente `NativeVoice` (voz más parecida primero, «Otra voz», atribución). API probada con curl (200 mp3, 404 jiao1_FV3, 422 entradas inválidas). Sin probar en el navegador. README sin actualizar para `07` ni la licencia.
- 2026-11-18: curva de aprendizaje por número de hablantes (`37`), PARADA a petición de la usuaria tras el fold 0 (resultados parciales en `outputs/escalera_dominio/curva_hablantes/curva.csv`): 4 / 8 / 16 hablantes → CNN 0,736 / 0,750 / 0,765 · frase 0,775 / 0,801 / 0,822. El script no se reanuda: relanzarlo repite el fold 0.

## Para retomar
```powershell
copy .env.example .env   # si no existe
cd frontend; npm ci; npm run build; cd ..
uv run --env-file .env uvicorn tonos.api:app --port 8000   # http://localhost:8000
```
Si cambia el F0 o un modelo: relanzar `02`, `03`, `21`, `22`, `23`, `25` (orden en el notebook) y regenerar el notebook revisando las cifras que cita su texto.

## Punto exacto al cerrar el 2026-11-18
Línea elegida: escalera de dominio como hilo del TFM. Tesis: la accuracy con nativos no mide la validez de un juez de pronunciación (contrafactual de cambio de tono). Hecho hoy: peldaño 1, contexto, HuBERT, contrafactuales, demo y Resultados en el frontend, guía tarareada, voz nativa, notebook 02 y auditoría.

**Al volver, en este orden (decidido por la usuaria):**
1. Arreglar la auditoría:
   - Alta: (1-2) zips movidos a `data/raw/` el 2026-11-19; falta que la usuaria cree el repo con los comandos dados. (4) `15` y `29` relanzados el 2026-11-19 en el scratchpad para comparar cifras (en curso). (5) Hecho: README aclara que la calibración es solo en Tone Perfect.
   - Media: (6) licencia resuelta (CC BY-NC 4.0): añadir atribución y cita en el README. (7) El generador del notebook 02 se copió tal cual a `scripts/38_generar_notebook_02.py` (uso: `uv run python scripts/38_generar_notebook_02.py notebooks/02_escalera_dominio.ipynb` y después nbconvert --execute): falta adaptarlo a las convenciones de los scripts (argparse, docstring de uso) y documentarlo en el README. (8) Una sola semilla. (9) Tests de `f0_on_frame_grid`, `fix_octave_jumps`, `utterance_bounds`, `impose_contour_on_span` y `lexical_tone_baseline`. (10) Despliegue con torch CPU. (11) Límite de uso en `/api/resynthesize` y `/api/lab/judge`.
   - Baja: (12) README: sección «Código» (2 de 15 módulos) y scripts `07` y `37`. (13) Decidir si se retira el pipeline antiguo de AISHELL-3 (`13`, `14`, `20`, `clean_f0`). (14) `tonos.main()` «Hello from tonos!», descripción de `pyproject.toml`, `Counter` sin usar en `11`. (15) `35` con `--layer 10` fijo: leerlo de las métricas de `31`. (16) `RecordButton` descarta el error. (17) Voz corregida ofrecida cuando el tono ya es correcto (suena igual): decidir diseño. (19) Dependencias con `>=` (las fija el lock).
2. Continuar el entrenamiento: relanzar `37` (curva por número de hablantes, ~1-1,5 h) y llevar el resultado al notebook 02. Con él, decidir si se prueba entrenamiento multidominio (Tone Perfect + AISHELL-3).
3. Probar en el navegador la voz nativa de Practicar.
4. Camino crítico del TFM: reclutar y grabar hispanohablantes (peldaño 2), con 1-2 nativos que marquen los tonos mal dichos, para validar la tesis con errores reales.

## Siguientes pasos (después de lo anterior)
1. Probar la app con micrófono real en Chrome, Firefox y Safari (Safari graba en mp4).
2. Desplegar en Hugging Face Spaces: torch CPU y solo dependencias de inferencia; el laboratorio añade HuBERT (~380 MB).
3. Perfil de voz: entrenar la CNN con desplazamientos verticales aleatorios para hacerla robusta al error de referencia (hasta 2 st). Sin decidir.
4. Modo juego (pilar D) y motor adaptativo (pilar E), si da tiempo.
5. Al terminar el proyecto: preparar la presentación para una sesión de innovación interna, con un asistente que entrena modelos de IA. Enfoque acordado: abrir con la demo «¿Quién te corrige?» y la tabla que se invierte, y contar después la escalera, la línea base léxica y los contrafactuales. Llevar respondido: falta de alumnos reales, fine-tuning de HuBERT frente a aumentación contrafactual, una sola semilla, errores sintéticos frente a reales, y que Tone Perfect es CC BY-NC (no comercial).

## Avisos
- `npm audit`: ReDoS en picomatch (herramienta de build de Vite), sin arreglo salvo bajar a Vite 6. No llega al navegador.
- La carpeta no es un repositorio git en este momento.
