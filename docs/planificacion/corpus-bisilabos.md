# Punto de control: corpus de bisílabos (días 1–3)

Fecha límite de la decisión: **18 nov 2026**.

## Candidato principal: AISHELL-3
- 88.035 grabaciones, 218 hablantes nativos, lectura de frases con emoción neutra. Descarga: OpenSLR SLR93 / www.aishelltech.com/aishell_3 (**licencia por confirmar en la página de descarga**).
- **Etiquetas: tonos REALIZADOS, no canónicos.** El pinyin se transcribió a mano escuchando el audio, así que ya refleja el sandhi (p. ej. 3+3 → 2+3). Esto resuelve el riesgo de "etiquetas canónicas ≠ tonos realizados".
- Inconvenientes:
  - Es habla continua, no palabras aisladas. Para recortar los bisílabos hace falta alineamiento forzado (Montreal Forced Aligner tiene modelo de mandarín) y hay coarticulación con las palabras vecinas.
  - Las frases son de TTS y el dominio es distinto al de la app (palabras aisladas).
  - Volumen grande: basta un subconjunto de hablantes.
- Hay literatura de clasificación de tonos sobre AISHELL-3 (79,5–92,6 % de exactitud, arXiv 2104.05657) que sirve como referencia.

## Alternativas vistas (sin corpus público directo)
- Estudios L2 con bisílabos aislados evaluados por nativos (aprendices japoneses, 80 palabras, 3 jueces): ninguno publica el audio.
- Corpus de evaluación de tonos L2 (Sci. Rep. 2025, 79.140 sílabas): mezcla grabaciones propias y corpus abiertos; no está claro que sea público.

## Comprobaciones (15 nov)
1. [x] **Acceso y licencia.** Réplica HF `shenyunhang/AISHELL-3` con la estructura original y descarga por hablante (`scripts/10_descargar_aishell3.py`). Licencia Apache 2.0 (OpenSLR SLR93).
2. [x] **Etiquetas = tonos realizados** (`scripts/11_analizar_etiquetas_aishell3.py`):
   - `train/label_train-set.txt` marca los límites de palabra prosódica (`%`, `$`), así que los bisílabos salen directamente sin segmentar el texto.
   - El 64,5 % de las palabras prosódicas son bisílabas: **211.190 bisílabos** de 172 hablantes (mediana de 1.334 por hablante). Hay muchos más hablantes mujeres (133) que hombres (35).
   - De los bisílabos canónicos 3+3, el **99,2 %** está etiquetado como 2+3.
   - Etiqueta frente a pypinyin: 88,7 % coinciden; el sandhi 3+3 y el de 一/不 están bien recogidos.
   - De los "otros" (4,8 %), el 49 % es sandhi entre palabras, que es correcto. El resto son casi todo heterónimos que pypinyin lee mal (因为, 提供, 应用…). El ruido real de etiqueta se estima en ~1–2 %.
3. [x] **Alineamiento con torchaudio MMS_FA** en lugar de MFA (que necesita conda): `src/tonos/align.py`, `scripts/12_alinear_aishell3.py`.
   - Prueba con SSB0544 (hombre) y SSB0502 (mujer): 899 frases, 3.828 bisílabos, 1–2 min en la GPU.
   - Los tramos CTC crudos son demasiado estrechos (mediana de 140 ms) y pierden la cola de la rima. Se arregla con `fill_gaps`: cada sílaba se extiende hasta la siguiente salvo que haya pausa → mediana de 220 ms.
   - Movimiento medio de F0 con `fill_gaps` (st): T1 −0,6 · T2 −0,5/+0,5 (bajada y subida) · T3 −1,8 · T4 −2,3. Son las formas esperables en habla continua.
4. [x] **Revisión manual** (15 nov): 30 bisílabos al azar de `outputs/aishell3/alineamiento/revision/`, escuchados y valorados como bien cortados.
5. [x] **DECISIÓN (15 nov): bisílabos SÍ**, con AISHELL-3 + MMS_FA + `fill_gaps`. Cumple los dos criterios: cortes limpios ≥ 80 % y pipeline en ≤ 1 día.

### Consecuencias para el diseño
- Etiqueta = tono realizado (incluye sandhi) → el modelo clasifica lo que se oye, no el tono del diccionario. Para la app, la etiqueta objetivo de un bisílabo es la realizada (p. ej. 你好 → 2+3).
- Habla continua ≠ palabras aisladas (Tone Perfect y las grabaciones de los hispanohablantes). Es otro cambio de dominio que hay que medir y comentar en la memoria.
- Desequilibrio de género (133 mujeres / 35 hombres) → splits por hablante estratificados por género; reportar métricas por género.
- Pares raros: los que tienen tono neutro en σ1 casi no existen, y el 3+3 realizado es residual (75 casos). Para clasificar pares: 4×4 de tonos plenos + neutro en σ2, o clasificar sílaba a sílaba con contexto.
- pYIN tiene errores de octava (outliers de ±13 st) → filtrado/suavizado o CREPE.

### Notas técnicas
- La caché de numba de librosa se corrompió cuando varios procesos de joblib la escribieron a la vez (access violation en `localmin`). Se arregla borrando `.venv/Lib/site-packages/librosa/**/*.nbi|*.nbc`. El script 02 ahora calienta la caché antes de paralelizar.
- Plan B si Tone Perfect no da acceso antes del 22 nov: monosílabos = sílabas alineadas de AISHELL-3 (habla continua, documentar el cambio de condición).

## Fuentes
- AISHELL-3 (arXiv 2010.11567): https://arxiv.org/abs/2010.11567
- Clasificación de tonos con contexto (arXiv 2104.05657): https://arxiv.org/pdf/2104.05657
- Evaluación de tonos L2 con red siamesa: https://www.nature.com/articles/s41598-025-08544-8
