# TFM — Entrenador de tonos del mandarín

Máster en Data Science e IA (enfoque ML). Entrega: menos de un mes desde el 15 nov 2026 (fecha exacta: _pendiente_).

## Idea
App interactiva para aprender tonos del mandarín: grabas, el modelo clasifica el tono, ves tu curva de pitch frente a la objetivo, escuchas **tu propia voz corregida** y practicas con un **modo juego** controlado por la voz.

## Alcance (versión 4 semanas)

### Pilar 1 — Clasificación de tonos (núcleo de ML, innegociable)
- Pipeline: recorte de silencios → F0 (pYIN vs CREPE) → normalización por hablante (semitonos respecto a la media del hablante).
- Modelos a comparar:
  1. Baseline: features de F0 + gradient boosting.
  2. CNN sobre contorno de F0.
  3. Embeddings preentrenados **congelados** (wav2vec2/HuBERT, idealmente en chino) + clasificador lineal.
- Calibración de probabilidades (temperature scaling).
- **Cambio de dominio:** entrenar con nativos, evaluar con hispanohablantes; medir la caída de rendimiento.
- Splits por hablante (leave-speaker-out).

### A — Corrección con tu propia voz (versión simplificada)
- Modelo ligero que predice parámetros del contorno objetivo (p. ej. coeficientes polinómicos) a partir del rango y la duración del hablante.
- Resíntesis con WORLD (pyworld) sustituyendo solo el F0.
- Comparación con plantilla fija si da tiempo.

### D — Modo juego (escaparate)
- Pitch en tiempo real en el navegador (YIN en AudioWorklet), calibración de rango al inicio.
- Un personaje sigue la curva del tono objetivo.

### E — Motor adaptativo (solo si sobra tiempo, ~3 días)
- BKT por contraste tonal + validación con aprendices simulados frente a selección aleatoria.

### Trabajo futuro (fuera de alcance)
Fine-tuning de wav2vec2, generador de contornos seq2seq/VAE, DKT, bandit, ONNX en el navegador, frases completas.

## Datos
- **Monosílabos:** Tone Perfect (Michigan State University).
- **Bisílabos — DECIDIDO (15 nov): SÍ, con AISHELL-3.** Etiquetas de tono realizado (sandhi incluido), 211k bisílabos de 172 hablantes, alineamiento silábico con torchaudio MMS_FA. Detalles en [corpus-bisilabos.md](corpus-bisilabos.md).
- **Tone Perfect:** acceso solicitado por formulario (15 nov). Si no hay respuesta antes del 22 nov → monosílabos sacados de las sílabas alineadas de AISHELL-3.
- **Test propio:** 6–8 hispanohablantes que hayan estudiado chino. Reclutar ya y grabar en la semana 1. Recoger el consentimiento.

## Calendario
| Semana | Fechas | Objetivos |
|---|---|---|
| 1 | 16–22 nov | Decisión de datos (bisílabos sí/no), pipeline F0, baseline, grabaciones de hispanohablantes |
| 2 | 23–29 nov | CNN + embeddings, cambio de dominio, calibración. **MVP de principio a fin desplegado.** Empezar la memoria |
| 3 | 30 nov–6 dic | A (voz corregida) + D (modo juego) |
| 4 | 7–13 dic | E si llegas, evaluación final, cierre de la memoria |

## Stack
- Backend: Python, FastAPI, librosa / parselmouth, pyworld, scikit-learn / LightGBM, PyTorch, HuggingFace.
- Frontend: React + Web Audio API.
- Experimentos: MLflow o W&B.

## Riesgos
- Tono 3 con creaky voice: el pitch tracking falla. Probar pYIN vs CREPE con grabaciones propias en la semana 1.
- WORLD genera artefactos con desplazamientos de F0 grandes; documentarlo, no esconderlo.
- Etiquetas canónicas ≠ tonos realizados (bisílabos).
- Plazo: si algo se atasca más de 2 días, recortar y pasar a lo siguiente.

## Primeras tareas (día 1)
1. Descargar Tone Perfect y explorar: hablantes, número de muestras, balance de tonos.
2. Script de extracción de F0 + visualización de contornos por tono.
3. Buscar corpus de bisílabos (AISHELL-3 y alternativas).
4. Mensaje para reclutar a los hispanohablantes.
