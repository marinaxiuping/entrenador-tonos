---
title: Entrenador de tonos del mandarín
emoji: 🎶
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
python_version: "3.10"
pinned: false
license: cc-by-nc-4.0
---

# Entrenador de tonos del mandarín

Graba una sílaba, mira tu curva de tono frente a la objetivo, escucha tu propia voz corregida y
compara cómo juzgan los errores de tono distintos modelos («¿Quién corrige?»).

## Datos y licencias

- **Tone Perfect** (Catherine Ryu, Mandarin Tone Perception & Production Team y Michigan State
  University Libraries), CC BY-NC 4.0: voz nativa de la práctica y datos de entrenamiento del
  modelo de Practicar. Por esta licencia, el Space no tiene uso comercial.
- **AISHELL-3** (Shi et al., Interspeech 2021), Apache 2.0: frases de la demo «¿Quién corrige?».
- **OMPAL** (Hsieh et al., Interspeech 2025), CC BY 4.0: errores reales de alumnos con los que se
  entrenó el detector «HuBERT + alumnos».
- **chinese-hubert-base** (TencentGameMate): modelo de habla preentrenado, se descarga al arrancar.
