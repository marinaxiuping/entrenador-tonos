# Progreso

El detalle de cada paso está en los notebooks (`01` Tone Perfect, `02` escalera de dominio, OMPAL y tesis) y en la memoria (`docs/memoria/memoria.pdf`). Las decisiones con fecha y motivo, en `decisions.md`.

## Estado (2026-11-21): listo para entregar, salvo lo de «Pendiente»
- Experimentos cerrados: peldaño 0 (Tone Perfect), peldaño 1 (AISHELL-3, contexto, léxico, contrafactuales, 3 semillas) y peldaño 2 (OMPAL: jueces, profesores, AUC pareada, umbral, detector, cota del juez de Practicar). Scripts 01-49.
- Tesis (en `decisions.md`, notebook 02 sección 9, Resultados y memoria): cómo se evalúa a un juez decide cuál parece mejor; solo los errores reales miden lo que la app necesita.
- App: Practicar, «¿Quién corrige?» (5 jueces, el detector entrenado con OMPAL incluido) y Resultados. Local, imagen Docker y Space de Hugging Face con ZeroGPU (`Eskarcho/entrenador-tonos`), funcionando.
- Memoria: `docs/memoria/` (HTML → PDF con Edge, 20 páginas, figuras de `figuras.py`, capturas de `capturas.py`, referencias verificadas en `sources/`).
- Calidad: 47 tests, CI en `.github/workflows/ci.yml` (simulado en contenedor; sin probar en GitHub).

## Sesión del 2026-11-21 (resumen)
Alineamiento de OMPAL dado por bueno a oído; umbral por partición (`44`); detector entrenado con alumnos (`45`-`47`) e integrado en la demo; errores fabricados en la demo (`48`); tesis reformulada y llevada al notebook 02 y a Resultados; auditoría (donantes de la demo, AUC pareada, fuga de C, cota del juez de Practicar `49`, tests de API y evaluación, CI, imagen Docker, precalentamiento); despliegue en Hugging Face con ZeroGPU; memoria completa con referencias verificadas; repo organizado para la entrega. Commits hechos por la usuaria: `d0e65b7`; lo posterior está sin commit (comandos en la última respuesta de la sesión).

## Cifras clave
- Tone Perfect (hablante no visto): LightGBM 0,983 · CNN 0,988 · HuBERT 0,990. CNN calibrada (T 1,28) en la app.
- AISHELL-3: modelo de la app 0,566 · CNN en dominio 0,768 · frase 0,833 · HuBERT 0,966 · léxico sin audio 0,934.
- Contrafactual de cambio de tono (`34`): detecta CNN 0,675 · frase 0,598 · HuBERT 0,310.
- OMPAL (17.001 sílabas, 46 alumnos): AUC CNN 0,831 · frase 0,830 · HuBERT 0,859 (HuBERT − CNN +0,028 [0,016; 0,040]). Al 10 % de falsa alarma: 0,52 / 0,49 / 0,53; detector entrenado con alumnos 0,68 (gana en las 5 particiones). Juez de Practicar en habla continua: AUC 0,780 (solo cota).

## Pendiente para la entrega
- Modo juego («Jugar», 2026-11-22): hecho y probado de punta a punta con micrófono simulado en Edge (T2 sobre T2: 100 %; T4 sobre T2: 19 %). Margen bajado a ±1,5 st tras ver que con ±2,5 un T2 a media subida sacaba el 100 %. Memoria actualizada (sección 9 con figura 6, tabla de decisiones, límites, trabajo futuro, resumen, referencia YIN; 20 páginas; apaisada tras la sección 9 y referencias sin salto de página) y captura en `capturas.py` con micrófono simulado. Falta: probarlo con micrófono real en Chrome, Firefox y Safari y volver a desplegar el Space.
- Subir el repo a GitHub (`git remote add origin <url>`, `git push -u origin main`) y ver que el CI pasa.

Hecho el 2026-11-22 (según la usuaria): Space público comprobado sin sesión, memoria revisada, app probada con micrófono en Chrome, Firefox y Safari.

## Fuera de alcance (declarado en la memoria)
Hispanohablantes (camino crítico original, no se pudo grabar), modo juego (D) y motor adaptativo (E), evaluación perceptiva de la voz corregida, consejo de Practicar sin probar con alumnos, ajuste fino de HuBERT y entrenar con errores fabricados midiendo en OMPAL.

## Para retomar
```powershell
copy .env.example .env   # si no existe
cd frontend; npm ci; npm run build; cd ..
uv run --env-file .env uvicorn tonos.api:app --port 8000   # http://localhost:8000
uv run pytest
```
Orden completo de los scripts en el README. Si cambian resultados: regenerar el notebook 02 (`38` + nbconvert), `36` (Resultados), `docs/memoria/figuras.py` y revisar las cifras citadas en la memoria. Para `print` con hanzi en consola: `PYTHONIOENCODING=utf-8`.

## Avisos
- Las tasas de la demo (`48`) no son una evaluación: siguen por encima de `34` (CNN 0,88 frente a 0,675) por una causa no identificada (descartados el criterio de frases y los hablantes). Citar solo `34`.
- Supuesto sin verificar en `43`: la posición del voto en el JSON de OMPAL es siempre el mismo anotador.
- El entrenamiento del detector usa parámetros de scikit-learn 1.9 (`use_legacy_attributes`); en el Space (Python 3.10, scikit-learn 1.7) solo se infiere, así que no afecta.
- `npm audit`: ReDoS en picomatch (herramienta de build de Vite), sin arreglo salvo bajar a Vite 6. No llega al navegador.
- Presentación interna prevista: abrir con la demo «¿Quién corrige?» y la tabla que se invierte; llevar respondido falta de hispanohablantes, ajuste fino frente a aumentación contrafactual, errores sintéticos frente a reales y licencia CC BY-NC de Tone Perfect.
