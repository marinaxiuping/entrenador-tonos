# Referencias bibliográficas de la memoria

- Fuente: búsquedas web del 2026-11-21 (páginas de los editores, ACL Anthology, ISCA Archive, PMLR, JMLR, DBLP, arXiv y Hugging Face). Cada referencia se comprobó en al menos una de esas fuentes; se anotan las dudas que quedaron.
- Fecha de registro: 2026-11-21.
- Participantes: no aplica.

## Corpus y modelos usados

1. **Tone Perfect.** Ryu, C., Mandarin Tone Perception & Production Team y Michigan State University Libraries. *Tone Perfect: Multimodal Database for Mandarin Chinese*. https://tone.lib.msu.edu (formato de cita de https://tone.lib.msu.edu/cite; licencia en `2026-11-18_brief_licencia-tone-perfect.md`).
2. **AISHELL-3.** Shi, Y., Bu, H., Xu, X., Zhang, S. y Li, M. (2021). AISHELL-3: A Multi-Speaker Mandarin TTS Corpus. *Proc. Interspeech 2021*. doi:10.21437/Interspeech.2021-755. https://www.isca-archive.org/interspeech_2021/shi21c_interspeech.html. 85 h de 218 hablantes nativos; Apache 2.0. Duda: las páginas aparecen como 2756-2760 en ISCA y 3526-3530 en otra ficha; se cita por DOI.
3. **OMPAL.** Hsieh et al. (2025). OMPAL: Bridging Speech and Learning with an Open-Source Mandarin Pronunciation Assessment Corpus for Global Learners. *Proc. Interspeech 2025*, 2415-2419. doi:10.21437/Interspeech.2025-983 (ver `2026-11-19_brief_corpus-ompal.md`).
4. **HuBERT.** Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R. y Mohamed, A. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29, 3451-3460. Volumen y páginas según DBLP; el DOI 10.1109/TASLP.2021.3122291 aparece asociado al artículo en otras fichas, pero no se resolvió directamente.
5. **chinese-hubert-base.** TencentGameMate. Modelo HuBERT base preentrenado con las 10.000 h del subconjunto L de WenetSpeech. https://huggingface.co/TencentGameMate/chinese-hubert-base (repositorio chinese_speech_pretrain). Revisión fijada en el proyecto: fce0375.
6. **MMS (alineamiento forzado MMS_FA).** Pratap, V. et al. (2024). Scaling Speech Technology to 1,000+ Languages. *Journal of Machine Learning Research*, 25(97), 1-52. https://www.jmlr.org/papers/v25/23-1318.html.
7. **WORLD.** Morise, M., Yokomori, F. y Ozawa, K. (2016). WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications. *IEICE Transactions on Information and Systems*, E99-D(7), 1877-1884. doi:10.1587/transinf.2015EDP7457.
8. **Parselmouth (Praat en Python).** Jadoul, Y., Thompson, B. y de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, 71, 1-15. doi:10.1016/j.wocn.2018.07.001. La documentación de Parselmouth pide citar también Praat: Boersma, P. y Weenink, D. *Praat: doing phonetics by computer* (programa informático).

## Métodos y antecedentes

9. **Calibración por temperatura.** Guo, C., Pleiss, G., Sun, Y. y Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proc. ICML 2017*, PMLR 70, 1321-1330.
10. **Líneas base que ignoran la entrada principal.** Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R. y Van Durme, B. (2018). Hypothesis Only Baselines in Natural Language Inference. *Proc. *SEM 2018*, 180-191. doi:10.18653/v1/S18-2023. Un modelo que solo ve la hipótesis supera a la clase mayoritaria en varios conjuntos de NLI: la tarea se resuelve con atajos.
11. **Datos contrafactuales.** Kaushik, D., Hovy, E. y Lipton, Z. C. (2020). Learning the Difference That Makes a Difference with Counterfactually-Augmented Data. *ICLR 2020*. Los clasificadores entrenados con datos originales fallan en sus versiones contrafactuales y al revés.
12. **Detección de errores de tono en aprendices.** Li, W., Chen, N. F., Siniscalchi, S. M. y Lee, C.-H. (2019). Improving Mandarin Tone Mispronunciation Detection for Non-Native Learners with Soft-Target Tone Labels and BLSTM-Based Deep Tone Models. *IEEE/ACM TASLP*, 27(12), 2012-2024. doi:10.1109/TASLP.2019.2936755. Etiquetas blandas porque los tonos de aprendices caen entre dos categorías; contexto entre sílabas con BLSTM.
13. **Detección de errores con HuBERT y pitch.** Wang, X., Shi, M. y Wang, Y. (2024). Pitch-Aware RNN-T for Mandarin Chinese Mispronunciation Detection and Diagnosis. *Proc. Interspeech 2024*. arXiv:2406.04595. HuBERT ajustado más una rama de F0; entrenado con nativos (AISHELL-1) y evaluado con aprendices (LATIC).
14. **Variación tonal en contexto.** Xu, Y. (1997). Contextual tonal variations in Mandarin. *Journal of Phonetics*, 25, 61-83. Los tonos vecinos modifican la realización de cada tono, con efectos de arrastre y anticipación distintos.

15. **YIN (pitch en tiempo real del juego).** de Cheveigné, A. y Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *Journal of the Acoustical Society of America*, 111(4), 1917-1930. doi:10.1121/1.1458024. Añadida y comprobada por búsqueda web el 2026-11-22; se cita al explicar la primera versión del juego, descartada.
