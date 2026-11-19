"""Grabaciones nativas de Tone Perfect para escuchar en la práctica.

Tone Perfect se distribuye con licencia CC BY-NC 4.0 (sources/2026-11-18_brief_licencia-tone-perfect.md):
se puede servir con atribución y sin uso comercial. El audio no se versiona; se lee de la
carpeta del corpus descargado.
"""

from __future__ import annotations

from pathlib import Path

from tonos.data import build_index


class ReferenceAudioLibrary:
    def __init__(self, root: str | Path):
        index, _ = build_index(root)
        self._paths = {(r.syllable, r.tone, r.speaker): Path(r.path) for r in index.itertuples()}

    def path_for(self, syllable: str, tone: int, speaker: str) -> Path | None:
        """Ruta de la grabación o None si no existe. El corpus escribe la ü como «v» (lv, nve)."""
        return self._paths.get((syllable.lower().replace("ü", "v"), tone, speaker.upper()))
