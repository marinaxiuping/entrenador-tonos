"""Alineamiento forzado a nivel de sílaba con torchaudio MMS_FA (wav2vec2 CTC, sin conda/MFA).

Cada sílaba pinyin (sin número de tono) se trata como una "palabra" del alineador.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import librosa
import numpy as np
import torch
from torchaudio.pipelines import MMS_FA

SR = MMS_FA.sample_rate  # 16 kHz


@dataclass
class SylSpan:
    syl: str
    start: float  # s
    end: float  # s
    score: float  # media de la probabilidad CTC de sus tokens


class SyllableAligner:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MMS_FA.get_model(with_star=False).to(self.device).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()

    @staticmethod
    def normalize(syl: str) -> str:
        return re.sub(r"[^a-zv]", "", syl.lower().replace("ü", "v"))

    @torch.inference_mode()
    def align(self, wav: np.ndarray, syllables: list[str]) -> list[SylSpan]:
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        emission, _ = self.model(x)
        words = [self.normalize(s) for s in syllables]
        spans = self.aligner(emission[0], self.tokenizer(words))
        sec_per_frame = wav.shape[0] / emission.shape[1] / SR
        return [
            SylSpan(s, sp[0].start * sec_per_frame, sp[-1].end * sec_per_frame,
                    float(np.mean([t.score for t in sp])))
            for s, sp in zip(syllables, spans)
        ]


def fill_gaps(spans: list[SylSpan], max_gap: float = 0.25, tail: float = 0.08) -> list[SylSpan]:
    """Los tramos CTC son 'picudos' y dejan huecos de blanco que se comen la cola de la rima
    (justo donde está el movimiento tonal). Cada sílaba se extiende hasta el inicio de la
    siguiente si el hueco es < max_gap; si es una pausa (o la última), se añade solo ``tail``."""
    out = []
    for i, s in enumerate(spans):
        nxt = spans[i + 1].start if i + 1 < len(spans) else None
        gap = (nxt - s.end) if nxt is not None else np.inf
        end = nxt if gap < max_gap else s.end + tail
        out.append(SylSpan(s.syl, s.start, max(end, s.end), s.score))
    return out


def load_16k(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y
