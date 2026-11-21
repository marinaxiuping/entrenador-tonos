"""Laboratorio "¿quién te corrige?": se cambia el tono de una sílaba de una frase real y se
compara qué dice cada juez, sin dependencias del framework web.

Es el contrafactual del script 34 aplicado a una sola sílaba elegida por el usuario. Los
modelos del paquete (script 35) se entrenaron sin los hablantes de las frases de la demo.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tonos.align import SR, load_16k
from tonos.cnn import ContourCNN, contour_inputs, predict_logits
from tonos.context import UtteranceTagger, predict_tagger_logits, syllable_inputs
from tonos.embeddings import SpeechEncoder
from tonos.error_detector import ToneErrorDetector
from tonos.inference import SpeakerProfile
from tonos.linear_probe import LinearProbe
from tonos.resynthesis import impose_contour_on_span, resynthesize_with_f0
from tonos.utterance_contours import syllable_contours

BUNDLE_FILE = "laboratorio.pt"
DETECTOR_FILE = "detector_ompal.pt"
JUDGES = ["cnn_silaba", "frase", "hubert", "detector", "lexico"]


class LabRequestError(ValueError):
    """Frase, sílaba o tono no válidos para la demo."""


@dataclass(frozen=True)
class Verdict:
    """Un juez de tono da ``tone`` y ``probabilities``; el detector de errores no clasifica el
    tono, sino que dice si el esperado está mal dicho (``alert``) con ``error_probability``."""

    judge: str
    tone: int | None
    probabilities: dict[int, float] | None
    error_probability: float | None = None
    alert: bool | None = None


@dataclass(frozen=True)
class JudgeResult:
    original_tone: int
    imposed_tone: int | None
    audio_wav: bytes
    original_contour_st: list[float]
    judged_contour_st: list[float] | None
    verdicts: list[Verdict]


def wav_bytes(waveform: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _rounded(contour: np.ndarray) -> list[float] | None:
    return None if np.isnan(contour).any() else contour.round(3).tolist()


class ToneJudgeLab:
    def __init__(self, bundle_dir: str | Path, device: str = "cpu"):
        bundle_dir = Path(bundle_dir)
        artifact = torch.load(bundle_dir / BUNDLE_FILE, map_location=device, weights_only=True)
        self.device = device
        self.tones: list[int] = artifact["tones"]
        self.n_points: int = artifact["n_points"]
        self.utterances: dict[str, dict] = {u["id"]: u for u in artifact["utterances"]}
        self.profiles = {s: SpeakerProfile(**p) for s, p in artifact["profiles"].items()}
        self.donor_pools = {int(t): np.array(c) for t, c in artifact["donor_pools"].items()}
        self.character_tones: dict[str, int] = artifact["lexicon"]["characters"]
        self.syllable_tones: dict[str, int] = artifact["lexicon"]["syllables"]
        self.audio_dir = bundle_dir / "audio"

        self.cnn = ContourCNN(self.n_points, len(self.tones)).to(device)
        self.cnn.load_state_dict(artifact["cnn"])
        self.cnn.eval()
        self.tagger = UtteranceTagger(self.n_points, len(self.tones), in_channels=3).to(device)
        self.tagger.load_state_dict(artifact["tagger"])
        self.tagger.eval()
        hubert = artifact["hubert"]
        self.hubert_layer: int = hubert["layer"]
        self.encoder = SpeechEncoder(hubert["model"], hubert["revision"], device)
        self.probe = LinearProbe.from_state_dict(hubert["probe"], device)
        detector = torch.load(bundle_dir / DETECTOR_FILE, map_location="cpu", weights_only=True)
        if detector["hubert_layer"] != self.hubert_layer:
            raise ValueError("el detector de errores usa otra capa de HuBERT: repetir los scripts 45 y 47")
        self.error_detector = ToneErrorDetector.from_state_dict(detector["detector"])
        self.error_threshold: float = detector["threshold"]
        self._waveforms: dict[str, np.ndarray] = {}

    def utterance_list(self) -> list[dict]:
        return list(self.utterances.values())

    def judge(self, utterance_id: str, syllable_index: int, imposed_tone: int | None) -> JudgeResult:
        """Juicio de los cinco jueces sobre una sílaba, con su tono original o con ``imposed_tone``."""
        utterance = self.utterances.get(utterance_id)
        if utterance is None:
            raise LabRequestError(f"frase desconocida: {utterance_id}")
        if not 0 <= syllable_index < len(utterance["syllables"]):
            raise LabRequestError("índice de sílaba fuera de la frase")
        syllable = utterance["syllables"][syllable_index]
        if not syllable["evaluable"]:
            raise LabRequestError("esa sílaba no tiene un tono pleno que se pueda cambiar")
        if imposed_tone is not None and (imposed_tone not in self.tones or imposed_tone == syllable["tone"]):
            raise LabRequestError(f"el tono impuesto debe ser uno de {self.tones} distinto del original")

        profile = self.profiles[utterance["speaker"]]
        spans = [(s["start"], s["end"]) for s in utterance["syllables"]]
        original = self._waveform(utterance_id)
        waveform = original if imposed_tone is None else resynthesize_with_f0(
            original, SR, profile.fmin, profile.fmax,
            lambda f0: impose_contour_on_span(f0, syllable["start"], syllable["end"],
                                              self._donor_contour(utterance_id, syllable_index, imposed_tone),
                                              profile.reference_hz))
        original_contours = syllable_contours([original], [spans], profile, self.n_points)
        judged_contours = original_contours if imposed_tone is None else syllable_contours(
            [waveform], [spans], profile, self.n_points)
        embedding = self._syllable_embedding(waveform, syllable)
        return JudgeResult(
            original_tone=syllable["tone"],
            imposed_tone=imposed_tone,
            audio_wav=wav_bytes(waveform),
            original_contour_st=_rounded(original_contours[syllable_index]),
            judged_contour_st=_rounded(judged_contours[syllable_index]),
            verdicts=[self._cnn_verdict(judged_contours[syllable_index]),
                      self._tagger_verdict(judged_contours, syllable_index),
                      self._hubert_verdict(embedding),
                      self._detector_verdict(embedding, syllable["tone"]),
                      self._lexicon_verdict(syllable)],
        )

    def utterance_logits(self, waveform: np.ndarray, spans: list[tuple[float, float]],
                         contours: np.ndarray) -> dict[str, np.ndarray]:
        """Logits (n_sílabas, n_tonos) de los tres jueces que oyen el audio, para todas las sílabas
        de una frase de fuera del paquete. La CNN por sílaba deja NaN en las sílabas sin F0."""
        cnn = np.full((len(contours), len(self.tones)), np.nan)
        voiced = ~np.isnan(contours).any(axis=1)
        if voiced.any():
            cnn[voiced] = predict_logits(self.cnn, contour_inputs(contours[voiced]), self.device)
        tagger = predict_tagger_logits(self.tagger, syllable_inputs(contours), np.array([[0, len(contours)]]),
                                       self.device)
        embeddings = self.encoder.segment_means(waveform, spans)[:, self.hubert_layer].astype(np.float32)
        hubert = self.probe.logits(torch.from_numpy(embeddings).to(self.device)).cpu().numpy()
        return {"cnn_silaba": cnn, "frase": tagger, "hubert": hubert}

    def _donor_contour(self, utterance_id: str, syllable_index: int, tone: int) -> np.ndarray:
        """Contorno real de ``tone`` elegido de forma fija por sílaba: repetir el juicio suena igual."""
        pool = self.donor_pools[tone]
        return pool[zlib.crc32(f"{utterance_id}:{syllable_index}:{tone}".encode()) % len(pool)]

    def _waveform(self, utterance_id: str) -> np.ndarray:
        if utterance_id not in self._waveforms:
            self._waveforms[utterance_id] = load_16k(str(self.audio_dir / f"{utterance_id}.wav"))
        return self._waveforms[utterance_id]

    def _verdict(self, judge: str, logits: np.ndarray) -> Verdict:
        probabilities = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=0).numpy()
        return Verdict(judge, self.tones[int(probabilities.argmax())],
                       {t: round(float(p), 4) for t, p in zip(self.tones, probabilities)})

    def _cnn_verdict(self, contour: np.ndarray) -> Verdict:
        if np.isnan(contour).any():
            return Verdict("cnn_silaba", None, None)
        return self._verdict("cnn_silaba", predict_logits(self.cnn, contour_inputs(contour[None]), self.device)[0])

    def _tagger_verdict(self, contours: np.ndarray, index: int) -> Verdict:
        bounds = np.array([[0, len(contours)]])
        logits = predict_tagger_logits(self.tagger, syllable_inputs(contours), bounds, self.device)
        return self._verdict("frase", logits[index])

    def _syllable_embedding(self, waveform: np.ndarray, syllable: dict) -> np.ndarray:
        return self.encoder.segment_means(waveform, [(syllable["start"], syllable["end"])])[0, self.hubert_layer]

    def _hubert_verdict(self, embedding: np.ndarray) -> Verdict:
        logits = self.probe.logits(torch.from_numpy(embedding[None]).to(self.device))[0].cpu().numpy()
        return self._verdict("hubert", logits)

    def _detector_verdict(self, embedding: np.ndarray, expected_tone: int) -> Verdict:
        """Si la sílaba lleva bien el tono del texto, no el que se le haya impuesto."""
        error_probability = float(self.error_detector.score(embedding[None], np.array([expected_tone]))[0])
        return Verdict("detector", None, None, round(error_probability, 4), error_probability > self.error_threshold)

    def _lexicon_verdict(self, syllable: dict) -> Verdict:
        tone = self.character_tones.get(syllable["hanzi"], self.syllable_tones.get(syllable["base"]))
        if tone is None:
            return Verdict("lexico", None, None)
        return Verdict("lexico", tone, {t: float(t == tone) for t in self.tones})
