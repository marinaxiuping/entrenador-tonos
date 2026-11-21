"""Capa HTTP de la API con dobles del clasificador y del laboratorio: sin cargar modelos ni HuBERT."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tonos.api import app, get_classifier
from tonos.inference import NoVoiceDetected, ToneAnalysis
from tonos.lab import JudgeResult, LabRequestError, Verdict
from tonos.rate_limit import SlidingWindowRateLimiter

TONES = [1, 2, 3, 4]


class FakeClassifier:
    tones = TONES

    def __init__(self, has_voice: bool = True):
        self.has_voice = has_voice

    def analyze(self, waveform, profile, target_tone):
        if not self.has_voice:
            raise NoVoiceDetected("no se detecta voz")
        return ToneAnalysis(tone=2, probabilities={t: 0.25 for t in TONES}, contour_st=[0.0, 1.0],
                            target_contour_st=None if target_tone is None else [0.0, 2.0])


class FakeLab:
    def judge(self, utterance_id, syllable_index, imposed_tone):
        if utterance_id != "frase":
            raise LabRequestError("frase desconocida")
        return JudgeResult(
            original_tone=3, imposed_tone=imposed_tone, audio_wav=b"RIFF",
            original_contour_st=[0.0], judged_contour_st=[1.0],
            verdicts=[Verdict("cnn_silaba", 1, {t: 0.25 for t in TONES}),
                      Verdict("detector", None, None, error_probability=0.8, alert=True)],
        )


@pytest.fixture
def client():
    # Sin «with»: no se ejecuta el lifespan, que cargaría los modelos reales.
    app.state.lab = FakeLab()
    app.state.heavy_request_limiter = SlidingWindowRateLimiter(max_requests=100, window_s=60)
    app.dependency_overrides[get_classifier] = FakeClassifier
    yield TestClient(app)
    app.dependency_overrides.clear()


def wav_upload() -> dict:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16_000, format="WAV")
    return {"file": ("silaba.wav", buffer.getvalue(), "audio/wav")}


def profile_form(**changes) -> dict:
    return {"reference_hz": "200", "fmin": "75", "fmax": "400", **changes}


class TestLabJudge:
    def test_returns_tone_verdicts_and_detector_alert(self, client):
        response = client.post("/api/lab/judge", json={"utterance_id": "frase", "syllable_index": 0, "imposed_tone": 1})

        assert response.status_code == 200
        verdicts = {v["judge"]: v for v in response.json()["verdicts"]}
        assert verdicts["cnn_silaba"]["tone"] == 1 and verdicts["cnn_silaba"]["alert"] is None
        assert verdicts["detector"] == {"judge": "detector", "tone": None, "probabilities": None,
                                        "error_probability": 0.8, "alert": True}

    def test_invalid_request_is_422(self, client):
        response = client.post("/api/lab/judge", json={"utterance_id": "otra", "syllable_index": 0})

        assert response.status_code == 422

    def test_missing_lab_is_503(self, client):
        app.state.lab = None

        assert client.post("/api/lab/judge", json={"utterance_id": "frase", "syllable_index": 0}).status_code == 503

    def test_too_many_requests_is_429(self, client):
        app.state.heavy_request_limiter = SlidingWindowRateLimiter(max_requests=1, window_s=60)
        body = {"utterance_id": "frase", "syllable_index": 0}

        assert client.post("/api/lab/judge", json=body).status_code == 200
        assert client.post("/api/lab/judge", json=body).status_code == 429


class TestAnalyze:
    def test_reports_whether_the_target_tone_was_produced(self, client):
        response = client.post("/api/analyze", files=wav_upload(), data=profile_form(target_tone="2"))

        assert response.status_code == 200
        assert response.json()["correct"] is True

    def test_unknown_target_tone_is_422(self, client):
        assert client.post("/api/analyze", files=wav_upload(), data=profile_form(target_tone="5")).status_code == 422

    def test_inverted_pitch_range_is_422(self, client):
        assert client.post("/api/analyze", files=wav_upload(), data=profile_form(fmin="400", fmax="75")).status_code == 422

    def test_silent_recording_is_422(self, client):
        app.dependency_overrides[get_classifier] = lambda: FakeClassifier(has_voice=False)

        assert client.post("/api/analyze", files=wav_upload(), data=profile_form()).status_code == 422
