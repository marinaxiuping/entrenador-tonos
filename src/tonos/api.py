"""API HTTP del entrenador de tonos: capa fina sobre tonos.inference.

Las rutas van bajo /api. Si existe el build del frontend, se sirve en / desde el mismo
proceso, de modo que frontend y API comparten origen y no hace falta CORS.

Configuración por variables de entorno (ver .env.example):
  TONOS_MODEL_PATH     artefacto de 25_entrenar_cnn_final.py
  TONOS_MAX_UPLOAD_MB  tamaño máximo de cada audio subido
  TONOS_FRONTEND_DIR   build del frontend (npm run build en frontend/)
  TONOS_TONE_PERFECT_DIR  corpus Tone Perfect descargado; si no existe, no se sirve la voz
                       nativa de la práctica (/api/reference-audio)
  TONOS_LAB_DIR        paquete de la demo "¿quién te corrige?" (35_preparar_laboratorio.py);
                       si no existe, la API funciona sin las rutas /api/lab

Arranque: uv run --env-file .env uvicorn tonos.api:app
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tonos.f0 import SR
from tonos.inference import NoVoiceDetected, SpeakerProfile, ToneClassifier
from tonos.lab import LabRequestError, ToneJudgeLab
from tonos.reference_audio import ReferenceAudioLibrary
from tonos.resynthesis import resynthesize_with_contour

MODEL_PATH = os.environ.get("TONOS_MODEL_PATH", "models/cnn_toneperfect_praat.pt")
FRONTEND_DIR = os.environ.get("TONOS_FRONTEND_DIR", "frontend/dist")
TONE_PERFECT_DIR = os.environ.get("TONOS_TONE_PERFECT_DIR", "data/raw/toneperfect")
LAB_DIR = os.environ.get("TONOS_LAB_DIR", "models/laboratorio")
MAX_UPLOAD_BYTES = int(float(os.environ.get("TONOS_MAX_UPLOAD_MB", "5")) * 1024 * 1024)
MAX_CALIBRATION_FILES = 12
MAX_RESYNTHESIS_RATE = 48_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se carga al arrancar para fallar de inmediato si falta el modelo, no en la primera petición.
    app.state.classifier = ToneClassifier(MODEL_PATH)
    # El laboratorio descarga HuBERT (~380 MB) la primera vez; es opcional para poder
    # desplegar solo la práctica.
    app.state.lab = ToneJudgeLab(LAB_DIR) if os.path.isdir(LAB_DIR) else None
    app.state.reference_audio = ReferenceAudioLibrary(TONE_PERFECT_DIR) if os.path.isdir(TONE_PERFECT_DIR) else None
    yield


app = FastAPI(title="Entrenador de tonos", lifespan=lifespan)
router = APIRouter(prefix="/api")


def get_classifier(request: Request) -> ToneClassifier:
    return request.app.state.classifier


def get_lab(request: Request) -> ToneJudgeLab:
    if request.app.state.lab is None:
        raise HTTPException(503, "el laboratorio no está instalado en este servidor")
    return request.app.state.lab


def get_reference_audio(request: Request) -> ReferenceAudioLibrary:
    if request.app.state.reference_audio is None:
        raise HTTPException(503, "las grabaciones de Tone Perfect no están instaladas en este servidor")
    return request.app.state.reference_audio


class ProfileResponse(BaseModel):
    reference_hz: float
    fmin: float
    fmax: float


class AnalysisResponse(BaseModel):
    tone: int
    probabilities: dict[int, float]
    contour_st: list[float]
    target_contour_st: list[float] | None
    correct: bool | None


class LabSyllable(BaseModel):
    hanzi: str
    pinyin: str
    tone: int
    evaluable: bool


class LabUtterance(BaseModel):
    id: str
    speaker: str
    gender: str
    text: str
    syllables: list[LabSyllable]


class JudgeRequest(BaseModel):
    utterance_id: str
    syllable_index: int
    imposed_tone: int | None = None


class VerdictResponse(BaseModel):
    judge: str
    tone: int | None
    probabilities: dict[int, float] | None


class JudgeResponse(BaseModel):
    original_tone: int
    imposed_tone: int | None
    audio_wav_base64: str
    original_contour_st: list[float] | None
    judged_contour_st: list[float] | None
    verdicts: list[VerdictResponse]


def decode_upload(upload: UploadFile) -> tuple[np.ndarray, int]:
    """Decodifica wav, flac, ogg o mp3 a mono, con su frecuencia de muestreo original."""
    content = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename}: supera {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        samples, sample_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        raise HTTPException(400, f"{upload.filename}: formato no soportado (wav, flac, ogg o mp3)")
    return samples.mean(axis=1), sample_rate


def read_waveform(upload: UploadFile) -> np.ndarray:
    """Audio a 16 kHz, la frecuencia con la que se entrenó el análisis de F0."""
    samples, sample_rate = decode_upload(upload)
    return librosa.resample(samples, orig_sr=sample_rate, target_sr=SR)


def speaker_profile(reference_hz: float, fmin: float, fmax: float) -> SpeakerProfile:
    if fmin >= fmax:
        raise HTTPException(422, "fmin debe ser menor que fmax")
    return SpeakerProfile(reference_hz, fmin, fmax)


def check_tone(tone: int | None, classifier: ToneClassifier) -> None:
    if tone is not None and tone not in classifier.tones:
        raise HTTPException(422, f"target_tone debe ser uno de {classifier.tones}")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/calibrate", response_model=ProfileResponse)
def calibrate(files: list[UploadFile] = File(..., description="Una sílaba en los cuatro tonos, p. ej. mā má mǎ mà"),
              classifier: ToneClassifier = Depends(get_classifier)) -> ProfileResponse:
    if len(files) > MAX_CALIBRATION_FILES:
        raise HTTPException(400, f"como máximo {MAX_CALIBRATION_FILES} grabaciones de calibración")
    try:
        profile = classifier.build_profile([read_waveform(f) for f in files])
    except NoVoiceDetected as error:
        raise HTTPException(422, str(error))
    return ProfileResponse(reference_hz=profile.reference_hz, fmin=profile.fmin, fmax=profile.fmax)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(file: UploadFile = File(...),
            reference_hz: float = Form(..., gt=0),
            fmin: float = Form(..., gt=0),
            fmax: float = Form(..., gt=0),
            target_tone: int | None = Form(None),
            classifier: ToneClassifier = Depends(get_classifier)) -> AnalysisResponse:
    profile = speaker_profile(reference_hz, fmin, fmax)
    check_tone(target_tone, classifier)
    try:
        result = classifier.analyze(read_waveform(file), profile, target_tone)
    except NoVoiceDetected as error:
        raise HTTPException(422, str(error))
    return AnalysisResponse(
        tone=result.tone,
        probabilities=result.probabilities,
        contour_st=result.contour_st,
        target_contour_st=result.target_contour_st,
        correct=None if target_tone is None else result.tone == target_tone,
    )


@router.post("/resynthesize", response_class=Response,
             responses={200: {"content": {"audio/wav": {}}, "description": "Tu grabación con el tono objetivo"}})
def resynthesize(file: UploadFile = File(...),
                 reference_hz: float = Form(..., gt=0),
                 fmin: float = Form(..., gt=0),
                 fmax: float = Form(..., gt=0),
                 target_tone: int = Form(...),
                 classifier: ToneClassifier = Depends(get_classifier)) -> Response:
    """Voz corregida: la grabación del usuario con el contorno de F0 del tono objetivo."""
    profile = speaker_profile(reference_hz, fmin, fmax)
    check_tone(target_tone, classifier)
    samples, sample_rate = decode_upload(file)
    # Se resintetiza a la frecuencia original para no perder calidad; por encima de 48 kHz
    # solo encarece WORLD sin mejora audible.
    if sample_rate > MAX_RESYNTHESIS_RATE:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=MAX_RESYNTHESIS_RATE)
        sample_rate = MAX_RESYNTHESIS_RATE
    audio = resynthesize_with_contour(samples, sample_rate, np.array(classifier.target_contours[target_tone]),
                                      profile.reference_hz, profile.fmin, profile.fmax)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buffer.getvalue(), media_type="audio/wav")


@router.get("/reference-audio/{syllable}/{tone}/{speaker}", response_class=FileResponse,
            responses={200: {"content": {"audio/mpeg": {}}, "description": "Sílaba dicha por un hablante nativo"}})
def reference_audio(syllable: str = PathParam(..., pattern=r"^[a-zü]{1,8}$"),
                    tone: int = PathParam(..., ge=1, le=4),
                    speaker: str = PathParam(..., pattern=r"^[FM]V[1-9]$"),
                    library: ReferenceAudioLibrary = Depends(get_reference_audio)) -> FileResponse:
    """Grabación de Tone Perfect (MSU Libraries, CC BY-NC 4.0)."""
    path = library.path_for(syllable, tone, speaker)
    if path is None:
        raise HTTPException(404, f"no hay grabación de {syllable}{tone} con la voz {speaker}")
    # El audio no cambia: el navegador puede guardarlo un día.
    return FileResponse(path, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/lab/utterances", response_model=list[LabUtterance])
def lab_utterances(lab: ToneJudgeLab = Depends(get_lab)) -> list[dict]:
    return lab.utterance_list()


@router.post("/lab/judge", response_model=JudgeResponse)
def lab_judge(request: JudgeRequest, lab: ToneJudgeLab = Depends(get_lab)) -> JudgeResponse:
    """Cambia el tono de una sílaba (o la deja como está) y devuelve el audio y el juicio de cada modelo."""
    try:
        result = lab.judge(request.utterance_id, request.syllable_index, request.imposed_tone)
    except LabRequestError as error:
        raise HTTPException(422, str(error))
    return JudgeResponse(
        original_tone=result.original_tone,
        imposed_tone=result.imposed_tone,
        audio_wav_base64=base64.b64encode(result.audio_wav).decode("ascii"),
        original_contour_st=result.original_contour_st,
        judged_contour_st=result.judged_contour_st,
        verdicts=[VerdictResponse(judge=v.judge, tone=v.tone, probabilities=v.probabilities) for v in result.verdicts],
    )


app.include_router(router)
# Después de las rutas de la API, para que el frontend no las tape.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
