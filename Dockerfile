# Imagen de la app para Hugging Face Spaces (Docker, CPU): API + frontend en un solo proceso.
# Necesita en el contexto de build models/laboratorio/ (script 35) y las grabaciones de
# Tone Perfect en data/raw/toneperfect/tone_perfect/ (CC BY-NC 4.0, ver README).

FROM node:24.18.0-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.11.9-slim-bookworm AS python-deps
# pyworld no publica wheels para Linux y se compila; g++ se queda en esta etapa.
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
WORKDIR /build

# Las versiones salen de uv.lock con dos cambios:
#  - torch y torchaudio, su build de CPU (el lock fija la de CUDA, que pesa varios GB y no
#    sirve sin GPU).
#  - setuptools < 81: el código fuente de pyworld 0.3.5 importa pkg_resources al cargarse
#    (los wheels de Windows no), y setuptools lo quitó en la 81.
COPY pyproject.toml uv.lock README.md ./
RUN uv venv /opt/venv \
 && uv export --frozen --no-dev --no-hashes --no-emit-project --no-header --no-annotate \
      | grep -vE '^(torch|torchaudio|setuptools)==|^(triton|nvidia-)' > requirements-cpu.txt \
 && uv pip install --python /opt/venv --no-deps -r requirements-cpu.txt setuptools==80.9.0 \
 && uv pip install --python /opt/venv --no-deps \
      --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
COPY src/ src/
RUN uv pip install --python /opt/venv --no-deps .


FROM python:3.11.9-slim-bookworm
COPY --from=python-deps /opt/venv /opt/venv

# Spaces ejecuta el contenedor con el uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user models/ models/
COPY --chown=user data/raw/toneperfect/tone_perfect/ data/raw/toneperfect/
COPY --chown=user --from=frontend /frontend/dist frontend/dist

# HuBERT se descarga al construir para que el arranque no dependa de la red.
RUN python -c "import torch; from tonos.embeddings import SpeechEncoder; \
hubert = torch.load('models/laboratorio/laboratorio.pt', map_location='cpu', weights_only=True)['hubert']; \
SpeechEncoder(hubert['model'], hubert['revision'], 'cpu')"
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "tonos.api:app", "--host", "0.0.0.0", "--port", "7860"]
