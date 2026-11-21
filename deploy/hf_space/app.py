"""Punto de entrada del Space de Hugging Face (SDK Gradio, hardware ZeroGPU).

Arranca la API de la app tal cual, sin interfaz de Gradio; la API sirve también el frontend en
«/». ZeroGPU exige al menos una función con @spaces.GPU y solo da GPU dentro de esas funciones:
aquí lo es el juicio de «¿Quién corrige?», que corre HuBERT, la CNN y la BiGRU en la GPU. Fuera
de ZeroGPU (por ejemplo, al probar en local), spaces.GPU no hace nada y todo va en CPU.

Las rutas por defecto de tonos.api (models/, frontend/dist/, data/raw/toneperfect/) coinciden con
la estructura que monta deploy/preparar_space.py.
"""

import os

import spaces  # Antes que torch: en ZeroGPU parchea CUDA al importarse.

ON_ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true"}
# tonos.api lee la configuración al importarse: hay que fijarla antes.
os.environ.setdefault("TONOS_LAB_DEVICE", "cuda" if ON_ZERO_GPU else "cpu")
os.environ.setdefault("TONOS_LAB_WARM_UP", "0" if ON_ZERO_GPU else "1")

import uvicorn  # noqa: E402

from tonos import api  # noqa: E402
from tonos.lab import ToneJudgeLab  # noqa: E402

judge_in_this_process = ToneJudgeLab.judge


@spaces.GPU(duration=15)
def judge_on_gpu(utterance_id: str, syllable_index: int, imposed_tone: int | None):
    # Solo argumentos simples: ZeroGPU los envía a otro proceso, que hereda el laboratorio cargado.
    return judge_in_this_process(api.app.state.lab, utterance_id, syllable_index, imposed_tone)


def judge(self: ToneJudgeLab, utterance_id: str, syllable_index: int, imposed_tone: int | None):
    # Se valida aquí: una excepción lanzada en el proceso de la GPU llega como error genérico (500).
    self.validate_request(utterance_id, syllable_index, imposed_tone)
    return judge_on_gpu(utterance_id, syllable_index, imposed_tone)


ToneJudgeLab.judge = judge

if __name__ == "__main__":
    print(f"ZeroGPU: {ON_ZERO_GPU} | dispositivo de la demo: {os.environ['TONOS_LAB_DEVICE']}", flush=True)
    uvicorn.run(api.app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
