import { encodeWav, mixToMono } from "./wav";

// El procesado de voz del navegador (supresión de ruido, control de ganancia) altera
// el pitch y la energía, así que se desactiva.
export const RAW_VOICE_CONSTRAINTS: MediaStreamConstraints = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
};

export interface RecordingSession {
  /** Para la grabación y devuelve el audio como WAV. */
  stop: () => Promise<Blob>;
}

/**
 * Graba del micrófono con el formato nativo del navegador (webm u mp4) y lo convierte a
 * WAV al parar, porque el backend no decodifica webm/opus.
 */
export async function startRecording(): Promise<RecordingSession> {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    throw new RecordingUnavailableError();
  }
  const stream = await navigator.mediaDevices.getUserMedia(RAW_VOICE_CONSTRAINTS);
  const releaseMicrophone = () => stream.getTracks().forEach((track) => track.stop());
  let recorder: MediaRecorder;
  try {
    recorder = new MediaRecorder(stream);
  } catch (error) {
    // Sin esto el micrófono quedaría abierto aunque no se grabe.
    releaseMicrophone();
    throw error;
  }
  const chunks: Blob[] = [];
  recorder.ondataavailable = (event) => chunks.push(event.data);
  const stopped = new Promise<void>((resolve) => {
    recorder.onstop = () => resolve();
  });
  recorder.start();

  return {
    stop: async () => {
      recorder.stop();
      await stopped;
      releaseMicrophone();
      const encoded = await new Blob(chunks, { type: recorder.mimeType }).arrayBuffer();
      const context = new AudioContext();
      try {
        const audio = await context.decodeAudioData(encoded);
        return encodeWav(mixToMono(audio), audio.sampleRate);
      } finally {
        await context.close();
      }
    },
  };
}

/** El navegador no ofrece grabación: página sin https o navegador sin MediaRecorder. */
export class RecordingUnavailableError extends Error {
  constructor() {
    super("grabación no disponible en este contexto");
    this.name = "RecordingUnavailableError";
  }
}

/** Mensaje para el usuario según por qué no se pudo empezar a grabar. */
export function describeRecordingStartError(error: unknown): string {
  if (error instanceof RecordingUnavailableError) {
    return window.isSecureContext
      ? "Este navegador no permite grabar audio. Prueba con otro navegador."
      : "El micrófono solo funciona si la página se abre con https.";
  }
  switch (error instanceof DOMException ? error.name : undefined) {
    case "NotAllowedError":
    case "SecurityError":
      return "No hay permiso para usar el micrófono. Revisa los permisos del navegador.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No se encuentra ningún micrófono conectado.";
    case "NotReadableError":
    case "AbortError":
      return "El micrófono no responde o lo está usando otra aplicación.";
    default:
      return "No se pudo empezar a grabar. Prueba otra vez.";
  }
}
