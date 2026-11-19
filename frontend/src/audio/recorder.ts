import { encodeWav, mixToMono } from "./wav";

export interface RecordingSession {
  /** Para la grabación y devuelve el audio como WAV. */
  stop: () => Promise<Blob>;
}

/**
 * Graba del micrófono con el formato nativo del navegador (webm u mp4) y lo convierte a
 * WAV al parar, porque el backend no decodifica webm/opus.
 */
export async function startRecording(): Promise<RecordingSession> {
  // El procesado de voz del navegador (supresión de ruido, control de ganancia) altera
  // el pitch y la energía, así que se desactiva.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const recorder = new MediaRecorder(stream);
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
      stream.getTracks().forEach((track) => track.stop());
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
