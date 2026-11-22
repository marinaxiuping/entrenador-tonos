import { RAW_VOICE_CONSTRAINTS, RecordingUnavailableError } from "./recorder";
import { estimatePitchHz, type PitchSearchRange } from "./yin";

export interface PitchTracker {
  stop: () => void;
}

// Unos 43 ms a 48 kHz; se agranda si no caben dos periodos de la voz más grave del rango.
const MIN_FRAME_SIZE = 2048;
// Mediana de los últimos fragmentos: quita los saltos de octava sueltos sin añadir retardo visible.
const SMOOTHING_FRAMES = 3;

/**
 * Abre el micrófono y llama a onPitch en cada fotograma de pantalla con la F0 suavizada en Hz,
 * o null si no hay voz. Hay que llamarla desde un gesto del usuario: Safari solo deja arrancar
 * el audio dentro de uno, por eso el contexto se crea antes de pedir el micrófono.
 */
export async function startPitchTracker(
  range: PitchSearchRange,
  onPitch: (hz: number | null) => void,
): Promise<PitchTracker> {
  if (!navigator.mediaDevices?.getUserMedia) throw new RecordingUnavailableError();
  const context = new AudioContext();
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(RAW_VOICE_CONSTRAINTS);
  } catch (error) {
    await context.close();
    throw error;
  }
  if (context.state === "suspended") await context.resume();

  const frameSize = frameSizeFor(context.sampleRate, range.minHz);
  const analyser = new AnalyserNode(context, { fftSize: frameSize });
  // Safari no procesa nodos que no acaban en la salida: se conecta a ella en silencio.
  const mute = new GainNode(context, { gain: 0 });
  context.createMediaStreamSource(stream).connect(analyser).connect(mute).connect(context.destination);

  const frame = new Float32Array(frameSize);
  const recentPitches: number[] = [];
  let animationFrame = requestAnimationFrame(function analyzeFrame() {
    analyser.getFloatTimeDomainData(frame);
    const hz = estimatePitchHz(frame, context.sampleRate, range);
    if (hz === null) {
      recentPitches.length = 0;
      onPitch(null);
    } else {
      recentPitches.push(hz);
      if (recentPitches.length > SMOOTHING_FRAMES) recentPitches.shift();
      onPitch(median(recentPitches));
    }
    animationFrame = requestAnimationFrame(analyzeFrame);
  });

  let isStopped = false;
  return {
    stop: () => {
      if (isStopped) return;
      isStopped = true;
      cancelAnimationFrame(animationFrame);
      stream.getTracks().forEach((track) => track.stop());
      void context.close();
    },
  };
}

/** Potencia de dos (lo que exige AnalyserNode) con sitio para la ventana y el retardo de YIN. */
function frameSizeFor(sampleRate: number, minHz: number): number {
  const longestPeriod = Math.ceil(sampleRate / minHz);
  let size = MIN_FRAME_SIZE;
  while (size < 2 * longestPeriod + 2) size *= 2;
  return size;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
