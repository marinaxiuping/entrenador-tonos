import toneGuideData from "../data/tone_guide.json";

interface ToneGuide {
  contour_st: number[];
  duration_s: number;
}

// Generado por scripts/06_exportar_guia_tonos.py: el mismo contorno objetivo que dibuja la
// gráfica y la duración típica de cada tono en Tone Perfect.
const TONE_GUIDES = toneGuideData as Record<string, ToneGuide>;

const ATTACK_S = 0.06;
const RELEASE_S = 0.08;
const PEAK_GAIN = 0.25;
// Armónicos que caen como 1/n² (como la fuente glotal) y filtro bajo: suena a «mmm»
// tarareado; un diente de sierra (1/n) sonaba metálico, como una trompeta.
const HUM_HARMONICS = 12;
const FILTER_HARMONICS = 2.5;

let sharedContext: AudioContext | null = null;

function audioContext(): AudioContext {
  sharedContext ??= new AudioContext();
  return sharedContext;
}

/** Tararea el contorno del tono a la altura de la voz del usuario y resuelve al terminar. */
export async function playToneGuide(tone: number, referenceHz: number): Promise<void> {
  const guide = TONE_GUIDES[String(tone)];
  if (!guide) throw new Error(`No hay guía para el tono ${tone}`);
  const context = audioContext();
  // Los navegadores suspenden el contexto hasta un gesto del usuario.
  if (context.state === "suspended") await context.resume();

  const start = context.currentTime + 0.05;
  const end = start + guide.duration_s;
  const frequencies = Float32Array.from(guide.contour_st, (st) => referenceHz * 2 ** (st / 12));

  const amplitudes = Float32Array.from({ length: HUM_HARMONICS + 1 }, (_, n) => (n === 0 ? 0 : 1 / n ** 2));
  const humWave = context.createPeriodicWave(amplitudes, new Float32Array(amplitudes.length));
  const oscillator = new OscillatorNode(context, { periodicWave: humWave, frequency: frequencies[0] });
  oscillator.frequency.setValueCurveAtTime(frequencies, start, guide.duration_s);
  const filter = new BiquadFilterNode(context, { type: "lowpass", frequency: referenceHz * FILTER_HARMONICS });
  const gain = new GainNode(context, { gain: 0 });
  gain.gain.setValueAtTime(0, start);
  gain.gain.linearRampToValueAtTime(PEAK_GAIN, start + ATTACK_S);
  gain.gain.setValueAtTime(PEAK_GAIN, end - RELEASE_S);
  gain.gain.linearRampToValueAtTime(0, end);

  oscillator.connect(filter).connect(gain).connect(context.destination);
  const finished = new Promise<void>((resolve) => {
    oscillator.onended = () => {
      oscillator.disconnect();
      resolve();
    };
  });
  oscillator.start(start);
  oscillator.stop(end + 0.02);
  return finished;
}
