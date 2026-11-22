export interface PitchSearchRange {
  minHz: number;
  maxHz: number;
}

// Umbral de la diferencia normalizada de YIN: por encima, el fragmento no es periódico.
const APERIODICITY_THRESHOLD = 0.15;
// Por debajo de esta energía (unos -40 dBFS) el fragmento es silencio o ruido de fondo.
const MIN_RMS = 0.01;

/**
 * Estima la frecuencia fundamental de un fragmento de audio con YIN (de Cheveigné y
 * Kawahara, 2002). Devuelve null si el fragmento es silencio o no tiene un periodo claro.
 */
export function estimatePitchHz(frame: Float32Array, sampleRate: number, range: PitchSearchRange): number | null {
  const minLag = Math.max(2, Math.floor(sampleRate / range.maxHz));
  const maxLag = Math.ceil(sampleRate / range.minHz);
  const windowSize = frame.length - maxLag - 1;
  if (windowSize < maxLag || minLag >= maxLag) {
    throw new Error("El fragmento es demasiado corto para el rango de frecuencias pedido");
  }
  if (rootMeanSquare(frame) < MIN_RMS) return null;

  const normalized = cumulativeMeanNormalizedDifference(frame, windowSize, maxLag + 1);
  const lag = firstDipBelowThreshold(normalized, minLag, maxLag);
  return lag === null ? null : sampleRate / refineLag(normalized, lag);
}

function rootMeanSquare(frame: Float32Array): number {
  let sumOfSquares = 0;
  for (const sample of frame) sumOfSquares += sample * sample;
  return Math.sqrt(sumOfSquares / frame.length);
}

function cumulativeMeanNormalizedDifference(frame: Float32Array, windowSize: number, maxLag: number): Float32Array {
  const normalized = new Float32Array(maxLag + 1);
  normalized[0] = 1;
  let runningSum = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    let difference = 0;
    for (let i = 0; i < windowSize; i++) {
      const delta = frame[i] - frame[i + lag];
      difference += delta * delta;
    }
    runningSum += difference;
    normalized[lag] = runningSum === 0 ? 1 : (difference * lag) / runningSum;
  }
  return normalized;
}

/** Primer mínimo local por debajo del umbral: tomar el primero evita los errores de octava baja. */
function firstDipBelowThreshold(normalized: Float32Array, minLag: number, maxLag: number): number | null {
  for (let lag = minLag; lag <= maxLag; lag++) {
    if (normalized[lag] < APERIODICITY_THRESHOLD) {
      while (lag < maxLag && normalized[lag + 1] < normalized[lag]) lag++;
      return lag;
    }
  }
  return null;
}

/** Interpolación parabólica alrededor del mínimo, para no quedar limitado a periodos enteros. */
function refineLag(normalized: Float32Array, lag: number): number {
  const before = normalized[lag - 1];
  const at = normalized[lag];
  const after = normalized[lag + 1];
  const curvature = before - 2 * at + after;
  return curvature > 0 ? lag + (before - after) / (2 * curvature) : lag;
}
