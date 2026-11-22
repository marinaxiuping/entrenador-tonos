export const ROUNDS_PER_GAME = 5;
// Margen alrededor de la curva objetivo. Con 2,5 st un T3 cantado plano sacaba el 100 % y un
// T2 que sube la mitad también; con 1,5 st se quedan en el 63 % y el 81 %, y una voz que sigue
// la forma con 1,5 st de desvío en el perfil sigue sacando el 100 %.
export const TOLERANCE_ST = 1.5;
// Una sílaba dura medio segundo: a esa velocidad no da tiempo a ver la curva y corregirse.
export const TIME_STRETCH = 2.2;
// El recorrido se divide en tramos y cada uno cuenta si la voz pasó por la curva en él:
// así no puntúa quien solo acierta al principio y se calla.
const SCORE_SEGMENTS = 16;

export interface PitchSample {
  /** Posición en el recorrido, de 0 (inicio) a 1 (final). */
  progress: number;
  semitones: number;
}

export function hzToSemitones(hz: number, referenceHz: number): number {
  return 12 * Math.log2(hz / referenceHz);
}

/** Altura de la curva objetivo en una posición del recorrido, interpolando entre puntos. */
export function targetAt(contour: number[], progress: number): number {
  const position = Math.min(Math.max(progress, 0), 1) * (contour.length - 1);
  const index = Math.floor(position);
  if (index >= contour.length - 1) return contour[contour.length - 1];
  const fraction = position - index;
  return contour[index] * (1 - fraction) + contour[index + 1] * fraction;
}

export function isOnTarget(sample: PitchSample, contour: number[]): boolean {
  return Math.abs(sample.semitones - targetAt(contour, sample.progress)) <= TOLERANCE_ST;
}

/** Fracción de tramos del recorrido en los que la voz estuvo dentro del margen de la curva. */
export function scoreRound(samples: PitchSample[], contour: number[]): number {
  const hitSegments = new Set<number>();
  for (const sample of samples) {
    if (sample.progress < 0 || sample.progress > 1 || !isOnTarget(sample, contour)) continue;
    hitSegments.add(Math.min(Math.floor(sample.progress * SCORE_SEGMENTS), SCORE_SEGMENTS - 1));
  }
  return hitSegments.size / SCORE_SEGMENTS;
}

export function describeScore(score: number): string {
  if (score >= 0.9) return "¡Clavado!";
  if (score >= 0.6) return "Casi: repasa la forma del tono.";
  return "Otra vez: escucha la melodía antes de cantar.";
}

export interface PitchAxis {
  bottomSt: number;
  topSt: number;
}

// Sitio por encima y por debajo de las curvas para ver la voz cuando se sale del margen.
const AXIS_PADDING_ST = 4;

/** Eje común a todas las rondas, para que la altura de cada tono se compare entre rondas. */
export function axisFor(contours: number[][]): PitchAxis {
  const values = contours.flat();
  return {
    bottomSt: Math.floor(Math.min(...values) - AXIS_PADDING_ST),
    topSt: Math.ceil(Math.max(...values) + AXIS_PADDING_ST),
  };
}
