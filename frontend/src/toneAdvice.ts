// Consejo para corregir un intento, a partir del rasgo que define cada tono. Compara el
// contorno del usuario con el objetivo (los dos en semitonos respecto a su tono medio):
// es una regla sencilla sobre la forma, no una salida del modelo.

const EDGE_POINTS = 6;
// El usuario tiene que conseguir al menos esta fracción del movimiento del objetivo.
const MIN_MOVEMENT_RATIO = 0.5;
const MAX_EXTRA_RANGE_ST = 3;
const MAX_HEIGHT_GAP_ST = 3;

interface ContourShape {
  start: number;
  end: number;
  lowest: number;
  mean: number;
  range: number;
}

const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

function describeShape(contour: number[]): ContourShape | null {
  const values = contour.filter(Number.isFinite);
  if (values.length < 2 * EDGE_POINTS) return null;
  const lowest = Math.min(...values);
  return {
    start: average(values.slice(0, EDGE_POINTS)),
    end: average(values.slice(-EDGE_POINTS)),
    lowest,
    mean: average(values),
    range: Math.max(...values) - lowest,
  };
}

/** Un consejo concreto para el tono objetivo, o null si la forma no deja claro qué falla. */
export function toneAdvice(userContour: number[], targetContour: number[], targetTone: number): string | null {
  const user = describeShape(userContour);
  const target = describeShape(targetContour);
  if (!user || !target) return null;

  switch (targetTone) {
    case 1:
      if (user.range > target.range + MAX_EXTRA_RANGE_ST) return "Mantén la voz quieta: el tono 1 no sube ni baja.";
      if (user.mean < target.mean - MAX_HEIGHT_GAP_ST) return "Ponlo más agudo: el tono 1 va en la parte alta de tu voz.";
      return null;
    case 2:
      if (user.end - user.start < MIN_MOVEMENT_RATIO * (target.end - target.start)) {
        return "Sube más hacia el final: el tono 2 empieza a media altura y termina arriba.";
      }
      return null;
    case 3:
      if (user.start - user.lowest < MIN_MOVEMENT_RATIO * (target.start - target.lowest)) {
        return "Baja más: el tono 3 cae hasta la zona más grave de tu voz.";
      }
      return null;
    case 4:
      if (user.start - user.end < MIN_MOVEMENT_RATIO * (target.start - target.end)) {
        return "Cae con más fuerza: el tono 4 empieza arriba y baja de golpe.";
      }
      if (user.start < target.start - MAX_HEIGHT_GAP_ST) return "Empieza más agudo: el tono 4 arranca en lo alto de tu voz.";
      return null;
    default:
      return null;
  }
}
