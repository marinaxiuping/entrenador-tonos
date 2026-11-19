// Contorno de cada tono en la escala de cinco niveles de Chao (5 = el más agudo): 55, 35, 214, 51.
const CHAO_LEVELS: Record<number, number[]> = { 1: [5, 5], 2: [3, 5], 3: [2, 1, 4], 4: [5, 1] };

const WIDTH = 40;
const HEIGHT = 24;
const MARGIN = 4;

const xAt = (position: number) => MARGIN + position * (WIDTH - 2 * MARGIN);
const yAt = (level: number) => MARGIN + ((5 - level) / 4) * (HEIGHT - 2 * MARGIN);

function shapePath(levels: number[]): string {
  if (levels.length === 2) {
    return `M${xAt(0)},${yAt(levels[0])} L${xAt(1)},${yAt(levels[1])}`;
  }
  // Curva cuadrática cuyo punto medio cae en el nivel más bajo del tono 3.
  const [start, low, end] = levels;
  const controlY = 2 * yAt(low) - (yAt(start) + yAt(end)) / 2;
  return `M${xAt(0)},${yAt(start)} Q${xAt(0.5)},${controlY} ${xAt(1)},${yAt(end)}`;
}

/** Dibujo pequeño de la forma del tono, para reconocerlo de un vistazo. */
export function ToneShape({ tone }: { tone: number }) {
  const levels = CHAO_LEVELS[tone];
  if (!levels) return null;
  return (
    <svg className="tone-shape" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} aria-hidden="true">
      <line className="tone-shape-guide" x1={MARGIN} x2={WIDTH - MARGIN} y1={HEIGHT / 2} y2={HEIGHT / 2} />
      <path className="tone-shape-line" d={shapePath(levels)} />
    </svg>
  );
}
