interface ContourChartProps {
  userContour: number[];
  targetContour: number[] | null;
  userLabel?: string;
  targetLabel?: string;
  note?: string;
}

const WIDTH = 360;
const HEIGHT = 200;
const PADDING = { top: 12, right: 12, bottom: 28, left: 40 };
// Rango mínimo del eje: un contorno plano no debe parecer una montaña por el autoescalado.
const MIN_HALF_RANGE_ST = 8;

function toPath(values: number[], x: (i: number) => number, y: (v: number) => number): string {
  return values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
}

export function ContourChart({
  userContour,
  targetContour,
  userLabel = "tu voz",
  targetLabel = "objetivo",
  note = "semitonos respecto a tu tono medio",
}: ContourChartProps) {
  const all = [...userContour, ...(targetContour ?? [])];
  const top = Math.max(MIN_HALF_RANGE_ST, Math.ceil(Math.max(...all) + 1));
  const bottom = Math.min(-MIN_HALF_RANGE_ST, Math.floor(Math.min(...all) - 1));
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
  const x = (i: number) => PADDING.left + (i / (userContour.length - 1)) * plotWidth;
  const y = (v: number) => PADDING.top + ((top - v) / (top - bottom)) * plotHeight;
  const ticks = [bottom, Math.round((bottom + top) / 2), 0, top].filter((v, i, arr) => arr.indexOf(v) === i);

  return (
    <figure className="contour-chart">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`Contorno de tono: ${userLabel} frente a ${targetLabel}`}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className={tick === 0 ? "axis-zero" : "grid"} x1={PADDING.left} x2={WIDTH - PADDING.right} y1={y(tick)} y2={y(tick)} />
            <text className="tick" x={PADDING.left - 6} y={y(tick)} textAnchor="end" dominantBaseline="middle">
              {tick}
            </text>
          </g>
        ))}
        <text className="tick" x={PADDING.left + plotWidth / 2} y={HEIGHT - 6} textAnchor="middle">
          tiempo
        </text>
        {targetContour && <path className="target-line" d={toPath(targetContour, x, y)} />}
        <path className="user-line" d={toPath(userContour, x, y)} />
      </svg>
      <figcaption>
        <span className="legend user">{userLabel}</span>
        {targetContour && <span className="legend target">{targetLabel}</span>}
        <span className="legend-note">{note}</span>
      </figcaption>
    </figure>
  );
}
