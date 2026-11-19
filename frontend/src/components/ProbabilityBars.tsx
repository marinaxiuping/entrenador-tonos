import { TONES } from "../syllables";

interface ProbabilityBarsProps {
  probabilities: Record<string, number>;
  targetTone: number;
}

/** Reparto de la probabilidad entre los cuatro tonos en una sola franja, con el objetivo marcado. */
export function ProbabilityBars({ probabilities, targetTone }: ProbabilityBarsProps) {
  const percentOf = (tone: number) => Math.round((probabilities[String(tone)] ?? 0) * 100);
  return (
    <div className="probability-strip">
      <div className="strip" aria-hidden="true">
        {TONES.map((tone) => (
          <span
            key={tone}
            className={`strip-segment tone-${tone}${tone === targetTone ? " is-target" : ""}`}
            style={{ width: `${percentOf(tone)}%` }}
          />
        ))}
      </div>
      <ul className="strip-legend">
        {TONES.map((tone) => (
          <li key={tone} className={`tone-${tone}${tone === targetTone ? " is-target" : ""}`}>
            T{tone} <strong>{percentOf(tone)} %</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}
