import { TONES } from "../syllables";

interface ProbabilityBarsProps {
  probabilities: Record<string, number>;
  targetTone: number;
}

export function ProbabilityBars({ probabilities, targetTone }: ProbabilityBarsProps) {
  return (
    <ul className="probability-bars">
      {TONES.map((tone) => {
        const probability = probabilities[String(tone)] ?? 0;
        return (
          <li key={tone} className={tone === targetTone ? "is-target" : undefined}>
            <span className="bar-label">T{tone}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${probability * 100}%` }} />
            </span>
            <span className="bar-value">{Math.round(probability * 100)} %</span>
          </li>
        );
      })}
    </ul>
  );
}
