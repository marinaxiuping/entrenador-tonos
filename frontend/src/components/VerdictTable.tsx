import type { Verdict } from "../api";
import { JUDGES, formatPercent } from "../judges";

interface VerdictTableProps {
  verdicts: Verdict[];
  originalTone: number;
  imposedTone: number | null;
}

interface Outcome {
  text: string;
  kind: "ok" | "ko" | "neutral";
}

function outcomeOf(tone: number | null, originalTone: number, imposedTone: number | null): Outcome {
  if (tone === null) return { text: "No detecta voz", kind: "neutral" };
  if (imposedTone === null) {
    return tone === originalTone ? { text: "Acierta", kind: "ok" } : { text: "Se equivoca", kind: "ko" };
  }
  if (tone === imposedTone) return { text: "Detecta el cambio", kind: "ok" };
  if (tone === originalTone) return { text: "Deja pasar el error", kind: "ko" };
  return { text: "Oye otro tono", kind: "neutral" };
}

export function VerdictTable({ verdicts, originalTone, imposedTone }: VerdictTableProps) {
  return (
    <table className="verdict-table">
      <thead>
        <tr>
          <th scope="col">Juez</th>
          <th scope="col">Oye</th>
          <th scope="col">Resultado</th>
        </tr>
      </thead>
      <tbody>
        {verdicts.map((verdict) => {
          const outcome = outcomeOf(verdict.tone, originalTone, imposedTone);
          const confidence = verdict.tone !== null ? verdict.probabilities?.[String(verdict.tone)] : undefined;
          return (
            <tr key={verdict.judge}>
              <th scope="row">
                {JUDGES[verdict.judge].name}
                <span className="judge-detail">{JUDGES[verdict.judge].hears}</span>
              </th>
              <td>
                {verdict.tone !== null ? `T${verdict.tone}` : "—"}
                {confidence !== undefined && verdict.judge !== "lexico" && (
                  <span className="judge-detail">{formatPercent(confidence)}</span>
                )}
              </td>
              <td>
                <span className={`outcome ${outcome.kind}`}>{outcome.text}</span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
