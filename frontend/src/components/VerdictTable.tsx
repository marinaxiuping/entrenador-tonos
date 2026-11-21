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

/** El detector juzga el tono del texto: avisar es acertar solo si se ha cambiado. */
function detectorOutcome(alert: boolean, imposedTone: number | null): Outcome {
  if (imposedTone === null) return alert ? { text: "Se equivoca", kind: "ko" } : { text: "Acierta", kind: "ok" };
  return alert ? { text: "Detecta el cambio", kind: "ok" } : { text: "Deja pasar el error", kind: "ko" };
}

function outcomeOf(verdict: Verdict, originalTone: number, imposedTone: number | null): Outcome {
  if (verdict.alert !== null) return detectorOutcome(verdict.alert, imposedTone);
  const tone = verdict.tone;
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
          const outcome = outcomeOf(verdict, originalTone, imposedTone);
          const confidence = verdict.tone !== null ? verdict.probabilities?.[String(verdict.tone)] : undefined;
          return (
            <tr key={verdict.judge}>
              <th scope="row">
                {JUDGES[verdict.judge].name}
                <span className="judge-detail">{JUDGES[verdict.judge].hears}</span>
              </th>
              <td>
                {verdict.alert !== null ? (
                  <>
                    {verdict.alert ? "Tono mal" : "Tono bien"}
                    {verdict.error_probability !== null && (
                      <span className="judge-detail">{formatPercent(verdict.error_probability)} de error</span>
                    )}
                  </>
                ) : (
                  <>
                    {verdict.tone !== null ? `T${verdict.tone}` : "—"}
                    {confidence !== undefined && verdict.judge !== "lexico" && (
                      <span className="judge-detail">{formatPercent(confidence)}</span>
                    )}
                  </>
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
