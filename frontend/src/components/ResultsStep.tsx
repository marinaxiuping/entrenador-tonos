import type { JudgeName } from "../api";
import resultsData from "../data/resultados.json";
import { JUDGES, JUDGE_ORDER, formatPercent } from "../judges";

// Generado por scripts/36_exportar_resultados.py a partir de las salidas de los experimentos.
interface Rate {
  etiqueta: string;
  accuracy: number;
  ic95?: number[];
}

interface Results {
  escalera: (Rate & { ece: number | null })[];
  contexto: (Rate & { modelo: string })[];
  f0_plano: (Rate & { condicion: string })[];
  inversion: { juez: JudgeName; accuracy_nativa: number; detecta: number; detecta_ic95: number[]; acepta: number }[];
  ejemplos: {
    frase: string;
    silaba: string;
    pinyin: string;
    tono_original: number;
    tono_impuesto: number;
    audio_original: string;
    audio_cambiado: string;
    veredictos: { juez: JudgeName; tono: number | null }[];
  }[];
  n: { silabas_aishell3: number; hablantes_aishell3: number; silabas_cambiadas: number };
}

const RESULTS = resultsData as Results;
const COUNT = new Intl.NumberFormat("es-ES");

function RateRows({ rows }: { rows: Rate[] }) {
  return (
    <ul className="rate-rows">
      {rows.map((row) => (
        <li key={row.etiqueta}>
          <span className="rate-label">{row.etiqueta}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${row.accuracy * 100}%` }} />
          </span>
          <span className="bar-value">
            {formatPercent(row.accuracy)}
            {row.ic95 && <span className="judge-detail">[{formatPercent(row.ic95[0])}; {formatPercent(row.ic95[1])}]</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function ResultsStep() {
  const inversion = JUDGE_ORDER.map((judge) => RESULTS.inversion.find((row) => row.juez === judge)).filter(
    (row) => row !== undefined,
  );

  return (
    <>
      <section className="card">
        <h2>1. El 99 % no sobrevive al habla real</h2>
        <p className="muted">
          El modelo de la app se entrenó con sílabas aisladas de estudio (Tone Perfect). En habla continua (AISHELL-3,{" "}
          {COUNT.format(RESULTS.n.silabas_aishell3)} sílabas de {RESULTS.n.hablantes_aishell3} hablantes) los tonos se
          comprimen y el modelo se hunde, además con una confianza que ya no merece.
        </p>
        <RateRows rows={RESULTS.escalera} />
        <p className="small muted">
          Error de calibración (ECE) del modelo sin tocar: {formatPercent(RESULTS.escalera[1].ece ?? 0)}. Entrenar en el
          dominio recupera buena parte; tener 14 veces más datos apenas añade.
        </p>
      </section>

      <section className="card">
        <h2>2. Más contexto, más accuracy</h2>
        <p className="muted">
          Mismas sílabas, mismos hablantes no vistos. Intervalos al 95 % por bootstrap de hablantes. Ojo a la última
          fila: sin oír nada, sabiendo solo qué carácter se dice, se acierta casi tanto como el mejor modelo.
        </p>
        <RateRows rows={RESULTS.contexto} />
      </section>

      <section className="card">
        <h2>3. Pero ¿quién oye de verdad el tono?</h2>
        <p className="muted">
          Se cambia por ordenador el tono de una sola sílaba en {COUNT.format(RESULTS.n.silabas_cambiadas)} frases, como
          haría un alumno que se equivoca. Un buen juez debe darse cuenta.
        </p>
        <table className="verdict-table">
          <thead>
            <tr>
              <th scope="col">Juez</th>
              <th scope="col">Acierta con nativos</th>
              <th scope="col">Detecta el error</th>
              <th scope="col">Lo deja pasar</th>
            </tr>
          </thead>
          <tbody>
            {inversion.map((row) => (
              <tr key={row.juez}>
                <th scope="row">
                  {JUDGES[row.juez].name}
                  <span className="judge-detail">{JUDGES[row.juez].hears}</span>
                </th>
                <td>{formatPercent(row.accuracy_nativa)}</td>
                <td>
                  <span className="outcome ok">{formatPercent(row.detecta)}</span>
                </td>
                <td>
                  <span className="outcome ko">{formatPercent(row.acepta)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small muted">
          El orden se invierte: el modelo que más acierta con nativos es el que más errores deja pasar, porque decide en
          buena parte por qué palabra se dice y no por cómo suena.
        </p>
        <h3>Con el tono borrado del todo</h3>
        <p className="small muted">Accuracy de HuBERT al aplanar el F0 de frases completas:</p>
        <RateRows rows={RESULTS.f0_plano} />
      </section>

      <section className="card">
        <h2>4. Escúchalo</h2>
        <p className="muted">Ejemplos elegidos al azar. Prueba los tuyos en la pestaña «¿Quién te corrige?».</p>
        {RESULTS.ejemplos.map((example) => (
          <div className="example" key={example.audio_cambiado}>
            <p>
              <span className="hanzi-inline">{example.frase}</span>
              <br />
              <strong>{example.silaba} {example.pinyin}</strong>: T{example.tono_original} cambiado a T{example.tono_impuesto}
            </p>
            <div className="example-audio">
              <label>
                Original
                <audio controls preload="none" src={example.audio_original} />
              </label>
              <label>
                Cambiado
                <audio controls preload="none" src={example.audio_cambiado} />
              </label>
            </div>
            <ul className="example-verdicts">
              {example.veredictos.map((verdict) => (
                <li key={verdict.juez}>
                  {JUDGES[verdict.juez].name}:{" "}
                  <span className={`outcome ${verdict.tono === example.tono_impuesto ? "ok" : verdict.tono === example.tono_original ? "ko" : "neutral"}`}>
                    {verdict.tono === null ? "sin voz" : `T${verdict.tono}`}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    </>
  );
}
