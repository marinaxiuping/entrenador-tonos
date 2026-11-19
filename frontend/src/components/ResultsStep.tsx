import type { JudgeName } from "../api";
import resultsData from "../data/resultados.json";
import { JUDGES, JUDGE_ORDER, formatPercent } from "../judges";
import { translateSentence } from "../translations";
import { AudioPlayer } from "./AudioPlayer";

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
  semillas: { n: number; sd_maxima_contrafactual: number; mismo_orden: boolean };
  n: { silabas_aishell3: number; hablantes_aishell3: number; silabas_cambiadas: number };
}

const RESULTS = resultsData as Results;
const COUNT = new Intl.NumberFormat("es-ES");

function findRow<T>(rows: T[], matches: (row: T) => boolean): T {
  const row = rows.find(matches);
  if (!row) throw new Error("Falta una fila en resultados.json: regenerarlo con el script 36");
  return row;
}

const TONE_PERFECT = RESULTS.escalera[0];
const ZERO_SHOT = RESULTS.escalera[1];
const HUBERT_ACCURACY = findRow(RESULTS.contexto, (row) => row.modelo === "hubert");
const LEXICON_ACCURACY = findRow(RESULTS.contexto, (row) => row.modelo === "lexico_sin_audio");
const CNN_DETECTION = findRow(RESULTS.inversion, (row) => row.juez === "cnn_silaba");
const HUBERT_DETECTION = findRow(RESULTS.inversion, (row) => row.juez === "hubert");

interface Stat {
  value: number;
  label: string;
}

/** Las dos cifras que resumen una sección, en grande, para leerla de un vistazo. */
function HeadlineStats({ first, second, connector }: { first: Stat; second: Stat; connector: string }) {
  return (
    <div className="headline-stats">
      <p className="big-stat">
        <span className="big-stat-value">{formatPercent(first.value)}</span>
        <span className="big-stat-label">{first.label}</span>
      </p>
      <span className="headline-connector" aria-hidden="true">{connector}</span>
      <p className="big-stat is-accent">
        <span className="big-stat-value">{formatPercent(second.value)}</span>
        <span className="big-stat-label">{second.label}</span>
      </p>
    </div>
  );
}

function SectionTitle({ number, children }: { number: number; children: string }) {
  return (
    <h2 className="section-title">
      <span className="section-number">{String(number).padStart(2, "0")}</span>
      {children}
    </h2>
  );
}

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
        <SectionTitle number={1}>El 99 % no sobrevive al habla real</SectionTitle>
        <HeadlineStats
          first={{ value: TONE_PERFECT.accuracy, label: "sílabas de estudio (Tone Perfect)" }}
          second={{ value: ZERO_SHOT.accuracy, label: "habla continua (AISHELL-3)" }}
          connector="→"
        />
        <p className="muted">
          El modelo de la app aprendió con sílabas aisladas. En {COUNT.format(RESULTS.n.silabas_aishell3)} sílabas de
          habla continua de {RESULTS.n.hablantes_aishell3} hablantes los tonos se comprimen y se hunde, con una confianza
          que ya no merece (ECE {formatPercent(ZERO_SHOT.ece ?? 0)}).
        </p>
        <RateRows rows={RESULTS.escalera} />
        <p className="small muted">Entrenar en el dominio recupera buena parte; 14 veces más datos apenas añaden.</p>
      </section>

      <section className="card">
        <SectionTitle number={2}>Más contexto, más accuracy</SectionTitle>
        <HeadlineStats
          first={{ value: HUBERT_ACCURACY.accuracy, label: "HuBERT, el mejor modelo" }}
          second={{ value: LEXICON_ACCURACY.accuracy, label: "sin oír nada, solo el carácter" }}
          connector="≈"
        />
        <p className="muted">
          Mismas sílabas y hablantes no vistos, con intervalos al 95 % por bootstrap de hablantes. Saber qué palabra se
          dice ya acierta casi tanto como el mejor modelo.
        </p>
        <RateRows rows={RESULTS.contexto} />
      </section>

      <section className="card">
        <SectionTitle number={3}>Pero ¿quién oye de verdad el tono?</SectionTitle>
        <HeadlineStats
          first={{ value: CNN_DETECTION.detecta, label: "errores que detecta la CNN por sílaba" }}
          second={{ value: HUBERT_DETECTION.detecta, label: "errores que detecta HuBERT" }}
          connector="vs"
        />
        <p className="muted">
          Se cambia por ordenador el tono de una sílaba en {COUNT.format(RESULTS.n.silabas_cambiadas)} frases, como
          haría un alumno que se equivoca. El orden se invierte: el que más acierta con nativos es el que más errores
          deja pasar, porque decide por la palabra y no por cómo suena.
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
          Repetido con {RESULTS.semillas.n} semillas de entrenamiento: las tasas varían como mucho{" "}
          {formatPercent(RESULTS.semillas.sd_maxima_contrafactual)} (desviación típica)
          {RESULTS.semillas.mismo_orden ? " y el orden de los jueces es el mismo en todas." : ", pero el orden de los jueces cambia entre semillas."}
        </p>
        <h3>Con el tono borrado del todo</h3>
        <p className="small muted">Accuracy de HuBERT al aplanar el F0 de frases completas:</p>
        <RateRows rows={RESULTS.f0_plano} />
      </section>

      <section className="card">
        <SectionTitle number={4}>Escúchalo</SectionTitle>
        <p className="muted">Ejemplos elegidos al azar. Prueba los tuyos en la pestaña «¿Quién corrige?».</p>
        {RESULTS.ejemplos.map((example) => {
          const translation = translateSentence(example.frase);
          return (
            <div className="example" key={example.audio_cambiado}>
              <p className="hanzi-inline">{example.frase}</p>
              {translation && <p className="translation">{translation}</p>}
              <p className="example-change">
                <strong>{example.silaba} {example.pinyin}</strong>: T{example.tono_original} cambiado a T{example.tono_impuesto}
              </p>
              <div className="example-audio">
                <AudioPlayer src={example.audio_original} label="Original" preload="none" />
                <AudioPlayer src={example.audio_cambiado} label="Cambiado" preload="none" />
              </div>
              <ul className="example-verdicts">
                {example.veredictos.map((verdict) => (
                  <li key={verdict.juez}>
                    {JUDGES[verdict.juez].name}{" "}
                    <span className={`outcome ${verdict.tono === example.tono_impuesto ? "ok" : verdict.tono === example.tono_original ? "ko" : "neutral"}`}>
                      {verdict.tono === null ? "sin voz" : `T${verdict.tono}`}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </section>
    </>
  );
}
