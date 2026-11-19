import { useEffect, useState } from "react";
import { base64ToWavBlob, fetchLabUtterances, judgeSyllable, type JudgeResult, type LabUtterance } from "../api";
import { JUDGE_ORDER } from "../judges";
import { TONES, TONE_NAMES, type Tone } from "../syllables";
import { translateSentence } from "../translations";
import { AudioPlayer } from "./AudioPlayer";
import { ContourChart } from "./ContourChart";
import { StepHeading } from "./StepHeading";
import { ToneShape } from "./ToneShape";
import { VerdictTable } from "./VerdictTable";

interface Judgement {
  result: JudgeResult;
  audioUrl: string;
}

export function LabStep() {
  const [utterances, setUtterances] = useState<LabUtterance[] | null>(null);
  const [utteranceId, setUtteranceId] = useState<string>("");
  const [syllableIndex, setSyllableIndex] = useState<number | null>(null);
  const [imposedTone, setImposedTone] = useState<Tone | null>(null);
  const [judgement, setJudgement] = useState<Judgement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isJudging, setIsJudging] = useState(false);

  useEffect(() => {
    fetchLabUtterances()
      .then((loaded) => {
        setUtterances(loaded);
        setUtteranceId(loaded[0]?.id ?? "");
      })
      .catch((failure: unknown) =>
        setError(failure instanceof Error ? failure.message : "No se pudieron cargar las frases."));
  }, []);

  // Libera el audio del juicio anterior al sustituirlo o al salir.
  useEffect(() => () => {
    if (judgement) URL.revokeObjectURL(judgement.audioUrl);
  }, [judgement]);

  const utterance = utterances?.find((candidate) => candidate.id === utteranceId);
  const syllable = utterance && syllableIndex !== null ? utterance.syllables[syllableIndex] : undefined;
  const translation = utterance ? translateSentence(utterance.text) : undefined;

  const selectUtterance = (id: string) => {
    setUtteranceId(id);
    setSyllableIndex(null);
    setImposedTone(null);
    setJudgement(null);
  };

  const selectSyllable = (index: number) => {
    setSyllableIndex(index);
    setImposedTone(null);
    setJudgement(null);
  };

  const judge = async (tone: Tone | null) => {
    if (!utterance || syllableIndex === null) return;
    setIsJudging(true);
    setError(null);
    try {
      const result = await judgeSyllable(utterance.id, syllableIndex, tone);
      const verdicts = JUDGE_ORDER.map((name) => result.verdicts.find((v) => v.judge === name)).filter((v) => v !== undefined);
      setJudgement({
        result: { ...result, verdicts },
        audioUrl: URL.createObjectURL(base64ToWavBlob(result.audio_wav_base64)),
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "No se pudo juzgar la sílaba.");
    } finally {
      setIsJudging(false);
    }
  };

  if (!utterances) {
    return (
      <section className="card">
        <h2>¿Quién te corrige?</h2>
        {error ? <p className="error" role="alert">{error}</p> : <p className="muted">Cargando frases…</p>}
      </section>
    );
  }

  return (
    <section className="card">
      <h2>¿Quién te corrige?</h2>
      <p className="muted">
        Frases reales de hablantes nativos que ningún modelo ha oído al entrenar. Cambia el tono de una sílaba,
        como haría un alumno que se equivoca, y mira qué modelo se da cuenta.
      </p>

      <StepHeading number={1}>Elige una frase</StepHeading>
      <div className="field">
        <label htmlFor="utterance-select">Frase</label>
        <select id="utterance-select" value={utteranceId} onChange={(event) => selectUtterance(event.target.value)}>
          {utterances.map((option) => (
            <option key={option.id} value={option.id}>
              {option.text}
            </option>
          ))}
        </select>
      </div>
      {translation && <p className="translation lab-translation">{translation}</p>}

      {utterance && <StepHeading number={2}>Toca la sílaba que quieres cambiar</StepHeading>}
      {utterance && (
        <div className="syllable-row" role="radiogroup" aria-label="Sílaba que se cambia">
          {utterance.syllables.map((option, index) => (
            <button
              key={index}
              type="button"
              role="radio"
              aria-checked={index === syllableIndex}
              className={index === syllableIndex ? "selected" : undefined}
              onClick={() => selectSyllable(index)}
              disabled={!option.evaluable}
              title={option.evaluable ? `Tono ${option.tone}` : "Tono neutro: no se puede cambiar"}
            >
              <span className="hanzi">{option.hanzi}</span>
              <span className="pinyin">{option.pinyin}</span>
            </button>
          ))}
        </div>
      )}

      {syllable && (
        <>
          <StepHeading number={3}>Cámbiale el tono</StepHeading>
          <p>
            <strong>{syllable.hanzi} {syllable.pinyin}</strong> se dice con el tono {syllable.tone}:{" "}
            {TONE_NAMES[syllable.tone as Tone]}. ¿A qué tono la cambias?
          </p>
          <div className="lab-tones" role="radiogroup" aria-label="Tono impuesto">
            {TONES.map((tone) => (
              <button
                key={tone}
                type="button"
                role="radio"
                aria-checked={tone === imposedTone}
                className={tone === imposedTone ? "selected" : undefined}
                onClick={() => setImposedTone(tone)}
                disabled={tone === syllable.tone}
              >
                <ToneShape tone={tone} />
                T{tone}
                <span className="tone-name">{tone === syllable.tone ? "original" : TONE_NAMES[tone]}</span>
              </button>
            ))}
          </div>
          <div className="lab-actions">
            <button type="button" className="secondary" onClick={() => judge(null)} disabled={isJudging}>
              Original
            </button>
            <button type="button" className="primary" onClick={() => judge(imposedTone)} disabled={isJudging || imposedTone === null}>
              {imposedTone === null ? "Elige un tono" : `Cambiar a T${imposedTone} y juzgar`}
            </button>
          </div>
        </>
      )}

      {isJudging && <p className="muted">Resintetizando y consultando a los modelos…</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {judgement && (
        <div className="result">
          <AudioPlayer
            src={judgement.audioUrl}
            autoPlay
            label={judgement.result.imposed_tone === null ? "Frase original" : `Frase con la sílaba en T${judgement.result.imposed_tone}`}
          />
          {judgement.result.judged_contour_st && (
            <ContourChart
              userContour={judgement.result.judged_contour_st}
              targetContour={judgement.result.imposed_tone === null ? null : judgement.result.original_contour_st}
              userLabel="lo que se oye"
              targetLabel="original"
              note="semitonos respecto a la media del hablante"
            />
          )}
          <VerdictTable
            verdicts={judgement.result.verdicts}
            originalTone={judgement.result.original_tone}
            imposedTone={judgement.result.imposed_tone}
          />
          {judgement.result.imposed_tone !== null && (
            <p className="muted small">
              El audio está modificado por ordenador (WORLD): solo cambia la entonación de esa sílaba. El contorno
              impuesto es el de una sílaba real de otro hablante con ese tono.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
