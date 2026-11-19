import { useEffect, useState } from "react";
import { analyze, type SpeakerProfile, type ToneAnalysis } from "../api";
import {
  SYLLABLES,
  TONES,
  TONE_NAMES,
  availableTones,
  findSyllable,
  randomPracticeItem,
  toneEntry,
  type Syllable,
  type Tone,
} from "../syllables";
import { toneAdvice } from "../toneAdvice";
import { ContourChart } from "./ContourChart";
import { CorrectedVoice } from "./CorrectedVoice";
import { ProbabilityBars } from "./ProbabilityBars";
import { ClosestNativeVoice, NativeVoice } from "./NativeVoice";
import { AudioPlayer } from "./AudioPlayer";
import { RecordButton } from "./RecordButton";
import { StepHeading } from "./StepHeading";
import { ToneGuideButton } from "./ToneGuideButton";
import { ToneShape } from "./ToneShape";

interface PracticeStepProps {
  profile: SpeakerProfile;
  onRecalibrate: () => void;
}

interface Attempt {
  analysis: ToneAnalysis;
  recording: Blob;
  audioUrl: string;
  syllable: string;
  targetTone: Tone;
}

export function PracticeStep({ profile, onRecalibrate }: PracticeStepProps) {
  const [syllable, setSyllable] = useState<Syllable>(() => findSyllable("ma"));
  const [targetTone, setTargetTone] = useState<Tone>(1);
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const target = toneEntry(syllable, targetTone);
  const advice =
    attempt && !attempt.analysis.correct && attempt.analysis.target_contour_st
      ? toneAdvice(attempt.analysis.contour_st, attempt.analysis.target_contour_st, attempt.targetTone)
      : null;

  // Libera el audio del intento anterior al sustituirlo o al salir.
  useEffect(() => () => {
    if (attempt) URL.revokeObjectURL(attempt.audioUrl);
  }, [attempt]);

  const selectSyllable = (next: Syllable, preferredTone: Tone = targetTone) => {
    const tones = availableTones(next);
    setSyllable(next);
    setTargetTone(tones.includes(preferredTone) ? preferredTone : tones[0]);
  };

  const pickRandom = () => {
    const item = randomPracticeItem();
    selectSyllable(item.syllable, item.tone);
  };

  const submit = async (recording: Blob) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const analysis = await analyze(recording, profile, targetTone);
      setAttempt({ analysis, recording, audioUrl: URL.createObjectURL(recording), syllable: syllable.syllable, targetTone });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "No se pudo analizar la grabación.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <p className="kicker">Paso 2 de 2</p>
          <h2>Practica</h2>
        </div>
        <button type="button" className="link" onClick={onRecalibrate}>
          Volver a calibrar
        </button>
      </div>

      <StepHeading number={1}>Elige la sílaba y el tono</StepHeading>
      <div className="field">
        <label htmlFor="syllable-select">Sílaba</label>
        <select
          id="syllable-select"
          value={syllable.syllable}
          onChange={(event) => selectSyllable(findSyllable(event.target.value))}
        >
          {SYLLABLES.map((option) => (
            <option key={option.syllable} value={option.syllable}>
              {option.syllable}
            </option>
          ))}
        </select>
        <button type="button" className="secondary" onClick={pickRandom}>
          Aleatoria
        </button>
      </div>

      <div className="tone-picker" role="radiogroup" aria-label="Tono objetivo">
        {TONES.map((tone) => {
          const entry = toneEntry(syllable, tone);
          return (
            <button
              key={tone}
              type="button"
              role="radio"
              aria-checked={tone === targetTone}
              className={tone === targetTone ? "selected" : undefined}
              onClick={() => setTargetTone(tone)}
              disabled={!entry}
            >
              <ToneShape tone={tone} />
              <span className="syllable">
                {entry ? entry.pinyin : "—"}
                {entry && entry.characters.length > 0 && <span className="characters">{entry.characters.join(" ")}</span>}
              </span>
              <span className="tone-name">
                T{tone} · {entry ? TONE_NAMES[tone] : "no existe como palabra"}
              </span>
            </button>
          );
        })}
      </div>

      <StepHeading number={2}>Escucha cómo suena</StepHeading>
      <div className="listen-panel">
        <ToneGuideButton tone={targetTone} referenceHz={profile.reference_hz} />
        {/* key: al cambiar de sílaba o tono no sigue sonando la grabación anterior */}
        <NativeVoice
          key={`${syllable.syllable}${targetTone}`}
          syllable={syllable.syllable}
          tone={targetTone}
          referenceHz={profile.reference_hz}
        />
      </div>

      <StepHeading number={3}>Grábate</StepHeading>
      <div className="record-area">
        <RecordButton
          label={`Grabar ${target?.pinyin ?? ""}`}
          onRecorded={submit}
          onError={setError}
          disabled={isAnalyzing}
          isProminent
        />
        <p className="muted small">Pulsa, di la sílaba y vuelve a pulsar. Se para sola a los 3 segundos.</p>
      </div>
      {isAnalyzing && <p className="muted">Analizando…</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {attempt && (
        <div className="result">
          <div className={`verdict ${attempt.analysis.correct ? "ok" : "ko"}`} role="status">
            <span className="verdict-label">{attempt.analysis.correct ? "Correcto" : "Por corregir"}</span>
            <p className="verdict-headline">
              {attempt.analysis.correct
                ? `Se reconoce el tono ${attempt.targetTone}.`
                : `Se oye el tono ${attempt.analysis.tone}, no el ${attempt.targetTone}.`}
            </p>
            {advice && <p className="advice">{advice}</p>}
          </div>
          <div className="result-grid">
            <ContourChart userContour={attempt.analysis.contour_st} targetContour={attempt.analysis.target_contour_st} />
            <div className="compare">
              <p className="label">Compara</p>
              <AudioPlayer src={attempt.audioUrl} label="Tu grabación" />
              <ClosestNativeVoice syllable={attempt.syllable} tone={attempt.targetTone} referenceHz={profile.reference_hz} />
              {/* Si el tono ya es correcto, la voz corregida sonaría igual que la grabación.
                  key: un intento nuevo empieza sin la voz corregida del anterior. */}
              {!attempt.analysis.correct && (
                <CorrectedVoice
                  key={attempt.audioUrl}
                  recording={attempt.recording}
                  profile={profile}
                  targetTone={attempt.targetTone}
                />
              )}
            </div>
          </div>
          <p className="label">Qué tono oye el modelo</p>
          <ProbabilityBars probabilities={attempt.analysis.probabilities} targetTone={attempt.targetTone} />
        </div>
      )}
    </section>
  );
}
