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
import { ContourChart } from "./ContourChart";
import { CorrectedVoice } from "./CorrectedVoice";
import { ProbabilityBars } from "./ProbabilityBars";
import { RecordButton } from "./RecordButton";
import { NativeVoice } from "./NativeVoice";
import { ToneGuideButton } from "./ToneGuideButton";

interface PracticeStepProps {
  profile: SpeakerProfile;
  onRecalibrate: () => void;
}

interface Attempt {
  analysis: ToneAnalysis;
  recording: Blob;
  audioUrl: string;
  targetTone: Tone;
}

export function PracticeStep({ profile, onRecalibrate }: PracticeStepProps) {
  const [syllable, setSyllable] = useState<Syllable>(() => findSyllable("ma"));
  const [targetTone, setTargetTone] = useState<Tone>(1);
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const target = toneEntry(syllable, targetTone);

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
      setAttempt({ analysis, recording, audioUrl: URL.createObjectURL(recording), targetTone });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "No se pudo analizar la grabación.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="card">
      <div className="card-header">
        <h2>2. Practica</h2>
        <button type="button" className="link" onClick={onRecalibrate}>
          Volver a calibrar
        </button>
      </div>

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

      <ToneGuideButton tone={targetTone} referenceHz={profile.reference_hz} />
      {/* key: al cambiar de sílaba o tono no sigue sonando la grabación anterior */}
      <NativeVoice
        key={`${syllable.syllable}${targetTone}`}
        syllable={syllable.syllable}
        tone={targetTone}
        referenceHz={profile.reference_hz}
      />

      <RecordButton
        label={`Grabar ${target?.pinyin ?? ""}`}
        onRecorded={submit}
        onError={setError}
        disabled={isAnalyzing}
      />
      {isAnalyzing && <p className="muted">Analizando…</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {attempt && (
        <div className="result">
          <p className={`verdict ${attempt.analysis.correct ? "ok" : "ko"}`}>
            {attempt.analysis.correct
              ? `¡Bien! Se reconoce el tono ${attempt.targetTone}.`
              : `Se reconoce el tono ${attempt.analysis.tone}, no el ${attempt.targetTone}.`}
          </p>
          <ContourChart userContour={attempt.analysis.contour_st} targetContour={attempt.analysis.target_contour_st} />
          <ProbabilityBars probabilities={attempt.analysis.probabilities} targetTone={attempt.targetTone} />
          <h3>Tu grabación</h3>
          <audio controls src={attempt.audioUrl} />
          {/* key: un intento nuevo empieza sin la voz corregida del anterior */}
          <CorrectedVoice
            key={attempt.audioUrl}
            recording={attempt.recording}
            profile={profile}
            targetTone={attempt.targetTone}
          />
        </div>
      )}
    </section>
  );
}
