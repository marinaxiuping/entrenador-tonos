import { useState } from "react";
import { calibrate, type SpeakerProfile } from "../api";
import { CALIBRATION_SYLLABLE, TONES, TONE_NAMES, findSyllable, toneEntry } from "../syllables";
import { RecordButton } from "./RecordButton";
import { ToneShape } from "./ToneShape";

const calibrationSyllable = findSyllable(CALIBRATION_SYLLABLE);

interface CalibrationStepProps {
  onCalibrated: (profile: SpeakerProfile) => void;
}

export function CalibrationStep({ onCalibrated }: CalibrationStepProps) {
  const [recordings, setRecordings] = useState<(Blob | null)[]>(TONES.map(() => null));
  const [error, setError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const isComplete = recordings.every((recording) => recording !== null);

  const storeRecording = (index: number, recording: Blob) => {
    setError(null);
    setRecordings((current) => current.map((existing, i) => (i === index ? recording : existing)));
  };

  const submit = async () => {
    setIsSending(true);
    setError(null);
    try {
      onCalibrated(await calibrate(recordings as Blob[]));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "No se pudo calibrar.");
    } finally {
      setIsSending(false);
    }
  };

  return (
    <section className="card">
      <p className="kicker">Paso 1 de 2</p>
      <h2>Calibra tu voz</h2>
      <p>
        Para comparar tu entonación con la de un nativo hace falta conocer tu tono medio. Graba la sílaba{" "}
        <strong>{CALIBRATION_SYLLABLE}</strong> en los cuatro tonos, cada una por separado y con tu voz natural.
      </p>
      <ol className="calibration-list">
        {TONES.map((tone, i) => (
          <li key={tone} className={recordings[i] ? "is-done" : undefined}>
            <ToneShape tone={tone} />
            <span className="calibration-tone">
              <span className="syllable">{toneEntry(calibrationSyllable, tone)?.pinyin}</span>
              <span className="tone-name">T{tone} · {TONE_NAMES[tone]}</span>
            </span>
            <RecordButton
              label={recordings[i] ? "Repetir" : "Grabar"}
              onRecorded={(recording) => storeRecording(i, recording)}
              onError={setError}
            />
            <span className="done" aria-label={recordings[i] ? "grabada" : undefined}>{recordings[i] ? "✓" : ""}</span>
          </li>
        ))}
      </ol>
      {error && <p className="error" role="alert">{error}</p>}
      <button type="button" className="primary" onClick={submit} disabled={!isComplete || isSending}>
        {isSending ? "Calculando…" : "Calcular mi perfil"}
      </button>
    </section>
  );
}
