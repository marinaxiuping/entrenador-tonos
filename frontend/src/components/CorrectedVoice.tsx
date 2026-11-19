import { useEffect, useState } from "react";
import { resynthesize, type SpeakerProfile } from "../api";
import { AudioPlayer } from "./AudioPlayer";

interface CorrectedVoiceProps {
  recording: Blob;
  profile: SpeakerProfile;
  targetTone: number;
}

/** Botón para escuchar la propia grabación con el contorno del tono objetivo. */
export function CorrectedVoice({ recording, profile, targetTone }: CorrectedVoiceProps) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  const load = async () => {
    setIsLoading(true);
    setError(null);
    try {
      setAudioUrl(URL.createObjectURL(await resynthesize(recording, profile, targetTone)));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "No se pudo generar la voz corregida.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="corrected-voice">
      {audioUrl ? (
        <AudioPlayer src={audioUrl} autoPlay label={`Tu voz corregida al tono ${targetTone}`} />
      ) : (
        <button type="button" className="secondary" onClick={load} disabled={isLoading}>
          {isLoading ? "Generando…" : `Escuchar tu voz con el tono ${targetTone}`}
        </button>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      <p className="muted small">Es tu grabación con la entonación cambiada por ordenador; puede sonar algo artificial.</p>
    </div>
  );
}
