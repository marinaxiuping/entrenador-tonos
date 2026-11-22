import { useState } from "react";
import { playToneGuide } from "../audio/toneGuide";

interface ToneGuideButtonProps {
  tone: number;
  referenceHz: number;
  disabled?: boolean;
}

/** Reproduce la melodía del tono objetivo a la altura de la voz del usuario, para imitarla. */
export function ToneGuideButton({ tone, referenceHz, disabled = false }: ToneGuideButtonProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const play = async () => {
    setIsPlaying(true);
    setError(null);
    try {
      await playToneGuide(tone, referenceHz);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "No se pudo reproducir la guía.");
    } finally {
      setIsPlaying(false);
    }
  };

  return (
    <div className="tone-guide">
      <button type="button" className="secondary" onClick={play} disabled={isPlaying || disabled}>
        {isPlaying ? "Sonando…" : `Escuchar la melodía del tono ${tone}`}
      </button>
      <p className="muted small">Un tarareo con la forma del tono, a la altura de tu voz. No es la pronunciación de la sílaba.</p>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
