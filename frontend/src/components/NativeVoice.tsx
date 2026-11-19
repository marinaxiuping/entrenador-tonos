import { useState } from "react";
import voicesData from "../data/reference_voices.json";

// Generado por scripts/07_exportar_voces_referencia.py: altura de voz de cada hablante de
// Tone Perfect, medida igual que el perfil del usuario.
interface ReferenceVoice {
  speaker: string;
  gender: "female" | "male";
  reference_hz: number;
}

const VOICES = voicesData as ReferenceVoice[];
const GENDER_NAMES = { female: "mujer", male: "hombre" } as const;

interface NativeVoiceProps {
  syllable: string;
  tone: number;
  referenceHz: number;
}

/** Voces nativas de la más parecida a la del usuario a la menos, por distancia en semitonos. */
function voicesByCloseness(referenceHz: number): ReferenceVoice[] {
  const distance = (voice: ReferenceVoice) => Math.abs(Math.log2(voice.reference_hz / referenceHz));
  return [...VOICES].sort((a, b) => distance(a) - distance(b));
}

/** Reproduce la sílaba dicha por un hablante nativo de Tone Perfect, para imitarla. */
export function NativeVoice({ syllable, tone, referenceHz }: NativeVoiceProps) {
  const voices = voicesByCloseness(referenceHz);
  const [voiceIndex, setVoiceIndex] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const voice = voices[voiceIndex];
  const source = `/api/reference-audio/${encodeURIComponent(syllable)}/${tone}/${voice.speaker}`;

  const listen = () => {
    setError(null);
    setIsListening(true);
  };

  const nextVoice = () => {
    setError(null);
    setVoiceIndex((voiceIndex + 1) % voices.length);
    setIsListening(true);
  };

  return (
    <div className="native-voice">
      <div className="native-voice-actions">
        <button type="button" className="secondary" onClick={listen}>
          Escuchar a un nativo
        </button>
        {isListening && (
          <button type="button" className="secondary" onClick={nextVoice}>
            Otra voz
          </button>
        )}
      </div>
      {isListening && !error && (
        <>
          <audio key={source} controls autoPlay src={source}
                 onError={() => setError("No hay grabación de esta voz para esta sílaba. Prueba otra voz.")} />
          <p className="muted small">
            Voz {voice.speaker} ({GENDER_NAMES[voice.gender]}){voiceIndex === 0 && ", la más parecida a la tuya"}.
          </p>
        </>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      <p className="muted small">
        Grabaciones de <a href="https://listings.lib.msu.edu/toneperfect/">Tone Perfect</a> (MSU Libraries), licencia{" "}
        <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a>.
      </p>
    </div>
  );
}
