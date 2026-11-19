import { useState } from "react";
import voicesData from "../data/reference_voices.json";
import { AudioPlayer } from "./AudioPlayer";

// Generado por scripts/07_exportar_voces_referencia.py: altura de voz de cada hablante de
// Tone Perfect, medida igual que el perfil del usuario.
interface ReferenceVoice {
  speaker: string;
  gender: "female" | "male";
  reference_hz: number;
}

const VOICES = voicesData as ReferenceVoice[];
export const GENDER_NAMES = { female: "mujer", male: "hombre" } as const;

interface NativeVoiceProps {
  syllable: string;
  tone: number;
  referenceHz: number;
}

/** Voces nativas de la más parecida a la del usuario a la menos, por distancia en semitonos. */
export function voicesByCloseness(referenceHz: number): ReferenceVoice[] {
  const distance = (voice: ReferenceVoice) => Math.abs(Math.log2(voice.reference_hz / referenceHz));
  return [...VOICES].sort((a, b) => distance(a) - distance(b));
}

export function referenceAudioUrl(syllable: string, tone: number, speaker: string): string {
  return `/api/reference-audio/${encodeURIComponent(syllable)}/${tone}/${speaker}`;
}

/** Reproduce la sílaba dicha por un hablante nativo de Tone Perfect, para imitarla. */
export function NativeVoice({ syllable, tone, referenceHz }: NativeVoiceProps) {
  const voices = voicesByCloseness(referenceHz);
  const [voiceIndex, setVoiceIndex] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const voice = voices[voiceIndex];
  const source = referenceAudioUrl(syllable, tone, voice.speaker);

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
          <AudioPlayer
            key={source}
            src={source}
            autoPlay
            label={`Voz ${voice.speaker} (${GENDER_NAMES[voice.gender]})${voiceIndex === 0 ? ", la más parecida a la tuya" : ""}`}
            onError={() => setError("No hay grabación de esta voz para esta sílaba. Prueba otra voz.")}
          />
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

/** La voz nativa más parecida, para comparar con el intento; si falta su grabación, pasa a la siguiente. */
export function ClosestNativeVoice({ syllable, tone, referenceHz }: NativeVoiceProps) {
  const voices = voicesByCloseness(referenceHz);
  const [voiceIndex, setVoiceIndex] = useState(0);
  if (voiceIndex >= voices.length) return null;
  const voice = voices[voiceIndex];
  const source = referenceAudioUrl(syllable, tone, voice.speaker);
  return (
    <AudioPlayer
      key={source}
      src={source}
      label={`Nativo · voz ${voice.speaker}`}
      onError={() => setVoiceIndex(voiceIndex + 1)}
    />
  );
}
