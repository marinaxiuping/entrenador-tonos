import { useState } from "react";
import type { SpeakerProfile } from "../api";
import { toneGuide } from "../audio/toneGuide";
import { TONES, TONE_NAMES, randomPracticeItem, toneEntry } from "../syllables";
import { ROUNDS_PER_GAME, TIME_STRETCH, axisFor, describeScore } from "../toneGame";
import { GameTrack } from "./GameTrack";
import { ToneGuideButton } from "./ToneGuideButton";
import { ToneShape } from "./ToneShape";

interface GameStepProps {
  profile: SpeakerProfile;
  onRecalibrate: () => void;
}

const AXIS = axisFor(TONES.map((tone) => toneGuide(tone).contour_st));

const newRounds = () => Array.from({ length: ROUNDS_PER_GAME }, randomPracticeItem);
const asPercent = (score: number) => `${Math.round(score * 100)} %`;

export function GameStep({ profile, onRecalibrate }: GameStepProps) {
  const [rounds, setRounds] = useState(newRounds);
  const [roundIndex, setRoundIndex] = useState(0);
  const [scores, setScores] = useState<number[]>([]);
  // Cambia en cada intento para que la pista empiece de cero al repetir la ronda.
  const [attempt, setAttempt] = useState(0);
  // El micrófono no cancela el eco: la guía sonando arrancaría el recorrido y puntuaría sola.
  const [isListening, setIsListening] = useState(false);

  const { syllable, tone } = rounds[roundIndex];
  const entry = toneEntry(syllable, tone);
  const guide = toneGuide(tone);
  const roundScore = scores[roundIndex];
  const isScored = roundScore !== undefined;
  const isLastRound = roundIndex === ROUNDS_PER_GAME - 1;
  const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;

  const retryRound = () => {
    setScores(scores.slice(0, roundIndex));
    setAttempt(attempt + 1);
  };

  const nextRound = () => {
    setRoundIndex(roundIndex + 1);
    setAttempt(attempt + 1);
  };

  const restart = () => {
    setRounds(newRounds());
    setRoundIndex(0);
    setScores([]);
    setAttempt(attempt + 1);
  };

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <p className="kicker">Juego</p>
          <h2>Sigue la curva</h2>
        </div>
        <button type="button" className="link" onClick={onRecalibrate}>
          Volver a calibrar
        </button>
      </div>
      <p>
        Canta la sílaba alargándola y lleva al personaje por la curva del tono. El recorrido va más lento que el
        habla para que te dé tiempo a corregirte. La puntuación la calcula el navegador comparando tu voz con la
        curva, no el modelo de la práctica.
      </p>

      <div className="game-round">
        <p className="label">
          Ronda {roundIndex + 1} de {ROUNDS_PER_GAME}
        </p>
        <div className="game-target">
          <ToneShape tone={tone} />
          <span className="syllable">
            {entry?.pinyin}
            {entry && entry.characters.length > 0 && <span className="characters">{entry.characters.join(" ")}</span>}
          </span>
          <span className="tone-name">T{tone} · {TONE_NAMES[tone]}</span>
        </div>
        <ToneGuideButton tone={tone} referenceHz={profile.reference_hz} disabled={isListening} />
      </div>

      <GameTrack
        key={attempt}
        contour={guide.contour_st}
        durationS={guide.duration_s * TIME_STRETCH}
        axis={AXIS}
        profile={profile}
        onFinished={(score) => setScores([...scores.slice(0, roundIndex), score])}
        onListeningChange={setIsListening}
      />

      {isScored && (
        <div className="game-score" role="status">
          <p className="game-score-value">{asPercent(roundScore)}</p>
          <p>{describeScore(roundScore)}</p>
          <div className="game-actions">
            <button type="button" className="secondary" onClick={retryRound}>
              Repetir ronda
            </button>
            {!isLastRound && (
              <button type="button" className="primary" onClick={nextRound}>
                Siguiente ronda
              </button>
            )}
          </div>
        </div>
      )}

      {isScored && isLastRound && (
        <div className="game-summary">
          <p className="label">Partida terminada</p>
          <p className="game-score-value">{asPercent(average)}</p>
          <p className="muted">Media de las {ROUNDS_PER_GAME} rondas.</p>
          <button type="button" className="primary" onClick={restart}>
            Jugar otra vez
          </button>
        </div>
      )}
    </section>
  );
}
