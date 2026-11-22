import { TONE_NAMES, toneEntry } from "../syllables";
import { accuracyByTone, type RoomRecord } from "../toneRooms";
import { ToneShape } from "./ToneShape";

interface GameOverProps {
  history: RoomRecord[];
  roomsPassed: number;
  record: number;
  isNewRecord: boolean;
  onRestart: () => void;
}

const OUTCOME_LABELS: Record<RoomRecord["outcome"], string> = {
  target: "Superada",
  decoy: "Trampa",
  none: "Sin vidas",
};

/** Fin de la partida: hasta dónde se llegó, el tono más flojo y el recorrido habitación a habitación. */
export function GameOver({ history, roomsPassed, record, isNewRecord, onRestart }: GameOverProps) {
  const accuracies = accuracyByTone(history);
  const weakest = accuracies.length > 1 && accuracies[0].hits < accuracies[0].attempts ? accuracies[0] : null;

  return (
    <div className="game-over">
      <div className="game-over-hero">
        <p className="label">Sin vidas</p>
        <p className="game-over-rooms">{roomsPassed}</p>
        <p className="game-over-headline">
          {roomsPassed === 1 ? "habitación superada" : "habitaciones superadas"}
        </p>
        <p className={isNewRecord ? "game-over-record is-new" : "game-over-record"}>
          {isNewRecord ? "¡Nuevo récord!" : `Tu récord: ${record}`}
        </p>
        {weakest && (
          <p className="muted">
            Tu tono más flojo ha sido el {weakest.tone} ({TONE_NAMES[weakest.tone]}): el modelo lo oyó en{" "}
            {weakest.hits} de {weakest.attempts} intentos.
          </p>
        )}
      </div>

      <p className="muted small">Cada habitación, con los tonos que oyó el modelo en cada intento; en verde, el que se pedía.</p>
      <ol className="game-over-rooms-list">
        {history.map(({ room, heardTones, outcome }, index) => {
          const entry = toneEntry(room.syllable, room.target);
          return (
            <li key={index} className={outcome === "target" ? "is-passed" : "is-trapped"}>
              <span className="game-over-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="game-over-item">
                <ToneShape tone={room.target} />
                <span className="syllable">{entry?.pinyin}</span>
                <span className="characters">{entry?.characters[0]}</span>
              </span>
              <span className="game-over-attempts" aria-label="Tonos que oyó el modelo en cada intento">
                {heardTones.map((heard, attempt) => (
                  <span key={attempt} className={heard === room.target ? "heard is-right" : "heard"}>
                    T{heard}
                  </span>
                ))}
              </span>
              <span className="game-over-outcome">{OUTCOME_LABELS[outcome]}</span>
            </li>
          );
        })}
      </ol>

      <button type="button" className="primary" onClick={onRestart}>
        Jugar otra vez
      </button>
    </div>
  );
}
