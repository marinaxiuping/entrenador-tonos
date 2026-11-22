import { useEffect, useRef, useState } from "react";
import { analyze, type SpeakerProfile } from "../api";
import { loadRecord, saveRecord } from "../recordStorage";
import { TONE_NAMES, type Tone } from "../syllables";
import { STARTING_LIVES, createRoom, doorFor, type Room, type RoomRecord } from "../toneRooms";
import { GameOver } from "./GameOver";
import { Lantern } from "./Lantern";
import { NativeVoice } from "./NativeVoice";
import { RecordButton } from "./RecordButton";
import { RoomScene, type DoorState, type Verdict } from "./RoomScene";
import { ToneGuideButton } from "./ToneGuideButton";

interface GameStepProps {
  profile: SpeakerProfile;
  onRecalibrate: () => void;
}

type GamePhase = "playing" | "analyzing" | "opening" | "over";

interface Feedback {
  kind: "ok" | "ko" | "info";
  text: string;
}

// Tiempo para ver la puerta abrirse y cruzarla antes de pasar a la siguiente habitación.
const DOOR_ANIMATION_MS = 1900;

export function GameStep({ profile, onRecalibrate }: GameStepProps) {
  const [room, setRoom] = useState<Room>(() => createRoom());
  // Cuenta todas las habitaciones visitadas, también las de la trampa: identifica la escena.
  const [roomsVisited, setRoomsVisited] = useState(1);
  const [roomsPassed, setRoomsPassed] = useState(0);
  const [lives, setLives] = useState(STARTING_LIVES);
  const [heardTones, setHeardTones] = useState<number[]>([]);
  const [history, setHistory] = useState<RoomRecord[]>([]);
  const [phase, setPhase] = useState<GamePhase>("playing");
  const [openedDoor, setOpenedDoor] = useState<{ side: Room["targetSide"]; state: Exclude<DoorState, "closed"> } | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [record, setRecord] = useState(loadRecord);
  const [isNewRecord, setIsNewRecord] = useState(false);
  const transitionRef = useRef<number | undefined>(undefined);
  const verdict: Verdict | null = feedback?.kind === "ok" ? "right" : feedback?.kind === "ko" ? "wrong" : null;

  useEffect(() => () => window.clearTimeout(transitionRef.current), []);

  const enterNextRoom = () => {
    setRoom(createRoom());
    setRoomsVisited((count) => count + 1);
    setHeardTones([]);
    setOpenedDoor(null);
    setFeedback(null);
    setPhase("playing");
  };

  const finishGame = (finalRoomsPassed: number) => {
    if (finalRoomsPassed > record) {
      saveRecord(finalRoomsPassed);
      setRecord(finalRoomsPassed);
      setIsNewRecord(true);
    }
    setPhase("over");
  };

  const leaveRoom = (outcome: RoomRecord["outcome"], tonesHeard: number[]) => {
    setHistory((current) => [...current, { room, heardTones: tonesHeard, outcome }]);
    const passed = roomsPassed + (outcome === "target" ? 1 : 0);
    const livesLeft = lives - (outcome === "decoy" ? 1 : 0);
    // La vida se pierde a la vista; el contador de habitaciones sube al cruzar la puerta.
    setLives(livesLeft);
    transitionRef.current = window.setTimeout(() => {
      setRoomsPassed(passed);
      if (livesLeft === 0) finishGame(passed);
      else enterNextRoom();
    }, DOOR_ANIMATION_MS);
  };

  const submit = async (recording: Blob) => {
    setPhase("analyzing");
    setFeedback(null);
    let heard: number;
    try {
      heard = (await analyze(recording, profile, room.target)).tone;
    } catch (failure) {
      setFeedback({ kind: "info", text: failure instanceof Error ? failure.message : "No se pudo analizar la grabación." });
      setPhase("playing");
      return;
    }
    const tonesHeard = [...heardTones, heard];
    setHeardTones(tonesHeard);
    const outcome = doorFor(room, heard);
    if (outcome === "none") {
      // Un tono que no está en ninguna puerta también es un error: cuesta una vida, pero no hay
      // puerta que cruzar y se sigue en la misma habitación.
      const livesLeft = lives - 1;
      setLives(livesLeft);
      setFeedback({
        kind: "ko",
        text: `Se ha oído el tono ${heard} (${TONE_NAMES[heard as Tone]}), que no abre ninguna puerta. Pierdes una vida.`,
      });
      if (livesLeft > 0) {
        setPhase("playing");
        return;
      }
      setPhase("opening");
      setHistory((current) => [...current, { room, heardTones: tonesHeard, outcome }]);
      transitionRef.current = window.setTimeout(() => finishGame(roomsPassed), DOOR_ANIMATION_MS);
      return;
    }
    const decoySide = room.targetSide === "left" ? "right" : "left";
    setOpenedDoor(outcome === "target" ? { side: room.targetSide, state: "open" } : { side: decoySide, state: "trap" });
    setFeedback(
      outcome === "target"
        ? { kind: "ok", text: "¡Bien dicho! La puerta se abre." }
        : { kind: "ko", text: `Se ha oído el tono ${heard}: era la puerta trampa. Pierdes una vida.` },
    );
    setPhase("opening");
    leaveRoom(outcome, tonesHeard);
  };

  const restart = () => {
    setRoomsPassed(0);
    setLives(STARTING_LIVES);
    setHistory([]);
    setIsNewRecord(false);
    enterNextRoom();
  };

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <p className="kicker">Juego</p>
          <h2>Las habitaciones</h2>
        </div>
        <button type="button" className="link" onClick={onRecalibrate}>
          Volver a calibrar
        </button>
      </div>

      {phase === "over" ? (
        <GameOver history={history} roomsPassed={roomsPassed} record={record} isNewRecord={isNewRecord} onRestart={restart} />
      ) : (
        <>
          <p>
            Cada habitación tiene dos puertas, cada una con un tono. Di la sílaba del cartel: si el modelo de la práctica
            oye su tono, se abre la puerta buena; si oye el de la otra, caes en la trampa y pierdes una vida, y
            cualquier otro tono también te cuesta una. Llega tan lejos como puedas.
          </p>

          <div className="room-hud">
            <span className="room-hud-depth">Habitación {roomsPassed + 1}</span>
            <span className="room-hud-lives" aria-label={`${lives} vidas de ${STARTING_LIVES}`}>
              {Array.from({ length: STARTING_LIVES }, (_, i) => (
                <Lantern key={i} isLit={i < lives} />
              ))}
            </span>
            <span className="room-hud-record">Récord: {record}</span>
          </div>

          {/* key: cada habitación nueva entra con su animación y con las puertas cerradas */}
          <RoomScene key={roomsVisited} room={room} openedDoor={openedDoor} verdict={verdict} attempt={heardTones.length} />

          <div className="room-controls">
            <RecordButton
              label="Pronunciar"
              onRecorded={submit}
              onError={(text) => setFeedback({ kind: "info", text })}
              disabled={phase !== "playing"}
              isProminent
            />
            {phase === "analyzing" && <p className="muted">Escuchando…</p>}
            {feedback && (
              <p className={`room-feedback is-${feedback.kind}`} role="status">
                {feedback.text}
              </p>
            )}
          </div>

          <details className="room-help">
            <summary>¿Cómo suena?</summary>
            <div className="listen-panel">
              <ToneGuideButton tone={room.target} referenceHz={profile.reference_hz} disabled={phase !== "playing"} />
              <NativeVoice
                key={`${room.syllable.syllable}${room.target}`}
                syllable={room.syllable.syllable}
                tone={room.target}
                referenceHz={profile.reference_hz}
              />
            </div>
          </details>
        </>
      )}
    </section>
  );
}
