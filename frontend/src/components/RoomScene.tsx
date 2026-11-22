import { TONE_NAMES, toneEntry, type Tone } from "../syllables";
import type { DoorSide, Room } from "../toneRooms";
import { Lantern } from "./Lantern";
import { ToneShape } from "./ToneShape";

export type DoorState = "closed" | "open" | "trap";
export type Verdict = "right" | "wrong";

interface RoomSceneProps {
  room: Room;
  /** Puerta abierta, si hay alguna, y cómo. */
  openedDoor: { side: DoorSide; state: Exclude<DoorState, "closed"> } | null;
  /** Sello del último intento: 对 si fue el tono pedido, 错 si no. */
  verdict: Verdict | null;
  /** Cambia en cada intento para que el sello vuelva a estamparse. */
  attempt: number;
}

const SEALS: Record<Verdict, { hanzi: string; label: string }> = {
  right: { hanzi: "对", label: "correcto" },
  wrong: { hanzi: "错", label: "incorrecto" },
};

/** La habitación: el cartel con la sílaba que hay que decir y las dos puertas con su tono. */
export function RoomScene({ room, openedDoor, verdict, attempt }: RoomSceneProps) {
  const entry = toneEntry(room.syllable, room.target);
  const decoySide: DoorSide = room.targetSide === "left" ? "right" : "left";
  const toneAt = (side: DoorSide): Tone => (side === room.targetSide ? room.target : room.decoy);
  const stateAt = (side: DoorSide): DoorState => (openedDoor?.side === side ? openedDoor.state : "closed");
  // Al abrir la puerta buena, la cámara entra por ella.
  const crossing = openedDoor?.state === "open" ? ` is-crossing-${openedDoor.side}` : "";

  return (
    // El marco no se mueve: recorta el zoom de la escena al cruzar la puerta.
    <div className="room-viewport">
      <div className={`room-scene${crossing}`}>
        <Lantern className="room-lantern is-left" />
        <Lantern className="room-lantern is-right" />
        <div className="room-sign" aria-label={`Pronuncia ${entry?.pinyin}`}>
          <span className="room-sign-label">Di · 请说</span>
          <span className="room-sign-hanzi">{entry?.characters[0]}</span>
          <span className="room-sign-pinyin">{entry?.pinyin}</span>
        </div>
        <div className="room-doors">
          {(["left", "right"] as const).map((side) => (
            <Door key={side} tone={toneAt(side)} state={stateAt(side)} isDecoy={side === decoySide} />
          ))}
        </div>
        <div className="room-floor" aria-hidden="true" />
        {verdict && (
          <span key={attempt} className={`room-seal is-${verdict}`} role="img" aria-label={SEALS[verdict].label}>
            {SEALS[verdict].hanzi}
          </span>
        )}
      </div>
    </div>
  );
}

function Door({ tone, state, isDecoy }: { tone: Tone; state: DoorState; isDecoy: boolean }) {
  const description = state === "closed" ? "cerrada" : isDecoy ? "abierta: era la trampa" : "abierta";
  return (
    <div className={`door is-${state}`} role="group" aria-label={`Puerta del tono ${tone}, ${description}`}>
      <div className="door-eave" aria-hidden="true" />
      <div className="door-plaque">
        <ToneShape tone={tone} />
        <span>
          T{tone}
          <span className="door-plaque-name"> · {TONE_NAMES[tone]}</span>
        </span>
      </div>
      <div className="door-frame">
        <div className="door-beyond" aria-hidden="true">
          {state === "trap" && <span className="door-trap">✕</span>}
        </div>
        <div className="door-leaf is-left" aria-hidden="true">
          <span className="door-ring" />
        </div>
        <div className="door-leaf is-right" aria-hidden="true">
          <span className="door-ring" />
        </div>
      </div>
    </div>
  );
}
