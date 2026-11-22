import { TONE_NAMES, toneEntry, type Tone } from "../syllables";
import type { DoorSide, Room } from "../toneRooms";
import { ToneShape } from "./ToneShape";

export type DoorState = "closed" | "open" | "trap";

interface RoomSceneProps {
  room: Room;
  /** Puerta abierta, si hay alguna, y cómo. */
  openedDoor: { side: DoorSide; state: Exclude<DoorState, "closed"> } | null;
}

/** La habitación: el cartel con la sílaba que hay que decir y las dos puertas con su tono. */
export function RoomScene({ room, openedDoor }: RoomSceneProps) {
  const entry = toneEntry(room.syllable, room.target);
  const decoySide: DoorSide = room.targetSide === "left" ? "right" : "left";
  const toneAt = (side: DoorSide): Tone => (side === room.targetSide ? room.target : room.decoy);
  const stateAt = (side: DoorSide): DoorState => (openedDoor?.side === side ? openedDoor.state : "closed");

  return (
    <div className="room-scene">
      <div className="room-sign" aria-label={`Pronuncia ${entry?.pinyin}`}>
        <span className="room-sign-label">Di</span>
        <span className="room-sign-hanzi">{entry?.characters[0]}</span>
        <span className="room-sign-pinyin">{entry?.pinyin}</span>
      </div>
      <div className="room-doors">
        {(["left", "right"] as const).map((side) => (
          <Door key={side} tone={toneAt(side)} state={stateAt(side)} isDecoy={side === decoySide} />
        ))}
      </div>
      <div className="room-floor" aria-hidden="true" />
    </div>
  );
}

function Door({ tone, state, isDecoy }: { tone: Tone; state: DoorState; isDecoy: boolean }) {
  const description = state === "closed" ? "cerrada" : isDecoy ? "abierta: era la trampa" : "abierta";
  return (
    <div className={`door is-${state}`} role="group" aria-label={`Puerta del tono ${tone}, ${description}`}>
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
        <div className="door-leaf" aria-hidden="true">
          <span className="door-knob" />
        </div>
      </div>
    </div>
  );
}
