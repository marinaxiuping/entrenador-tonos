import { TONES, randomPracticeItem, type Syllable, type Tone } from "./syllables";

export const STARTING_LIVES = 3;

export type DoorSide = "left" | "right";

export interface Room {
  syllable: Syllable;
  target: Tone;
  /** Tono de la puerta trampa. */
  decoy: Tone;
  targetSide: DoorSide;
}

/** Qué abre el tono que ha oído el modelo: la puerta buena, la trampa o ninguna. */
export type DoorOutcome = "target" | "decoy" | "none";

export interface RoomRecord {
  room: Room;
  /** Tonos que oyó el modelo en cada intento, en orden. */
  heardTones: number[];
  /** Cómo se salió: por la puerta buena, por la trampa o, con «none», sin vidas y sin abrir ninguna. */
  outcome: DoorOutcome;
}

export function createRoom(): Room {
  const { syllable, tone } = randomPracticeItem();
  const decoys = TONES.filter((candidate) => candidate !== tone);
  return {
    syllable,
    target: tone,
    decoy: decoys[Math.floor(Math.random() * decoys.length)],
    targetSide: Math.random() < 0.5 ? "left" : "right",
  };
}

export function doorFor(room: Room, heardTone: number): DoorOutcome {
  if (heardTone === room.target) return "target";
  if (heardTone === room.decoy) return "decoy";
  return "none";
}

export interface ToneAccuracy {
  tone: Tone;
  attempts: number;
  hits: number;
}

/** Aciertos por tono pedido sobre todos los intentos, del peor al mejor. */
export function accuracyByTone(records: RoomRecord[]): ToneAccuracy[] {
  const byTone = new Map<Tone, ToneAccuracy>();
  for (const { room, heardTones } of records) {
    const accuracy = byTone.get(room.target) ?? { tone: room.target, attempts: 0, hits: 0 };
    accuracy.attempts += heardTones.length;
    accuracy.hits += heardTones.filter((heard) => heard === room.target).length;
    byTone.set(room.target, accuracy);
  }
  return [...byTone.values()].sort((a, b) => a.hits / a.attempts - b.hits / b.attempts);
}
