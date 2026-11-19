import syllablesData from "./data/syllables.json";

export const TONES = [1, 2, 3, 4] as const;
export type Tone = (typeof TONES)[number];

export const TONE_NAMES: Record<Tone, string> = {
  1: "alto y plano",
  2: "ascendente",
  3: "bajo (desciende y sube)",
  4: "descendente",
};

export interface ToneEntry {
  pinyin: string;
  characters: string[];
}

export interface Syllable {
  syllable: string;
  tones: Partial<Record<string, ToneEntry>>;
}

// Generado por scripts/05_exportar_silabas_toneperfect.py: solo combinaciones sílaba-tono
// que son palabras reales en Tone Perfect.
export const SYLLABLES = syllablesData as Syllable[];

export const CALIBRATION_SYLLABLE = "ma";

export function findSyllable(name: string): Syllable {
  const found = SYLLABLES.find((entry) => entry.syllable === name);
  if (!found) throw new Error(`Sílaba desconocida: ${name}`);
  return found;
}

export function toneEntry(syllable: Syllable, tone: Tone): ToneEntry | undefined {
  return syllable.tones[String(tone)];
}

export function availableTones(syllable: Syllable): Tone[] {
  return TONES.filter((tone) => toneEntry(syllable, tone) !== undefined);
}

export function randomPracticeItem(): { syllable: Syllable; tone: Tone } {
  const syllable = SYLLABLES[Math.floor(Math.random() * SYLLABLES.length)];
  const tones = availableTones(syllable);
  return { syllable, tone: tones[Math.floor(Math.random() * tones.length)] };
}
