import translationsData from "./data/sentence_translations.json";

// Escritas a mano para las frases de AISHELL-3 que usa la app; AISHELL-3 no trae traducción.
const TRANSLATIONS: Record<string, string> = translationsData;

/** Traducción al castellano de una frase en caracteres, o undefined si no está traducida. */
export function translateSentence(sentence: string): string | undefined {
  return TRANSLATIONS[sentence];
}
