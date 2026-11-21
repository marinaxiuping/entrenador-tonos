import type { JudgeName } from "./api";

interface JudgeInfo {
  name: string;
  hears: string;
}

export const JUDGE_ORDER: JudgeName[] = ["cnn_silaba", "frase", "hubert", "detector", "lexico"];

export const JUDGES: Record<JudgeName, JudgeInfo> = {
  cnn_silaba: { name: "CNN por sílaba", hears: "Oye solo el tono de esa sílaba" },
  frase: { name: "CNN + BiGRU", hears: "Oye el tono de toda la frase" },
  hubert: { name: "HuBERT", hears: "Oye todo el audio (modelo de habla preentrenado)" },
  detector: { name: "HuBERT + alumnos", hears: "Aprendió de errores reales de alumnos: avisa si el tono está mal" },
  lexico: { name: "Léxico", hears: "No oye nada: usa el tono habitual del carácter" },
};

const PERCENT = new Intl.NumberFormat("es-ES", { style: "percent", maximumFractionDigits: 1 });

/** Con espacio de no separación antes de «%» para que no se parta en dos líneas. */
export function formatPercent(value: number): string {
  return PERCENT.format(value).replace(" ", " ");
}
