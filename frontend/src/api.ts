export interface SpeakerProfile {
  reference_hz: number;
  fmin: number;
  fmax: number;
}

export interface ToneAnalysis {
  tone: number;
  probabilities: Record<string, number>;
  contour_st: number[];
  target_contour_st: number[] | null;
  correct: boolean | null;
}

export class ApiError extends Error {}

interface ValidationIssue {
  msg: string;
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((issue: ValidationIssue) => issue.msg).join("; ");
  } catch {
    // Respuesta sin JSON (p. ej. un proxy caído): se usa el estado HTTP.
  }
  return `Error ${response.status} del servidor`;
}

async function post(path: string, form: FormData): Promise<Response> {
  const response = await fetch(`/api${path}`, { method: "POST", body: form });
  if (!response.ok) throw new ApiError(await readErrorMessage(response));
  return response;
}

async function postForm<T>(path: string, form: FormData): Promise<T> {
  return (await post(path, form)).json() as Promise<T>;
}

function attemptForm(recording: Blob, profile: SpeakerProfile, targetTone: number): FormData {
  const form = new FormData();
  form.append("file", recording, "intento.wav");
  form.append("reference_hz", String(profile.reference_hz));
  form.append("fmin", String(profile.fmin));
  form.append("fmax", String(profile.fmax));
  form.append("target_tone", String(targetTone));
  return form;
}

export function calibrate(recordings: Blob[]): Promise<SpeakerProfile> {
  const form = new FormData();
  recordings.forEach((recording, i) => form.append("files", recording, `calibracion_${i + 1}.wav`));
  return postForm<SpeakerProfile>("/calibrate", form);
}

export function analyze(recording: Blob, profile: SpeakerProfile, targetTone: number): Promise<ToneAnalysis> {
  return postForm<ToneAnalysis>("/analyze", attemptForm(recording, profile, targetTone));
}

/** Tu grabación resintetizada con el contorno del tono objetivo (WAV). */
export async function resynthesize(recording: Blob, profile: SpeakerProfile, targetTone: number): Promise<Blob> {
  return (await post("/resynthesize", attemptForm(recording, profile, targetTone))).blob();
}

export interface LabSyllable {
  hanzi: string;
  pinyin: string;
  tone: number;
  evaluable: boolean;
}

export interface LabUtterance {
  id: string;
  speaker: string;
  gender: string;
  text: string;
  syllables: LabSyllable[];
}

export type JudgeName = "cnn_silaba" | "frase" | "hubert" | "lexico";

export interface Verdict {
  judge: JudgeName;
  tone: number | null;
  probabilities: Record<string, number> | null;
}

export interface JudgeResult {
  original_tone: number;
  imposed_tone: number | null;
  audio_wav_base64: string;
  original_contour_st: number[] | null;
  judged_contour_st: number[] | null;
  verdicts: Verdict[];
}

export async function fetchLabUtterances(): Promise<LabUtterance[]> {
  const response = await fetch("/api/lab/utterances");
  if (!response.ok) throw new ApiError(await readErrorMessage(response));
  return response.json() as Promise<LabUtterance[]>;
}

/** Cambia el tono de una sílaba (null = la deja como está) y pide el juicio de cada modelo. */
export async function judgeSyllable(utteranceId: string, syllableIndex: number,
                                    imposedTone: number | null): Promise<JudgeResult> {
  const response = await fetch("/api/lab/judge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ utterance_id: utteranceId, syllable_index: syllableIndex, imposed_tone: imposedTone }),
  });
  if (!response.ok) throw new ApiError(await readErrorMessage(response));
  return response.json() as Promise<JudgeResult>;
}

export function base64ToWavBlob(base64: string): Blob {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  return new Blob([bytes], { type: "audio/wav" });
}
