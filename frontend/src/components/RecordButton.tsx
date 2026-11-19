import { useEffect, useRef, useState } from "react";
import { startRecording, type RecordingSession } from "../audio/recorder";

interface RecordButtonProps {
  label: string;
  onRecorded: (recording: Blob) => void;
  onError: (message: string) => void;
  disabled?: boolean;
  // Una sílaba aislada dura menos de 1,5 s; el límite evita grabaciones olvidadas.
  maxSeconds?: number;
}

type RecordingState = "idle" | "recording" | "processing";

export function RecordButton({ label, onRecorded, onError, disabled = false, maxSeconds = 3 }: RecordButtonProps) {
  const [state, setState] = useState<RecordingState>("idle");
  const sessionRef = useRef<RecordingSession | null>(null);
  const timeoutRef = useRef<number | undefined>(undefined);

  // Si el botón desaparece a mitad de grabación, se cierra el micrófono igualmente.
  useEffect(
    () => () => {
      window.clearTimeout(timeoutRef.current);
      sessionRef.current?.stop().catch(() => undefined);
      sessionRef.current = null;
    },
    [],
  );

  const stop = async () => {
    const session = sessionRef.current;
    if (!session) return;
    sessionRef.current = null;
    window.clearTimeout(timeoutRef.current);
    setState("processing");
    try {
      onRecorded(await session.stop());
    } catch {
      onError("No se pudo procesar la grabación. Prueba otra vez.");
    } finally {
      setState("idle");
    }
  };

  const start = async () => {
    try {
      sessionRef.current = await startRecording();
    } catch {
      onError("No hay acceso al micrófono. Revisa los permisos del navegador.");
      return;
    }
    setState("recording");
    timeoutRef.current = window.setTimeout(stop, maxSeconds * 1000);
  };

  const captions: Record<RecordingState, string> = {
    idle: label,
    recording: "Parar",
    processing: "Procesando…",
  };
  return (
    <button
      type="button"
      className={`record-button ${state}`}
      onClick={state === "recording" ? stop : start}
      disabled={disabled || state === "processing"}
    >
      <span className="record-dot" aria-hidden="true" />
      {captions[state]}
    </button>
  );
}
