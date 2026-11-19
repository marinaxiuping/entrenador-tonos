import { useEffect, useRef, useState } from "react";
import { describeRecordingStartError, startRecording, type RecordingSession } from "../audio/recorder";

interface RecordButtonProps {
  label: string;
  onRecorded: (recording: Blob) => void;
  onError: (message: string) => void;
  disabled?: boolean;
  // El botón principal de la pantalla va más grande y en color.
  isProminent?: boolean;
  // Una sílaba aislada dura menos de 1,5 s; el límite evita grabaciones olvidadas.
  maxSeconds?: number;
}

type RecordingState = "idle" | "recording" | "processing";

export function RecordButton({
  label,
  onRecorded,
  onError,
  disabled = false,
  isProminent = false,
  maxSeconds = 3,
}: RecordButtonProps) {
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
    } catch (error) {
      console.error("Error al procesar la grabación", error);
      onError("No se pudo procesar la grabación. Prueba otra vez.");
    } finally {
      setState("idle");
    }
  };

  const start = async () => {
    try {
      sessionRef.current = await startRecording();
    } catch (error) {
      console.error("Error al empezar a grabar", error);
      onError(describeRecordingStartError(error));
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
      className={`record-button ${state}${isProminent ? " prominent" : ""}`}
      onClick={state === "recording" ? stop : start}
      disabled={disabled || state === "processing"}
    >
      <span className="record-dot" aria-hidden="true" />
      {captions[state]}
    </button>
  );
}
