import { useEffect, useRef, useState } from "react";
import type { SpeakerProfile } from "../api";
import { startPitchTracker, type PitchTracker } from "../audio/pitchTracker";
import { describeRecordingStartError } from "../audio/recorder";
import {
  TOLERANCE_ST,
  hzToSemitones,
  isOnTarget,
  scoreRound,
  type PitchAxis,
  type PitchSample,
} from "../toneGame";

interface GameTrackProps {
  contour: number[];
  durationS: number;
  axis: PitchAxis;
  profile: SpeakerProfile;
  onFinished: (score: number) => void;
  /** Avisa de cuándo está el micrófono abierto, para no tararear la guía encima. */
  onListeningChange: (isListening: boolean) => void;
}

type TrackPhase = "idle" | "waiting" | "running" | "done";

// Tamaño lógico del lienzo; el CSS lo escala al ancho disponible.
const WIDTH = 640;
const HEIGHT = 280;
const PADDING_X = 28;
const PADDING_Y = 16;
const CHARACTER_RADIUS = 11;
// Fotogramas seguidos con voz para arrancar: un golpe o un clic no deben lanzar el recorrido.
const VOICED_FRAMES_TO_START = 3;
const WAIT_FOR_VOICE_MS = 6000;
// Un hueco mayor que esto en el recorrido es un silencio: el rastro se corta en vez de unirse.
const MAX_TRAIL_GAP = 0.04;

interface Scene {
  contour: number[];
  axis: PitchAxis;
  trail: PitchSample[];
  character: PitchSample | null;
}

/** Una ronda: la curva objetivo y el personaje que sube y baja con la voz. */
export function GameTrack({ contour, durationS, axis, profile, onFinished, onListeningChange }: GameTrackProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackerRef = useRef<PitchTracker | null>(null);
  const waitTimeoutRef = useRef<number | undefined>(undefined);
  const isMountedRef = useRef(true);
  const [phase, setPhase] = useState<TrackPhase>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (canvasRef.current) drawScene(canvasRef.current, { contour, axis, trail: [], character: null });
  }, [contour, axis]);

  const releaseMicrophone = () => {
    window.clearTimeout(waitTimeoutRef.current);
    trackerRef.current?.stop();
    trackerRef.current = null;
  };

  useEffect(() => onListeningChange(phase === "waiting" || phase === "running"), [phase, onListeningChange]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      releaseMicrophone();
    };
  }, []);

  const start = async () => {
    setError(null);
    setPhase("waiting");
    const trail: PitchSample[] = [];
    let voicedFrames = 0;
    let startedAt: number | null = null;

    const handlePitch = (hz: number | null) => {
      const canvas = canvasRef.current;
      if (!canvas || !trackerRef.current) return;
      const semitones = hz === null ? null : hzToSemitones(hz, profile.reference_hz);
      if (startedAt === null) {
        voicedFrames = semitones === null ? 0 : voicedFrames + 1;
        if (voicedFrames >= VOICED_FRAMES_TO_START) {
          startedAt = performance.now();
          window.clearTimeout(waitTimeoutRef.current);
          setPhase("running");
        }
        const character = semitones === null ? null : { progress: 0, semitones };
        drawScene(canvas, { contour, axis, trail, character });
        return;
      }
      const progress = (performance.now() - startedAt) / 1000 / durationS;
      const character = semitones === null ? null : { progress: Math.min(progress, 1), semitones };
      if (character && progress <= 1) trail.push(character);
      drawScene(canvas, { contour, axis, trail, character });
      if (progress >= 1) {
        releaseMicrophone();
        setPhase("done");
        onFinished(scoreRound(trail, contour));
      }
    };

    try {
      const tracker = await startPitchTracker({ minHz: profile.fmin, maxHz: profile.fmax }, handlePitch);
      // Si la ronda se cerró mientras el navegador pedía permiso, el micrófono no debe quedar abierto.
      if (!isMountedRef.current) {
        tracker.stop();
        return;
      }
      trackerRef.current = tracker;
    } catch (failure) {
      console.error("Error al abrir el micrófono", failure);
      setError(describeRecordingStartError(failure));
      setPhase("idle");
      return;
    }
    waitTimeoutRef.current = window.setTimeout(() => {
      releaseMicrophone();
      setError("No se oye tu voz. Acércate al micrófono y vuelve a empezar.");
      setPhase("idle");
    }, WAIT_FOR_VOICE_MS);
  };

  return (
    <div className="game-track">
      <canvas
        ref={canvasRef}
        width={WIDTH * window.devicePixelRatio}
        height={HEIGHT * window.devicePixelRatio}
        role="img"
        aria-label="Curva del tono objetivo y el recorrido de tu voz"
      />
      {phase === "idle" && (
        <button type="button" className="primary" onClick={start}>
          Empezar
        </button>
      )}
      {phase === "waiting" && <p className="muted">Canta cuando quieras: el recorrido arranca al oír tu voz.</p>}
      {phase === "running" && <p className="muted">Sigue la curva…</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

function drawScene(canvas: HTMLCanvasElement, scene: Scene): void {
  const context = canvas.getContext("2d");
  if (!context) return;
  const styles = getComputedStyle(canvas);
  const color = (name: string) => styles.getPropertyValue(name).trim();
  const x = (progress: number) => PADDING_X + progress * (WIDTH - 2 * PADDING_X);
  const y = (semitones: number) => {
    const { bottomSt, topSt } = scene.axis;
    const clamped = Math.min(Math.max(semitones, bottomSt), topSt);
    return PADDING_Y + ((topSt - clamped) / (topSt - bottomSt)) * (HEIGHT - 2 * PADDING_Y);
  };
  const steps = scene.contour.length - 1;
  const progressAt = (i: number) => i / steps;

  context.setTransform(canvas.width / WIDTH, 0, 0, canvas.height / HEIGHT, 0, 0);
  context.clearRect(0, 0, WIDTH, HEIGHT);

  context.strokeStyle = color("--hairline");
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(PADDING_X, y(0));
  context.lineTo(WIDTH - PADDING_X, y(0));
  context.stroke();

  context.fillStyle = color("--sand-light");
  context.beginPath();
  scene.contour.forEach((value, i) => context.lineTo(x(progressAt(i)), y(value + TOLERANCE_ST)));
  for (let i = steps; i >= 0; i--) context.lineTo(x(progressAt(i)), y(scene.contour[i] - TOLERANCE_ST));
  context.closePath();
  context.fill();

  context.strokeStyle = color("--target-line");
  context.lineWidth = 3;
  context.lineCap = "round";
  context.beginPath();
  scene.contour.forEach((value, i) => context.lineTo(x(progressAt(i)), y(value)));
  context.stroke();

  context.strokeStyle = color("--user-line");
  context.lineWidth = 2.5;
  context.beginPath();
  scene.trail.forEach((sample, i) => {
    const isAfterGap = i === 0 || sample.progress - scene.trail[i - 1].progress > MAX_TRAIL_GAP;
    if (isAfterGap) context.moveTo(x(sample.progress), y(sample.semitones));
    else context.lineTo(x(sample.progress), y(sample.semitones));
  });
  context.stroke();

  if (scene.character) {
    const bodyColor = color(isOnTarget(scene.character, scene.contour) ? "--ok" : "--user-line");
    drawCharacter(context, x(scene.character.progress), y(scene.character.semitones), bodyColor, color("--paper-light"));
  }
}

function drawCharacter(context: CanvasRenderingContext2D, centerX: number, centerY: number, body: string, eyes: string): void {
  context.fillStyle = body;
  context.beginPath();
  context.arc(centerX, centerY, CHARACTER_RADIUS, 0, 2 * Math.PI);
  context.fill();
  context.fillStyle = eyes;
  for (const offset of [-4, 4]) {
    context.beginPath();
    context.arc(centerX + offset, centerY - 3, 2, 0, 2 * Math.PI);
    context.fill();
  }
}
