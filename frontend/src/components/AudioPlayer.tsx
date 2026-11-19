import { useEffect, useRef, useState, type MouseEvent } from "react";

interface AudioPlayerProps {
  src: string;
  label: string;
  autoPlay?: boolean;
  preload?: "none" | "metadata" | "auto";
  onError?: () => void;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Reproductor con el estilo de la app: botón de play/pausa y barra de progreso que se puede pulsar. */
export function AudioPlayer({ src, label, autoPlay = false, preload = "metadata", onError }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [src]);

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      // play() falla si el navegador bloquea el audio o el archivo no carga; onError ya lo avisa.
      audio.play().catch(() => setIsPlaying(false));
    } else {
      audio.pause();
    }
  };

  const seek = (event: MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    audio.currentTime = ((event.clientX - bounds.left) / bounds.width) * duration;
  };

  const progress = duration ? (currentTime / duration) * 100 : 0;
  return (
    <div className={`audio-player${isPlaying ? " is-playing" : ""}`}>
      <audio
        ref={audioRef}
        src={src}
        autoPlay={autoPlay}
        preload={preload}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
        onDurationChange={(event) => setDuration(event.currentTarget.duration)}
        onError={onError}
      />
      <button type="button" className="audio-toggle" onClick={toggle} aria-label={`${isPlaying ? "Pausar" : "Reproducir"}: ${label}`}>
        <span className={isPlaying ? "icon-pause" : "icon-play"} aria-hidden="true" />
      </button>
      <div className="audio-body">
        <span className="audio-label">{label}</span>
        <div
          className="audio-track"
          onClick={seek}
          role="progressbar"
          aria-label={`Progreso: ${label}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress)}
        >
          <span className="audio-progress" style={{ width: `${progress}%` }} />
        </div>
      </div>
      <span className="audio-time">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </div>
  );
}
