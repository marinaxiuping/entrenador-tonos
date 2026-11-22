/** Farolillo chino: decora la habitación y, apagado o encendido, cuenta las vidas. */
export function Lantern({ className, isLit = true }: { className?: string; isLit?: boolean }) {
  return (
    <svg
      className={`lantern${isLit ? "" : " is-unlit"}${className ? ` ${className}` : ""}`}
      viewBox="0 0 40 64"
      aria-hidden="true"
    >
      <line className="lantern-cord" x1="20" y1="0" x2="20" y2="8" />
      <rect className="lantern-cap" x="12" y="8" width="16" height="5" rx="1" />
      <ellipse className="lantern-body" cx="20" cy="29" rx="17" ry="16" />
      <path className="lantern-rib" d="M20 13 V45 M11 15 Q4 29 11 43 M29 15 Q36 29 29 43" />
      <rect className="lantern-cap" x="12" y="44" width="16" height="5" rx="1" />
      <path className="lantern-tassel" d="M17 49 L16 63 M20 49 V63 M23 49 L24 63" />
    </svg>
  );
}
