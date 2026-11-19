import type { ReactNode } from "react";

/** Título de un paso numerado dentro de una tarjeta, para que el orden de uso se vea solo. */
export function StepHeading({ number, children }: { number: number; children: ReactNode }) {
  return (
    <h3 className="step-heading">
      <span className="step-number" aria-hidden="true">{String(number).padStart(2, "0")}</span>
      {children}
    </h3>
  );
}
