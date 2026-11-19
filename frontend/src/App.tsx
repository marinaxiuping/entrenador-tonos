import { useState } from "react";
import type { SpeakerProfile } from "./api";
import { CalibrationStep } from "./components/CalibrationStep";
import { LabStep } from "./components/LabStep";
import { PracticeStep } from "./components/PracticeStep";
import { ResultsStep } from "./components/ResultsStep";
import { clearProfile, loadProfile, saveProfile } from "./profileStorage";

const TABS = [
  { id: "practice", label: "Practicar" },
  { id: "lab", label: "¿Quién te corrige?" },
  { id: "results", label: "Resultados" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export function App() {
  const [profile, setProfile] = useState<SpeakerProfile | null>(loadProfile);
  const [tab, setTab] = useState<TabId>("practice");

  const handleCalibrated = (newProfile: SpeakerProfile) => {
    saveProfile(newProfile);
    setProfile(newProfile);
  };

  const handleRecalibrate = () => {
    clearProfile();
    setProfile(null);
  };

  return (
    <main className="app">
      <header>
        <h1>Entrenador de tonos</h1>
        <p className="muted">Graba una sílaba y compara tu entonación con la de un hablante nativo.</p>
      </header>
      <nav className="tabs" role="tablist" aria-label="Secciones">
        {TABS.map((option) => (
          <button
            key={option.id}
            type="button"
            role="tab"
            aria-selected={tab === option.id}
            className={tab === option.id ? "selected" : undefined}
            onClick={() => setTab(option.id)}
          >
            {option.label}
          </button>
        ))}
      </nav>
      {tab === "practice" &&
        (profile ? (
          <PracticeStep profile={profile} onRecalibrate={handleRecalibrate} />
        ) : (
          <CalibrationStep onCalibrated={handleCalibrated} />
        ))}
      {tab === "lab" && <LabStep />}
      {tab === "results" && <ResultsStep />}
    </main>
  );
}
