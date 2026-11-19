import { useState } from "react";
import type { SpeakerProfile } from "./api";
import { CalibrationStep } from "./components/CalibrationStep";
import { LabStep } from "./components/LabStep";
import { PracticeStep } from "./components/PracticeStep";
import { ResultsStep } from "./components/ResultsStep";
import { clearProfile, loadProfile, saveProfile } from "./profileStorage";

const TABS = [
  { id: "practice", label: "Practicar" },
  { id: "lab", label: "¿Quién corrige?" },
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
      <header className="masthead">
        <div className="masthead-bar">
          <span className="label">Entrenador de tonos · mandarín para hispanohablantes</span>
          <nav className="nav" role="tablist" aria-label="Secciones">
            {TABS.map((option, index) => (
              <button
                key={option.id}
                type="button"
                role="tab"
                aria-selected={tab === option.id}
                className={tab === option.id ? "selected" : undefined}
                onClick={() => setTab(option.id)}
              >
                <span className="nav-index">{String(index + 1).padStart(2, "0")}</span>
                {option.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="masthead-title">
          {/* 声调: «tono» en chino. */}
          <span className="masthead-hanzi" aria-hidden="true">声调</span>
          <div>
            <h1>Cuatro tonos, una sílaba.</h1>
            <p className="lede">Graba una sílaba en mandarín y compara tu entonación con la de un hablante nativo.</p>
          </div>
        </div>
      </header>
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
