import type { SpeakerProfile } from "./api";

const STORAGE_KEY = "tonos.speakerProfile";

// El perfil se guarda en el navegador solo por comodidad: si se pierde (modo privado,
// datos borrados), basta con volver a calibrar.
export function loadProfile(): SpeakerProfile | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? (JSON.parse(stored) as SpeakerProfile) : null;
  } catch {
    return null;
  }
}

export function saveProfile(profile: SpeakerProfile): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
  } catch {
    // Sin almacenamiento disponible: el perfil vive solo en esta sesión.
  }
}

export function clearProfile(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nada que borrar si el almacenamiento no está disponible.
  }
}
