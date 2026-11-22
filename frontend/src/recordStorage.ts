const STORAGE_KEY = "tonos.roomsRecord";

// El récord se guarda en el navegador solo por comodidad: si se pierde, se empieza de cero.
export function loadRecord(): number {
  try {
    return Number(localStorage.getItem(STORAGE_KEY)) || 0;
  } catch {
    return 0;
  }
}

export function saveRecord(rooms: number): void {
  try {
    localStorage.setItem(STORAGE_KEY, String(rooms));
  } catch {
    // Sin almacenamiento disponible: el récord vive solo en esta sesión.
  }
}
