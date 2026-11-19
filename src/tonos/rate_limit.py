"""Límite de peticiones por cliente con ventana deslizante, en memoria.

Protege las rutas que ocupan la CPU durante cientos de milisegundos (resíntesis con WORLD,
juicio con HuBERT). Vive en el proceso: con varios workers o réplicas cada uno lleva su
propia cuenta, que basta para un despliegue de un solo proceso como el de Spaces.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        if max_requests < 1 or window_s <= 0:
            raise ValueError("max_requests debe ser al menos 1 y window_s positivo")
        self.max_requests = max_requests
        self.window_s = window_s
        self._clock = clock
        self._requests: dict[str, deque[float]] = {}
        self._last_sweep = clock()
        # Las rutas síncronas de FastAPI se ejecutan en un pool de hilos.
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> bool:
        """Registra la petición y dice si entra en el límite; las rechazadas no cuentan."""
        with self._lock:
            now = self._clock()
            self._sweep_idle_clients(now)
            timestamps = self._requests.setdefault(client_key, deque())
            self._drop_expired(timestamps, now)
            if len(timestamps) >= self.max_requests:
                return False
            timestamps.append(now)
            return True

    def _drop_expired(self, timestamps: deque[float], now: float) -> None:
        while timestamps and timestamps[0] <= now - self.window_s:
            timestamps.popleft()

    def _sweep_idle_clients(self, now: float) -> None:
        # Sin esta limpieza, cada cliente que pasa una vez dejaría su entrada para siempre.
        if now - self._last_sweep < self.window_s:
            return
        for client_key in list(self._requests):
            self._drop_expired(self._requests[client_key], now)
            if not self._requests[client_key]:
                del self._requests[client_key]
        self._last_sweep = now
