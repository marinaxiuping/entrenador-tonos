"""Pipeline de F0: carga -> recorte de silencios -> F0 -> semitonos por hablante -> contorno fijo."""

from __future__ import annotations

import librosa
import numpy as np
import parselmouth
from joblib import Parallel, delayed

SR = 16_000
HOP_S = 0.01  # 10 ms
FMIN, FMAX = 60.0, 500.0


def load(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt


def f0_pyin(y: np.ndarray, sr: int = SR, fmin: float = FMIN,
            fmax: float = FMAX) -> tuple[np.ndarray, np.ndarray]:
    """F0 en Hz (NaN en tramos sordos) y tiempos en s."""
    hop = int(HOP_S * sr)
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=hop)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    return f0, t


def f0_praat(y: np.ndarray, sr: int = SR, fmin: float = FMIN,
             fmax: float = FMAX) -> tuple[np.ndarray, np.ndarray]:
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    return f0, pitch.xs()


_CREPE_DEVICE = None


def f0_crepe(y: np.ndarray, sr: int = SR, fmin: float = FMIN, fmax: float = FMAX,
             periodicity_thr: float = 0.21, model: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """CREPE (torchcrepe); frames con periodicidad < umbral → NaN.

    Decodificación weighted_argmax + autocast fp16 en GPU: 3,4x más rápido que fp32 y con F0
    idéntico; Viterbi no aporta frente a argmax + clean_f0 (dif. mediana 0,06 st) y es más lento.
    """
    global _CREPE_DEVICE
    import torch
    import torchcrepe

    if _CREPE_DEVICE is None:
        _CREPE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    hop = int(HOP_S * sr)
    x = torch.from_numpy(y).float().unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.float16, enabled=_CREPE_DEVICE == "cuda"):
        f0, per = torchcrepe.predict(x, sr, hop, fmin, fmax, model=model, batch_size=2048,
                                     device=_CREPE_DEVICE, return_periodicity=True,
                                     decoder=torchcrepe.decode.weighted_argmax)
    per = torchcrepe.filter.median(per.float(), 3)
    f0 = f0[0].float().cpu().numpy().astype(float)
    f0[per[0].cpu().numpy() < periodicity_thr] = np.nan
    return f0, np.arange(f0.size) * HOP_S


F0_METHODS = {"pyin": f0_pyin, "praat": f0_praat, "crepe": f0_crepe}


def f0_on_frame_grid(f0: np.ndarray, times: np.ndarray, n_samples: int, sr: int = SR) -> np.ndarray:
    """Coloca el F0 en una rejilla cuyo frame i corresponde a i * HOP_S segundos.

    Praat devuelve el primer frame a media ventana del inicio, no en t=0; sin este ajuste
    los tiempos de un alineamiento caerían varios frames desplazados sobre el F0.
    """
    grid = np.full(int(n_samples / sr / HOP_S) + 1, np.nan)
    frame_ids = np.round(times / HOP_S).astype(int)
    inside = frame_ids < grid.size
    grid[frame_ids[inside]] = f0[inside]
    return grid


def clean_f0(f0: np.ndarray, max_dev_st: float = 9.0, medfilt: int = 5) -> np.ndarray:
    """Corrige errores de octava y suaviza. Enfoque descartado: lo sustituye ``fix_octave_jumps``.

    Se conserva porque la sección 3.1 del notebook 01 lo ejecuta para mostrar por qué se
    descartó (altera la mitad de los tonos 4 de Tone Perfect).

    Frames a más de ``max_dev_st`` semitonos de la mediana de la frase se desplazan ±12 st
    si así vuelven al rango; si no, se descartan (NaN). Después, filtro de mediana sobre
    cada tramo sonoro contiguo (no cruza huecos sordos).
    """
    f0 = f0.astype(float).copy()
    ok = np.isfinite(f0)
    if ok.sum() < 3:
        return f0
    st = 12 * np.log2(f0[ok] / np.median(f0[ok]))
    for shift in (12.0, -12.0):
        fix = (np.abs(st) > max_dev_st) & (np.abs(st + shift) <= max_dev_st)
        st[fix] += shift
    st[np.abs(st) > max_dev_st] = np.nan
    f0[ok] = np.median(f0[ok]) * 2 ** (st / 12)
    if medfilt > 1:
        from scipy.ndimage import median_filter

        ok = np.isfinite(f0)
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= medfilt:
                f0[a:b] = median_filter(f0[a:b], size=medfilt, mode="nearest")
    return f0


def fix_octave_jumps(f0: np.ndarray, ref_hz: float, fmin: float, fmax: float,
                     max_jump_st: float = 6.0, min_shift_gap_st: float = 9.0, edge_frames: int = 3,
                     max_anchor_offset_st: float = 9.0) -> np.ndarray:
    """Recoloca por octavas los tramos separados por saltos imposibles para la voz.

    La voz no cambia más de unos pocos semitonos en 10 ms, así que un salto mayor de
    ``max_jump_st`` entre frames sonoros consecutivos parte el contorno en tramos. Se toma
    como correcto el tramo más largo cuya mediana está a menos de ``max_anchor_offset_st``
    de la referencia del hablante (el más largo a secas falla en tonos 3 con voz creaky,
    donde el error de octava ocupa casi toda la sílaba). Hacia cada lado, cada tramo se
    desplaza un múltiplo de 12 st para unirlo con su vecino ya corregido, pero solo si el
    hueco entre ambos es de al menos ``min_shift_gap_st``: los saltos de 6-9 st a través de
    un tramo sordo pueden ser reales y redondearlos a una octava los estropea. El hueco se
    mide con la mediana de ``edge_frames`` frames a cada lado para que un frame anómalo en
    el borde no arrastre al tramo siguiente. Al final se descartan los frames fuera de
    [fmin, fmax]. A diferencia de ``clean_f0``, no compara con la mediana de la sílaba, que
    confundiría la caída real del tono 4 con un error.
    """
    f0 = f0.astype(float).copy()
    voiced = np.flatnonzero(np.isfinite(f0))
    if voiced.size < 2:
        return f0
    st = 12 * np.log2(f0[voiced] / ref_hz)
    run_starts = np.r_[0, np.flatnonzero(np.abs(np.diff(st)) > max_jump_st) + 1]
    run_ends = np.r_[run_starts[1:], st.size]
    run_lengths = run_ends - run_starts
    in_register = np.array([abs(np.median(st[a:b])) <= max_anchor_offset_st
                            for a, b in zip(run_starts, run_ends)])
    anchor = int(np.argmax(np.where(in_register, run_lengths, -1)))

    def head(run: int) -> float:
        return float(np.median(st[run_starts[run]:min(run_ends[run], run_starts[run] + edge_frames)]))

    def tail(run: int) -> float:
        return float(np.median(st[max(run_starts[run], run_ends[run] - edge_frames):run_ends[run]]))

    def align(run: int, gap: float) -> None:
        if abs(gap) >= min_shift_gap_st:
            st[run_starts[run]:run_ends[run]] += 12 * np.round(gap / 12)

    for run in range(anchor + 1, run_starts.size):
        align(run, tail(run - 1) - head(run))
    for run in range(anchor - 1, -1, -1):
        align(run, head(run + 1) - tail(run))
    corrected = ref_hz * 2 ** (st / 12)
    corrected[(corrected < fmin) | (corrected > fmax)] = np.nan
    f0[voiced] = corrected
    return f0


def waveform_f0(y: np.ndarray, method: str = "pyin", top_db: float = 30.0,
                fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    f0, _ = F0_METHODS[method](trim_silence(y, top_db=top_db), fmin=fmin, fmax=fmax)
    return f0


def extract_f0(path: str, method: str = "pyin", top_db: float = 30.0,
               fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    return waveform_f0(load(path), method, top_db, fmin, fmax)


def speaker_pitch_range(f0_list: list[np.ndarray], floor_factor: float = 0.5,
                        ceiling_factor: float = 1.5) -> tuple[float, float]:
    """Rango de búsqueda de F0 adaptado al hablante (dos pasadas, De Looze y Hirst 2008).

    Con el rango genérico 60-500 Hz el tracker cae en errores de octava; acotarlo a
    [floor_factor * Q1, ceiling_factor * Q3] del F0 de una primera pasada impide que elija
    candidatos a una octava de distancia del registro del hablante.
    El suelo es más bajo que el 0,75 habitual porque los tonos 3 y 4 bajan de forma
    legítima hasta ~12 st por debajo de la media del hablante.
    """
    voiced = np.concatenate([f[np.isfinite(f)] for f in f0_list])
    q1, q3 = np.percentile(voiced, [25, 75])
    return max(FMIN, floor_factor * q1), min(FMAX, ceiling_factor * q3)


def extract_corpus_f0(paths: list[str], speakers: list[str], method: str = "praat",
                      n_jobs: int = -1) -> list[np.ndarray]:
    """F0 de un corpus de sílabas con corrección de octava por hablante.

    Primera pasada con el rango genérico para estimar el rango de cada hablante, segunda
    pasada acotada a ese rango y, por último, ``fix_octave_jumps`` frente a su referencia.
    """
    def by_speaker(f0s: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
        grouped: dict[str, list[np.ndarray]] = {}
        for f0, s in zip(f0s, speakers):
            grouped.setdefault(s, []).append(f0)
        return grouped

    # Compila/caché numba en el proceso principal antes de paralelizar: si varios workers
    # escriben la caché de librosa a la vez se corrompe (access violation en localmin).
    extract_f0(paths[0], method)
    f0s = Parallel(n_jobs=n_jobs)(delayed(extract_f0)(p, method) for p in paths)
    ranges = {s: speaker_pitch_range(fs) for s, fs in by_speaker(f0s).items()}
    f0s = Parallel(n_jobs=n_jobs)(
        delayed(extract_f0)(p, method, fmin=ranges[s][0], fmax=ranges[s][1])
        for p, s in zip(paths, speakers)
    )
    refs = {s: speaker_reference_hz(fs) for s, fs in by_speaker(f0s).items()}
    return [fix_octave_jumps(f0, refs[s], *ranges[s]) for f0, s in zip(f0s, speakers)]


def speaker_reference_hz(f0_list: list[np.ndarray]) -> float:
    """Media geométrica del F0 sonoro del hablante (= media en escala de semitonos)."""
    voiced = np.concatenate([f[np.isfinite(f)] for f in f0_list])
    return float(np.exp(np.mean(np.log(voiced))))


def hz_to_st(f0: np.ndarray, ref_hz: float) -> np.ndarray:
    return 12.0 * np.log2(f0 / ref_hz)


def fixed_length_contour(f0: np.ndarray, n: int = 30, min_voiced: int = 5) -> np.ndarray:
    """Recorta NaN de los extremos, interpola huecos internos y remuestrea a n puntos.

    Devuelve un array de NaN si hay menos de ``min_voiced`` frames sonoros.
    """
    idx = np.flatnonzero(np.isfinite(f0))
    if idx.size < min_voiced:
        return np.full(n, np.nan)
    seg = f0[idx[0] : idx[-1] + 1]
    x = np.arange(seg.size)
    ok = np.isfinite(seg)
    seg = np.interp(x, x[ok], seg[ok])
    return np.interp(np.linspace(0, seg.size - 1, n), x, seg)
