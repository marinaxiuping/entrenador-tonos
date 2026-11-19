import numpy as np
import pytest

from tonos.f0 import HOP_S, SR, f0_on_frame_grid, fix_octave_jumps

REF_HZ, FMIN, FMAX = 200.0, 60.0, 500.0


def semitones_above(hz: float, st: float) -> float:
    return hz * 2 ** (st / 12)


class TestF0OnFrameGrid:
    def test_shifts_praat_frames_to_their_time(self):
        # Praat empieza a media ventana de t=0: aquí el primer frame está a 32 ms.
        times = 0.032 + np.arange(5) * HOP_S
        f0 = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        grid = f0_on_frame_grid(f0, times, n_samples=SR)

        assert grid.size == 101
        assert np.isnan(grid[:3]).all()
        np.testing.assert_array_equal(grid[3:8], f0)
        assert np.isnan(grid[8:]).all()

    def test_drops_frames_past_the_end_of_the_audio(self):
        times = np.array([0.0, 0.05, 0.2])
        grid = f0_on_frame_grid(np.array([100.0, 120.0, 140.0]), times, n_samples=int(0.1 * SR))

        assert grid.size == 11
        assert grid[0] == 100.0 and grid[5] == 120.0
        assert np.isfinite(grid).sum() == 2


class TestFixOctaveJumps:
    def test_leaves_a_smooth_tone_4_fall_untouched(self):
        fall = np.geomspace(260.0, 150.0, 40)

        np.testing.assert_allclose(fix_octave_jumps(fall, REF_HZ, FMIN, FMAX), fall)

    def test_brings_a_doubled_stretch_back_down(self):
        f0 = np.full(50, REF_HZ)
        f0[20:30] *= 2

        np.testing.assert_allclose(fix_octave_jumps(f0, REF_HZ, FMIN, FMAX), np.full(50, REF_HZ))

    def test_anchors_on_the_run_in_register_even_if_shorter(self):
        # Tono 3 con voz creaky: el tramo a media octava ocupa casi toda la sílaba.
        f0 = np.r_[np.full(10, REF_HZ), np.full(30, REF_HZ / 2)]

        np.testing.assert_allclose(fix_octave_jumps(f0, REF_HZ, FMIN, FMAX), np.full(40, REF_HZ))

    def test_keeps_a_real_jump_smaller_than_the_shift_gap(self):
        # 7 st parte el contorno (> 6 st) pero no llega a los 9 st que hacen falta para desplazar una octava.
        f0 = np.r_[np.full(20, REF_HZ), np.full(20, semitones_above(REF_HZ, 7))]

        np.testing.assert_allclose(fix_octave_jumps(f0, REF_HZ, FMIN, FMAX), f0)

    def test_corrects_across_an_unvoiced_gap_and_keeps_it_unvoiced(self):
        f0 = np.r_[np.full(20, REF_HZ), np.full(5, np.nan), np.full(15, REF_HZ * 2)]
        corrected = fix_octave_jumps(f0, REF_HZ, FMIN, FMAX)

        assert np.isnan(corrected[20:25]).all()
        np.testing.assert_allclose(corrected[25:], REF_HZ)

    def test_discards_frames_outside_the_search_range(self):
        f0 = np.r_[np.full(20, REF_HZ), np.full(5, 550.0)]
        corrected = fix_octave_jumps(f0, REF_HZ, FMIN, fmax=260.0)

        np.testing.assert_allclose(corrected[:20], REF_HZ)
        # 550 Hz baja a 275 Hz, que sigue por encima de fmax.
        assert np.isnan(corrected[20:]).all()

    @pytest.mark.parametrize("f0", [np.array([np.nan, np.nan]), np.array([np.nan, 400.0])])
    def test_returns_contours_with_under_two_voiced_frames_as_they_are(self, f0):
        np.testing.assert_array_equal(fix_octave_jumps(f0, REF_HZ, FMIN, FMAX), f0)

    def test_does_not_modify_its_input(self):
        f0 = np.r_[np.full(20, REF_HZ), np.full(10, REF_HZ * 2)]
        original = f0.copy()
        fix_octave_jumps(f0, REF_HZ, FMIN, FMAX)

        np.testing.assert_array_equal(f0, original)
