import numpy as np

from tonos.resynthesis import FRAME_PERIOD_MS, impose_contour_on_span

REF_HZ = 200.0


def frame_at(seconds: float) -> int:
    return int(round(seconds * 1000 / FRAME_PERIOD_MS))


class TestImposeContourOnSpan:
    def test_changes_only_the_voiced_frames_inside_the_span(self):
        f0 = np.full(100, 150.0)
        f0[30:33] = 0.0
        start, end = frame_at(0.1), frame_at(0.25)
        imposed = impose_contour_on_span(f0, 0.1, 0.25, np.zeros(64), REF_HZ)

        np.testing.assert_array_equal(imposed[:start], f0[:start])
        np.testing.assert_array_equal(imposed[end:], f0[end:])
        np.testing.assert_array_equal(imposed[30:33], 0.0)
        voiced_in_span = np.r_[start:30, 33:end]
        np.testing.assert_allclose(imposed[voiced_in_span], REF_HZ)

    def test_stretches_the_contour_between_the_first_and_last_voiced_frame(self):
        f0 = np.zeros(60)
        f0[22:38] = 150.0
        rise = np.linspace(-2.0, 4.0, 64)
        imposed = impose_contour_on_span(f0, 0.1, 0.2, rise, REF_HZ)

        np.testing.assert_allclose(imposed[22], REF_HZ * 2 ** (-2 / 12))
        np.testing.assert_allclose(imposed[37], REF_HZ * 2 ** (4 / 12))
        assert np.all(np.diff(imposed[22:38]) > 0)
        assert np.all(imposed[:22] == 0) and np.all(imposed[38:] == 0)

    def test_does_not_modify_its_input(self):
        f0 = np.full(60, 150.0)
        original = f0.copy()
        impose_contour_on_span(f0, 0.1, 0.2, np.zeros(64), REF_HZ)

        np.testing.assert_array_equal(f0, original)
