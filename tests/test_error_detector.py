import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tonos.error_detector import ToneErrorDetector


def make_syllables(seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(400, 6))
    tones = np.repeat([1, 2, 3, 4], 100)
    labels = features[:, 0] + rng.normal(scale=0.5, size=400) > 0.8
    speakers = np.tile(np.arange(8), 50)
    return features, tones, labels, speakers


class TestToneErrorDetector:
    def test_scores_match_scikit_learn(self):
        features, tones, labels, speakers = make_syllables()
        detector = ToneErrorDetector.fit(features, tones, labels, speakers, c_by_tone={t: 0.1 for t in range(1, 5)})
        rows = tones == 2
        scaler = StandardScaler().fit(features[rows])
        reference = LogisticRegression(C=0.1, max_iter=3000).fit(scaler.transform(features[rows]), labels[rows])
        expected = reference.predict_proba(scaler.transform(features[rows]))[:, 1]
        np.testing.assert_allclose(detector.score(features[rows], tones[rows]), expected, rtol=1e-6)

    def test_state_dict_round_trip_keeps_scores(self):
        features, tones, labels, speakers = make_syllables()
        detector = ToneErrorDetector.fit(features, tones, labels, speakers)
        restored = ToneErrorDetector.from_state_dict(detector.state_dict())
        np.testing.assert_allclose(restored.score(features, tones), detector.score(features, tones))
        assert restored.chosen_c() == detector.chosen_c()

    def test_unknown_expected_tone_scores_nan(self):
        features, tones, labels, speakers = make_syllables()
        detector = ToneErrorDetector.fit(features[tones < 4], tones[tones < 4], labels[tones < 4], speakers[tones < 4])
        assert np.isnan(detector.score(features[:1], np.array([4]))).all()
