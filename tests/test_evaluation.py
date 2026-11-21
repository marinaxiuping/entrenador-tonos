import json

import numpy as np
import pandas as pd

from tonos.error_detector import false_alarm_threshold, out_of_fold_scores
from tonos.evaluation import speaker_bootstrap_auc
from tonos.ompal import is_tone_error, partition_utterances


def make_scored_syllables(seed: int = 0):
    rng = np.random.default_rng(seed)
    speakers = np.repeat(np.arange(10), 40)
    labels = rng.random(400) < 0.3
    noisy = labels + rng.normal(scale=0.8, size=400)
    return labels, noisy, speakers


class TestSpeakerBootstrapAuc:
    def test_identical_judges_have_zero_paired_difference(self):
        labels, noisy, speakers = make_scored_syllables()
        scores = pd.DataFrame({"a": noisy, "b": noisy})

        boot = speaker_bootstrap_auc(labels, scores, speakers, n_resamples=50, seed=0)

        np.testing.assert_array_equal(boot[:, 0], boot[:, 1])

    def test_perfect_judge_scores_one_in_every_resample(self):
        labels, noisy, speakers = make_scored_syllables()
        scores = pd.DataFrame({"perfecto": labels.astype(float), "ruidoso": noisy})

        boot = speaker_bootstrap_auc(labels, scores, speakers, n_resamples=50, seed=0)

        assert (boot[:, 0] == 1).all()
        assert (boot[:, 1] < 1).all()

    def test_same_seed_gives_same_resamples(self):
        labels, noisy, speakers = make_scored_syllables()
        scores = pd.DataFrame({"a": noisy})

        first = speaker_bootstrap_auc(labels, scores, speakers, n_resamples=20, seed=3)
        second = speaker_bootstrap_auc(labels, scores, speakers, n_resamples=20, seed=3)

        np.testing.assert_array_equal(first, second)


class TestFalseAlarmThreshold:
    def test_alerts_on_the_target_share_of_correct_syllables(self):
        rng = np.random.default_rng(0)
        scores = rng.random(10_000)
        labels = np.zeros(10_000, dtype=bool)

        threshold = false_alarm_threshold(scores, labels, false_alarm=0.1)

        assert abs((scores > threshold).mean() - 0.1) < 0.001

    def test_ignores_syllables_with_errors(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.99, 0.99])
        labels = np.array([False, False, False, False, True, True])

        assert false_alarm_threshold(scores, labels, false_alarm=0.0) == 0.4


class TestOutOfFoldScores:
    def test_scores_every_row(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(300, 4))
        tones = np.tile([1, 2, 3], 100)
        labels = features[:, 0] > 0.5
        speakers = np.repeat(np.arange(10), 30)

        scores = out_of_fold_scores(features, tones, labels, speakers, n_folds=5)

        assert not np.isnan(scores).any()
        assert ((scores >= 0) & (scores <= 1)).all()


class TestOmpal:
    def test_reads_the_utterance_ids_of_a_partition(self, tmp_path):
        (tmp_path / "test").mkdir()
        (tmp_path / "test" / "test_2_scores.json").write_text(
            json.dumps({"00200101": {"accuracy": 3}, "00100205": {"accuracy": 4}}), encoding="utf-8")

        assert partition_utterances(tmp_path, "test", 2) == {"00200101", "00100205"}

    def test_tone_error_needs_a_majority_of_annotators(self):
        votes = pd.DataFrame({"tone_error_votes": [0, 1, 2, 3, np.nan]})

        assert is_tone_error(votes).tolist() == [False, False, True, True, False]
