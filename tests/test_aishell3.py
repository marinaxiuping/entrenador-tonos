import numpy as np
import pandas as pd

from tonos.aishell3 import lexical_tone_baseline, utterance_bounds


class TestUtteranceBounds:
    def test_returns_start_and_exclusive_end_of_each_utterance(self):
        ids = np.array(["a", "a", "b", "c", "c", "c"])

        np.testing.assert_array_equal(utterance_bounds(ids), [[0, 2], [2, 3], [3, 6]])

    def test_single_utterance(self):
        np.testing.assert_array_equal(utterance_bounds(np.array(["a"] * 4)), [[0, 4]])

    def test_no_utterances_for_an_empty_table(self):
        assert utterance_bounds(np.array([], dtype=str)).shape == (0, 2)


def make_table(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["hanzi", "pinyin", "tone"])


class TestLexicalToneBaseline:
    def test_predicts_the_most_frequent_tone_of_the_character_in_other_folds(self):
        table = make_table([
            ("好", "hao3", 3), ("好", "hao3", 3), ("好", "hao4", 4), ("好", "hao4", 4),
        ])
        fold_of = np.array([0, 0, 0, 1])

        assert lexical_tone_baseline(table, fold_of)[3] == 3

    def test_falls_back_to_the_toneless_syllable_for_unseen_characters(self):
        table = make_table([
            ("妈", "ma1", 1), ("妈", "ma1", 1), ("马", "ma3", 3), ("麻", "ma2", 2),
        ])
        fold_of = np.array([0, 0, 0, 1])

        assert lexical_tone_baseline(table, fold_of)[3] == 1

    def test_falls_back_to_the_most_frequent_tone_overall(self):
        table = make_table([
            ("是", "shi4", 4), ("是", "shi4", 4), ("妈", "ma1", 1), ("你", "ni3", 3),
        ])
        fold_of = np.array([0, 0, 0, 1])

        assert lexical_tone_baseline(table, fold_of)[3] == 4

    def test_does_not_use_the_tones_of_the_evaluated_fold(self):
        table = make_table([
            ("好", "hao3", 3), ("好", "hao4", 4), ("好", "hao4", 4),
            ("好", "hao4", 4), ("好", "hao3", 3), ("好", "hao3", 3),
        ])
        fold_of = np.array([0, 0, 0, 0, 1, 1])
        predictions = lexical_tone_baseline(table, fold_of)

        np.testing.assert_array_equal(predictions[4:], [4, 4])
        np.testing.assert_array_equal(predictions[:4], [3, 3, 3, 3])
