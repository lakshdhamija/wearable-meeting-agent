import unittest

import numpy as np

from voiceprint_eval import consistency, decide, folds, mix_babble


class VoiceprintTest(unittest.TestCase):
    def test_names_only_above_threshold_and_margin(self):
        scores = np.array([
            [0.80, 0.20],  # clear winner
            [0.40, 0.10],  # below threshold
            [0.62, 0.60],  # above threshold, but too close to the runner-up
        ])
        np.testing.assert_array_equal(decide(scores, threshold=0.5, margin=0.05), [0, -1, -1])

    def test_consistency_is_low_when_a_label_holds_two_voices(self):
        dur = np.full(4, 10.0)
        same = np.tile([1.0, 0.0], (4, 1))
        mixed = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        self.assertAlmostEqual(consistency(same, dur, [0, 1, 2, 3]), 1.0)
        self.assertAlmostEqual(consistency(mixed, dur, [0, 1, 2, 3]), 0.0)

    def test_folds_take_one_stranger_per_series(self):
        enroll = {"meeting": np.array(["X1a"] * 4 + ["Y2a"] * 4), "speaker": np.array(list("pqrstuvw"))}
        self.assertEqual(folds(enroll), [{"p", "t"}, {"q", "u"}, {"r", "v"}, {"s", "w"}])

    def test_mix_hits_the_requested_ratio(self):
        rng = np.random.default_rng(0)
        speech = rng.normal(size=16000)
        talkers = [rng.normal(size=8000), rng.normal(size=24000)]
        noise = mix_babble(speech, talkers, snr_db=5) - speech
        ratio_db = 10 * np.log10(np.mean(speech ** 2) / np.mean(noise ** 2))
        self.assertAlmostEqual(ratio_db, 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
