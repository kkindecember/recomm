import unittest
import numpy as np
from experiment.phase21.text_b_fusion_diagnostic import decomposition, cohort_summary


class FusionDiagnosticTest(unittest.TestCase):
    def test_crossed_coefficients_and_history_penalty(self):
        scores = np.array([[.1, .8, .3], [.4, .2, .5], [.3, .4, .9]], dtype=np.float32)
        w_a, w_b = np.array([.5, .3, .2]), np.array([.2, .2, .6])
        new = np.array([.7, .6, .1])
        got = decomposition(scores, new, w_a, w_b, .15, .25, [1, 1, 3])
        for name, weights, penalty, semantic in [('AA', w_a, .15, scores[2]), ('AB', w_a, .15, new),
                                                ('BA', w_b, .25, scores[2]), ('BB', w_b, .25, new)]:
            expected = [sum(weights[k] * (semantic[i] if k == 2 else scores[k, i]) for k in range(3))
                        - (penalty if i in (0, 2) else 0.) for i in range(3)]
            np.testing.assert_allclose(got[name], expected, atol=1e-7)
        np.testing.assert_array_equal(scores[2], np.array([.3, .4, .9], dtype=np.float32))

    def test_gained_and_lost_are_both_counted_without_dropping_users(self):
        rr = {k: np.array(v) for k, v in dict(text_A=[20, 1, 2, 30], text_B=[1, 20, 2, 30],
              SASRec=[30]*4, ItemCF=[30]*4, GRAM=[30]*4, AA=[30, 1, 2, 30], BB=[30, 30, 2, 1]).items()}
        c = cohort_summary(rr)
        self.assertEqual(c['all']['users'], 4)
        self.assertEqual(c['B_text_gained']['users'], 1)
        self.assertEqual(c['B_text_gained']['top10']['BB'], 0)
        self.assertEqual(c['B_text_lost']['users'], 1)
        self.assertEqual(c['B_text_unique']['users'], 2)

    def test_b_only_worker_uses_cached_fit_and_existing_resumable_run(self):
        from experiment.phase21.text_b_followup import FollowupWorker
        from experiment.phase21.text_adaptation import GenerationWorker
        from experiment.phase21.night_run import ContinueGeneration
        self.assertIs(FollowupWorker.fit_signals, GenerationWorker.fit_signals)
        self.assertIs(FollowupWorker.run, ContinueGeneration.run)


if __name__ == '__main__': unittest.main()
