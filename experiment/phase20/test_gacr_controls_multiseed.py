import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiment.phase20.gacr_controls_multiseed import group_masks, remap_fixed_records, score_with_progress
from experiment.phase20.gacr_controls_multiseed_queue import decision
from experiment.phase9.eval_cf0_b3_beamfusion import CF0B2ItemHead, score_item_head


class CachedMultiseedTest(unittest.TestCase):
    def test_progress_scoring_exactly_matches_original_including_last_batch(self):
        torch.manual_seed(2)
        torch.set_num_threads(1)
        cfg = dict(num_items=65, max_history=4, d_model=8, num_layers=1, num_heads=2,
                   dropout=0., temperature_initial=.1)
        model = CF0B2ItemHead(**{k: v for k, v in cfg.items() if k != 'temperature_initial'}, temperature=.1)
        records = [{'history': [1, 4, 7][:1 + i % 3], 'candidate_ids': list(range(1 + i, 51 + i))} for i in range(5)]
        reference = copy.deepcopy(records)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'head.pt'
            torch.save({'model_config': cfg, 'model_state_dict': model.state_dict()}, checkpoint)
            score_item_head(reference, checkpoint, 2)
            score_with_progress(records, checkpoint, 2, lambda a, b: calls.append((a, b)))
        for actual, expected in zip(records, reference):
            np.testing.assert_array_equal(actual['cf'], expected['cf'])
        self.assertEqual(calls, [(2, 5), (4, 5), (5, 5)])

    def test_domain_frequency_boundary_is_inclusive_and_not_hardcoded(self):
        lookup = {'a': {'target': 1}, 'b': {'target': 2}, 'c': {'target': 3}}
        groups = group_masks(['a', 'b', 'c'], lookup, np.asarray([0, 5, 6, 7]), {'b'}, 6)
        np.testing.assert_array_equal(groups['target_frequency_le6'], [True, True, False])
        np.testing.assert_array_equal(groups['outside_c1_training_users'], [True, False, True])
        toys = group_masks(['a', 'b', 'c'], lookup, np.asarray([0, 5, 6, 7]), set(), 5)
        np.testing.assert_array_equal(toys['target_frequency_le5'], [True, False, False])

    def test_fixed_reordering_aligns_cf_and_preserves_absent_targets(self):
        ids = list(range(1, 51))
        original = {'user': 'a', 'candidate_ids': ids, 'seq': np.arange(50.),
                    'cf': np.arange(50.) * 3, 'target_position': 3}
        predicted = {'user_id': 'a', 'ranked_item_ids': ids[::-1], 'scores': list(range(50))}
        result = remap_fixed_records([original], [predicted])[0]
        np.testing.assert_array_equal(result['cf'], original['cf'][::-1])
        self.assertEqual(result['target_position'], 46)
        self.assertEqual(remap_fixed_records([{**original, 'target_position': -1}], [predicted])[0]['target_position'], -1)
        with self.assertRaises(AssertionError):
            remap_fixed_records([original], [{**predicted, 'ranked_item_ids': list(range(2, 52))}])

    def test_next_training_rule_does_not_select_one_lucky_seed(self):
        good = {'baseline': {'ndcg@10': .08}, 'aggregate': {}}
        for method in ('full', 'fixed'):
            good['aggregate'][method] = {'ndcg@10': {'mean_delta_vs_original_pcrf': .002,
                 'min': .081, 'positive_seed_count_vs_original_pcrf': 3},
                 'hit@10': {'mean_delta_vs_original_pcrf': .003}}
        result = decision({'Beauty': good, 'Toys': good})
        self.assertTrue(result['fixed']['supports_next_training'])
        bad = copy.deepcopy(good)
        bad['aggregate']['fixed']['ndcg@10']['positive_seed_count_vs_original_pcrf'] = 1
        bad['aggregate']['fixed']['ndcg@10']['min'] = .07
        result = decision({'Beauty': bad, 'Toys': good})
        self.assertFalse(result['fixed']['supports_next_training'])
        self.assertFalse(result['automatic_training_started'])


if __name__ == '__main__':
    unittest.main()
