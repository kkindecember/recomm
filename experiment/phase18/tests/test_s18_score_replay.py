import unittest

import torch
from transformers import T5Config, T5ForConditionalGeneration

from experiment.phase18.core import score_replay as r


class ScoreReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2023)

    def test_precision_restores_on_exception(self):
        before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        with self.assertRaises(ValueError):
            with r.precision(False):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise ValueError()
        self.assertEqual(before, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_invalid_initial_duplicate_parent_is_excluded(self):
        step = {'input_prefixes': [[], []], 'input_valid': [True, False]}
        self.assertEqual(r.parent_indices(step, [[2, 3], [4, 5]], 0), [0, 0])
        step['input_valid'][1] = True
        with self.assertRaises(RuntimeError):
            r.parent_indices(step, [[2, 3]], 0)

    def test_native_cache_reorder_and_full_prefix_scores(self):
        model = T5ForConditionalGeneration(T5Config(vocab_size=12, d_model=16, d_ff=32, num_heads=2,
            num_layers=1, num_decoder_layers=1, dropout_rate=0., decoder_start_token_id=0,
            eos_token_id=1, pad_token_id=0)).eval()
        hidden = torch.randn(1, 4, 16)
        mask = torch.ones(1, 4, dtype=torch.long)
        paths = [[2, 4, 6], [3, 5, 7]]
        steps = [
            {'input_prefixes': [[], []], 'input_valid': [True, False],
             'retained': [{'parent_index': 0, 'token': 2}, {'parent_index': 0, 'token': 3}]},
            {'input_prefixes': [[2], [3]], 'input_valid': [True, True],
             'retained': [{'parent_index': 1, 'token': 5}, {'parent_index': 0, 'token': 4}]},
            {'input_prefixes': [[3, 5], [2, 4]], 'input_valid': [True, True], 'retained': []}]
        with torch.no_grad():
            cached = r.fixed_frontier(model, hidden, mask, steps, paths, 'cpu', True)
            full = r.fixed_frontier(model, hidden, mask, steps, paths, 'cpu', False)
            paired = r.path_full(model, hidden, mask, paths, 'cpu')
            pair_cached = r.path_cached(model, hidden, mask, paths, 'cpu')
            single = r.path_single(model, hidden, mask, paths, 'cpu')
        for other in [full, paired, pair_cached, single]:
            # CPU tiny-model rounding tolerance; not a scientific gate for real GPU results.
            torch.testing.assert_close(cached['logp'], other['logp'], rtol=1e-5, atol=1e-5)
            self.assertLess(r.compare(cached, other)['max_absolute_score_difference'], 1e-4)
        self.assertEqual(cached['summary']['tokens'], paths)
        self.assertEqual(len(cached['summary']['selected_log_probabilities'][0]), 3)

    def test_score_uses_full_vocabulary_without_trie_renormalization(self):
        logits = torch.tensor([[[0., 0., 0., 0.]]])
        p = r.pack(logits, logits.log_softmax(-1), [[2]], 'cpu')
        self.assertAlmostEqual(p['summary']['sum_float32_scores'][0], -1.38629436, places=6)


if __name__ == '__main__':
    unittest.main()
