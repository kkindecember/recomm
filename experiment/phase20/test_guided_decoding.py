"""Scientific invariants for the frozen search intervention (CPU, no downloads)."""
import unittest
import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration, LogitsProcessorList
from experiment.phase20.guided_decoding_core import PrefixIndex, PrefixGuidance, rewards, rank_of


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.paths = [[0, 2, 1], [0, 3, 4, 1], [0, 3, 5, 6, 1]]
        self.index = PrefixIndex(self.paths, [1, 2, 3])
        self.reward = np.array([0., -.8, .2, .9])

    def test_telescoping_variable_length(self):
        phi = self.index.potentials(self.reward)
        for item, path in enumerate(self.paths, 1):
            increments = [phi[self.index.lookup[tuple(path[:i+1])]] -
                          phi[self.index.lookup[tuple(path[:i])]] for i in range(1, len(path))]
            self.assertAlmostEqual(sum(increments) / (len(path)-1), self.reward[item])

    def test_global_parent_offset_is_preserved(self):
        guide = PrefixGuidance(self.index, self.reward, .3)
        x = torch.tensor([[0, 2], [0, 3]])
        scores = torch.full((2, 8), -torch.inf)
        scores[0, 1] = -1.; scores[1, 4:6] = -1.
        actual = guide(x, scores.clone())
        self.assertEqual(actual[0, 1], scores[0, 1])
        self.assertNotEqual(actual[1, 4], actual[1, 5])
        self.assertFalse(torch.isfinite(actual[0, 7]))

    def test_real_beam_terminal_score_and_zero_identity(self):
        torch.manual_seed(10)
        model = T5ForConditionalGeneration(T5Config(vocab_size=12, d_model=16, d_ff=32,
            num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)).eval()
        def allowed(batch, prefix):
            node = self.index.lookup.get(tuple(prefix.tolist()))
            return list(self.index.children[node]) if node is not None else []
        args = dict(input_ids=torch.tensor([[7, 8, 1]]), num_beams=3, num_return_sequences=3,
            max_length=5, prefix_allowed_tokens_fn=allowed, output_scores=True,
            return_dict_in_generate=True, length_penalty=1.)
        with torch.no_grad():
            original = model.generate(**args)
            zero = model.generate(**args, logits_processor=LogitsProcessorList([PrefixGuidance(self.index, self.reward, 0)]))
            torch.testing.assert_close(original.sequences, zero.sequences, rtol=0, atol=0)
            torch.testing.assert_close(original.sequences_scores, zero.sequences_scores, rtol=0, atol=0)
            output = model.generate(**args, logits_processor=LogitsProcessorList([PrefixGuidance(self.index, self.reward, .3)]))
            for seq, actual in zip(output.sequences.tolist(), output.sequences_scores):
                path = seq[:seq.index(1)+1]; item = self.index.path_to_item[tuple(path)]
                labels = torch.tensor([path[1:]])
                logits = model(input_ids=args['input_ids'], labels=labels).logits.log_softmax(-1)
                native = logits.gather(-1, labels[..., None]).mean().item()
                self.assertAlmostEqual(actual.item(), native + .3*self.reward[item], places=5)

    def test_shuffle_is_deterministic_and_preserves_bucket_distribution(self):
        rng = np.random.RandomState(7)
        freq = np.r_[0, np.repeat([1, 2, 5, 15], 10)]
        cf = rng.normal(size=(3, len(freq)))
        a, perm = rewards(cf, freq, 'alice'); b, perm2 = rewards(cf, freq, 'alice')
        np.testing.assert_array_equal(perm, perm2)
        np.testing.assert_array_equal(a['shuffled_cf'], b['shuffled_cf'])
        self.assertGreater(np.sum(a['real_cf'] != a['shuffled_cf']), 20)
        buckets = np.floor(np.log2(freq[1:])).astype(int)
        np.testing.assert_array_equal(buckets, buckets[perm])
        np.testing.assert_array_equal(a['shuffled_cf'][1:], a['real_cf'][1:][perm])
        for signal in a.values(): self.assertTrue(np.isfinite(signal).all())

    def test_target_outside_candidates_is_miss(self):
        self.assertEqual(rank_of([1, 2], [.2, .3], 3), 51)
        self.assertEqual(rank_of([1, 2], [.3, .3], 1), 1)


if __name__ == '__main__':
    unittest.main()
