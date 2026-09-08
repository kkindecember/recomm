import unittest

import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList

from experiment.phase18.core.beam_boundary_trace import BoundaryTrace


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def step(self, trace, prefixes, logits):
        scorer = BeamSearchScorer(batch_size=1, num_beams=trace.width, device=torch.device("cpu"))
        ids = torch.tensor(prefixes)
        scores = torch.tensor(logits)
        cumulative = trace.cumulative if trace.cumulative is not None else torch.tensor([0.] + [-1e9] * (trace.width - 1))
        with trace.installed():
            self.assertIs(trace(ids, scores), scores)
            values, indices = (scores + cumulative[:, None]).reshape(1, -1).topk(2 * trace.width)
            scorer.process(ids, values, indices % scores.shape[-1], indices // scores.shape[-1],
                           pad_token_id=0, eos_token_id=1)
        return trace.steps[-1]

    def test_local_first_single_child_loses_to_other_parent(self):
        trace = BoundaryTrace([2, 4, 1], width=2)
        trace.valid = [True, True]
        trace.cumulative = torch.tensor([-10., -1.])
        s = self.step(trace, [[0, 2], [0, 3]],
                      [[-float("inf")] * 4 + [0., -float("inf")],
                       [-float("inf")] * 4 + [-1., -2.]])
        self.assertEqual(s["target"]["local_rank_strict"], 1)
        self.assertEqual(s["target"]["legal_child_count"], 1)
        self.assertFalse(s["target"]["retained_active"])
        self.assertEqual(trace.first_drop, 0)
        self.assertEqual(s["boundary"]["prefix"], [3, 5])

    def test_initial_invalid_beams_and_ties(self):
        s = self.step(BoundaryTrace([2, 1], width=2), [[0], [0]],
                      [[-float("inf"), -float("inf"), -1., -1.]] * 2)
        self.assertEqual(s["input_valid"], [True, False])
        self.assertEqual(s["legal_extensions"], 2)
        self.assertEqual(s["target"]["global_tied"], 2)
        self.assertEqual(s["target"]["local_rank_strict"], 1)
        self.assertTrue(s["target"]["retained_active"])
        self.assertEqual(sum(c["valid"] for c in s["top2b"]), 2)

    def test_eos_is_added_to_heap_separately(self):
        t = BoundaryTrace([2, 1], width=2)
        t.valid = [True, True]
        t.cumulative = torch.tensor([-1., -2.])
        s = self.step(t, [[0, 2], [0, 3]],
                      [[-float("inf"), -0.1, -3., -4.], [-float("inf"), -5., -1., -2.]])
        self.assertTrue(s["target"]["eos_offered"])
        self.assertFalse(s["target"]["retained_active"])
        self.assertIsNone(t.first_drop)
        self.assertEqual(s["eos_heap_after"][0]["prefix"], [2])

    def test_hooks_restore_after_exception(self):
        original = BeamSearchScorer.process
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with BoundaryTrace([2, 1], width=2).installed():
                raise RuntimeError("forced")
        self.assertIs(BeamSearchScorer.process, original)

    def test_native_generation_identity_variable_lengths(self):
        torch.manual_seed(2023)
        model = T5ForConditionalGeneration(T5Config(vocab_size=12, d_model=16, d_ff=32,
                   num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
                   decoder_start_token_id=0, eos_token_id=1, pad_token_id=0)).eval()
        paths = [[0, 2, 1], [0, 3, 4, 1], [0, 3, 5, 1], [0, 6, 7, 8, 1]]
        def allowed(batch_id, sentence):
            p = sentence.tolist()
            return sorted({s[len(p)] for s in paths if len(s) > len(p) and s[:len(p)] == p})
        args = dict(input_ids=torch.tensor([[4, 5, 1]]), num_beams=2, num_return_sequences=2,
                    max_length=5, prefix_allowed_tokens_fn=allowed,
                    return_dict_in_generate=True, output_scores=True)
        original = model.generate(**args)
        trace = BoundaryTrace([3, 4, 1], width=2)
        with trace.installed():
            traced = model.generate(**args, logits_processor=LogitsProcessorList([trace]))
        self.assertTrue(torch.equal(original.sequences, traced.sequences))
        self.assertTrue(torch.equal(original.sequences_scores, traced.sequences_scores))
        self.assertEqual(traced.sequences.tolist(), trace.final["sequences"])
        self.assertTrue(all(s["native_reconstruction_exact"] for s in trace.steps))
        self.assertTrue(any(s["eos_offered"] for s in trace.steps))


if __name__ == "__main__":
    unittest.main()
