"""Meaningful isolation and author-parity checks before the GPU run."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
import torch

from experiment.phase20.guided_decoding_job import ROOT, read, write, now
from experiment.phase20.lime_comparison_core import gate_folds, histories_from_bank, semantic_users
from experiment.phase20.catalog_scope_core import ranks
sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
from lime_rec.itemcf import build_itemcf_model
from lime_rec.experts import ItemCFExpert, LLMSemanticExpert
from lime_rec.evaluation import normalize_score_rows
from lime_rec.data import _read_metadata
from scripts.data.download_gram_data import convert_meta
from experiment.phase20.lime_comparison_worker import Worker


class TestLimeComparison(unittest.TestCase):
    def test_actual_author_metadata_covers_both_catalogs(self):
        for domain in ('Toys', 'Beauty'):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)/'meta.jsonl'
                convert_meta(ROOT/f'GRAM/rec_datasets/{domain}/item_plain_text.txt', p)
                text = _read_metadata(p)
                records = [json.loads(line) for line in (ROOT/f'artifacts/phase20/nearest_comparison_20260918/{domain}.items.jsonl').open()]
                self.assertTrue(all(text.get(r['source_item_id']) for r in records))

    def test_histories_and_disjoint_folds(self):
        bank = dict(users=np.array(['b', 'a']), offsets=np.array([0, 3, 5]), items=np.array([1, 3, 2, 4, 5]))
        self.assertEqual(histories_from_bank(bank, ['a', 'b']), [[4, 5], [1, 3, 2]])
        users = [f'u{i}' for i in range(20)]
        folds = gate_folds(users, np.arange(3, 20))
        self.assertEqual(sorted(np.concatenate(folds)), list(range(3, 20)))
        for predicted in folds:
            fit = np.setdiff1d(np.arange(20), predicted)
            self.assertFalse(set(predicted) & set(fit))
            self.assertTrue({0, 1, 2}.issubset(fit))

    def test_author_itemcf_train_only_formula(self):
        histories = {'x': ['a', 'b', 'c'], 'y': ['a', 'b']}
        model = build_itemcf_model(dataset_name='synthetic', histories=histories,
            item_ids=['a', 'b', 'c', 'heldout'], top_k=100, window_size=20)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'cf.json'; p.write_text(json.dumps(model))
            cf = ItemCFExpert(str(p), ['a', 'b', 'c', 'heldout'])
            result = cf.score('x', ['a', 'b'])
            expected = [1., .5, 1.5/np.sqrt(2), 0.]
            np.testing.assert_allclose(result, expected, atol=1e-7)
            self.assertTrue(model['training_only'])

    def test_semantic_and_normalization_match_author(self):
        rng = np.random.RandomState(4)
        embedding = rng.normal(size=(80, 12)).astype(np.float32)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        histories = [[1, 2, 8], list(range(1, 70)), [3]]
        ids = [str(i) for i in range(1, 81)]
        author = LLMSemanticExpert(ids, {}, embeddings=embedding)
        actual = semantic_users(histories, embedding) @ author.item_embeddings_norm.T
        for j, h in enumerate(histories):
            np.testing.assert_allclose(actual[j], author.score('', list(map(str, h))), atol=2e-7)
        raw = np.array([[2, 2, 2], [0, 4, 2]], dtype=np.float32)
        np.testing.assert_allclose(normalize_score_rows(raw), [[0, 0, 0], [0, 1, .5]], atol=1e-8)

    def test_full_gate_prediction_and_common_score_identity(self):
        torch.set_num_threads(2)
        rng = np.random.RandomState(8)
        w = Worker.__new__(Worker)
        w.c = read(ROOT/'experiment/phase20/lime_comparison_20260920_v1.json')
        w.c['score_batch'] = 4
        w.n, w.v, w.device = 16, 80, torch.device('cpu')
        w.users = np.array([f'user{i:03d}' for i in range(w.n)])
        w.histories = [[i%10+1, 11] for i in range(w.n)]
        w.targets = np.arange(40, 56)
        w.eval = np.arange(4, 16)
        w.q = dict(history=np.array(w.histories), candidate_ids=np.tile(np.arange(20, 70), (w.n, 1)))
        w.directory = rng.normal(size=(w.n, w.v+1))
        w.collaborative = rng.normal(size=(w.n, w.v+1))
        w.weight = .9
        w.status = lambda *a, **kw: None
        experts = [rng.normal(size=(w.n, w.v)).astype('float32') for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            w.out = Path(tmp)
            gate = w.fit_gate(0, 'synthetic', np.arange(4), w.eval, experts)
            result = w.new_results()
            w.evaluate_gate(0, gate, w.eval, experts, result, {int(i): j for j, i in enumerate(w.eval)})
            for rr in result['ranks'].values(): self.assertTrue((rr > 0).all())
            common = .1*w.directory+.9*w.collaborative
            expected = []
            for i in w.eval:
                ids = [j for j in w.q['candidate_ids'][i] if j not in w.histories[i]]
                ordered = sorted(ids, key=lambda j: (-common[i, j], j))
                expected.append(ordered.index(w.targets[i])+1)
            np.testing.assert_array_equal(result['ranks']['common_G50'], expected)
            saved = torch.load(w.out/'gate_seed0_synthetic.pt')
            self.assertFalse(set(saved['fit_indices']) & set(saved['predicted_indices']))
            # Fail closed on accidentally fitting the evaluation user's label.
            with self.assertRaises(AssertionError):
                w.fit_gate(0, 'invalid', [4, 5], [5, 6], experts)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestLimeComparison))
    out = ROOT/'artifacts/phase20/lime_comparison_20260920_v1'
    write(out/'tests.json', dict(state='PASSED' if result.wasSuccessful() else 'FAILED',
        updated_at=now(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
    sys.exit(0 if result.wasSuccessful() else 1)
