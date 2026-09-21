"""Range corruption protection and reuse checks against all completed models."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from experiment.phase20.guided_decoding_job import ROOT, read, write, now
from experiment.phase20.lime_comparison_recovery import validate_range_response, check_file, ORIGINAL, CONFIG
from experiment.phase20.lime_comparison_worker import Worker, ItemCFExpert, arrays
from experiment.phase20.lime_comparison_core import histories_from_bank
from experiment.phase20 import lime_comparison_recovery as recovery


class TestRecovery(unittest.TestCase):
    def test_ranges_reject_wrong_start_and_full_response(self):
        header = 'HTTP/1.1 206 Partial Content\nContent-Range: bytes 10-19/30\n'
        validate_range_response(header, 10, 10, 19, 30)
        validate_range_response(header, 4, 10, 19, 30)  # interrupted but correctly aligned prefix
        for h, size in [('HTTP/1.1 200 OK\n', 30), (header, 11),
                         (header.replace('10-19', '0-9'), 10), (header, 0)]:
            with self.assertRaises(AssertionError): validate_range_response(h, size, 10, 19, 30)

    def test_complete_hash_rejects_same_length_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'weights'; p.write_bytes(b'complete')
            entry = dict(size=8, lfs=dict(oid=hashlib.sha256(b'complete').hexdigest()))
            check_file(p, entry)
            p.write_bytes(b'corrupt!')
            with self.assertRaises(AssertionError): check_file(p, entry)

    def test_second_recovery_resumes_latest_prefix_and_preserves_first_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder)/'run'
            out = original/'recovery_v2'; out.mkdir(parents=True)
            source = original/'recovery_v1/assets'
            model = source/'bge-base-en-v1.5'; model.mkdir(parents=True)
            complete = b'0123456789abcdefghijklmnopqrstuvwxyz'
            prefix = complete[:23]
            (model/'pytorch_model.bin.part').write_bytes(prefix)
            files = {
                'pytorch_model.bin': complete,
                '1_Pooling/config.json': json.dumps({'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False}).encode(),
                'modules.json': json.dumps([{'type': 'sentence_transformers.models.'+x} for x in ['Transformer', 'Pooling', 'Normalize']]).encode(),
                'sentence_bert_config.json': b'{"max_seq_length": 512}',
            }
            entries = []
            for name, content in files.items():
                entries.append(dict(path=name, size=len(content), lfs=dict(oid=hashlib.sha256(content).hexdigest())))
                if name != 'pytorch_model.bin':
                    target = model/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
            write(source/'huggingface_tree.json', entries)
            ranges = []
            def fetch(args, **kwargs):
                start, end = map(int, args[args.index('--range')+1].split('-'))
                ranges.append((start, end))
                Path(args[args.index('--dump-header')+1]).write_text(f'HTTP/1.1 206 Partial Content\nContent-Range: bytes {start}-{end}/{len(complete)}\n')
                Path(args[args.index('--output')+1]).write_bytes(complete[start:end+1])
                return SimpleNamespace(returncode=0, poll=lambda: 0)
            with patch.object(recovery, 'ORIGINAL', original), patch.object(recovery, 'OUT', out), \
                 patch.object(recovery, 'ROOT', Path(folder)), patch.object(recovery.subprocess, 'Popen', side_effect=fetch):
                recovery.download()
            self.assertEqual(ranges, [(len(prefix), len(complete)-1)])
            self.assertEqual((model/'pytorch_model.bin.part').read_bytes(), prefix)
            self.assertEqual((out/'assets/bge-base-en-v1.5/pytorch_model.bin').read_bytes(), complete)
            self.assertEqual(read(out/'assets/manifest.json')['resumed_bytes'], len(prefix))
            self.assertEqual(read(out/'assets/status.json')['state'], 'COMPLETED')

    def test_all_six_real_checkpoints_reused_without_optimizer(self):
        for domain, vocab in [('Toys', 11924), ('Beauty', 12101)]:
            with tempfile.TemporaryDirectory() as folder:
                w = Worker.__new__(Worker)
                w.c = read(CONFIG); w.resume_from = ORIGINAL/domain
                w.out = Path(folder); w.inputs = {}; w.completed_seeds = []; w.timings = {}; w.v = vocab
                w.status = lambda *a, **kw: None
                with patch('experiment.phase20.lime_comparison_worker.train_sasrec', side_effect=AssertionError('must not retrain')):
                    w.train()
                self.assertEqual(w.completed_seeds, [0, 1, 2])
                self.assertEqual(read(w.out/'reused_training.json')['new_sasrec_optimizer_steps'], 0)

    def test_both_completed_itemcf_caches_match_author(self):
        for d in ['Toys', 'Beauty']:
            q = arrays(ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{d}/queries.npz')
            bank = arrays(ROOT/f'artifacts/phase20/nearest_comparison_20260918/{d}.train_sequences.npz')
            histories = histories_from_bank(bank, q['users'])
            import json
            records = [json.loads(line) for line in (ROOT/f'artifacts/phase20/nearest_comparison_20260918/{d}.items.jsonl').open()]
            ids = [r['source_item_id'] for r in records]
            cf = ItemCFExpert(str(ORIGINAL/d/'itemcf.json'), ids)
            scores = np.load(ORIGINAL/d/'itemcf_scores.npy', mmap_mode='r')
            self.assertEqual(scores.shape, (len(q['users']), len(ids)))
            for i in np.linspace(0, len(histories)-1, 32, dtype=int):
                expected = cf.score(str(q['users'][i]), [ids[j-1] for j in histories[i]])
                np.testing.assert_array_equal(scores[i], expected)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestRecovery))
    write(ORIGINAL/'recovery_tests.json', dict(state='PASSED' if result.wasSuccessful() else 'FAILED',
        updated_at=now(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
