import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from experiment.phase21.night_resume_core import prediction_prefix, process_identity, identity_alive


def sample(j):
    return dict(user_id=str(j),target=2,history=[1])


def row(j):
    value=dict(ids=[1,2,3],scores=[.3,.2,.1])
    return dict(sample(j),sample_index=j,native=value,late=value,guided=value)


class PrefixTest(unittest.TestCase):
    def test_torn_tail_preserves_complete_rows_and_archives_fragment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'predictions.jsonl'; good=(json.dumps(row(0))+'\n').encode()
            p.write_bytes(good+b'{"user_id":')
            got=prediction_prefix(p,[(j,sample(j)) for j in range(2)],repair=True)
            self.assertEqual(got,[row(0)]); self.assertEqual(p.read_bytes(),good)
            self.assertEqual(next(Path(tmp).glob('*.bin')).read_bytes(),b'{"user_id":')

    def test_valid_final_record_without_newline_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p'; p.write_text(json.dumps(row(0)))
            self.assertEqual(len(prediction_prefix(p,[(0,sample(0))],repair=True)),1)
            self.assertTrue(p.read_bytes().endswith(b'\n'))

    def test_wrong_user_and_corrupt_complete_line_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p'; p.write_text(json.dumps(row(1))+'\n')
            with self.assertRaises(AssertionError): prediction_prefix(p,[(0,sample(0))],repair=True)
            p.write_bytes(b'{bad}\n')
            with self.assertRaises(ValueError): prediction_prefix(p,[(0,sample(0))],repair=True)

    def test_boot_and_start_identity_prevent_pid_reuse(self):
        ident=process_identity(os.getpid()); self.assertTrue(identity_alive(ident))
        self.assertFalse(identity_alive(dict(ident,boot='old-boot')))
        self.assertFalse(identity_alive(dict(ident,start='0')))

    def test_generation_only_decodes_unsaved_suffix(self):
        from experiment.phase21 import night_run as app
        class Fake(app.ContinueGeneration):
            def fit_signals(self): self.fit_steps=12
            def smoke_control(self): pass
            def status(self,**kw): pass
            def digest(self): return 'unchanged'
            def context(self,j): return j
            def decode(self,ctx,*args):
                self.completed+=1
                class Guide:
                    def rescore(self,*args): return np.array([.3,.2,.1])
                return dict(ids=[1,2,3],paths=[[1],[2],[3]],scores=np.array([.3,.2,.1]),guide=Guide())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); w=Fake(); w.folder=root; w.arm='C'; w.domain='Toys'; w.completed=0
            w.samples=[sample(j) for j in range(512)]; w.development=list(range(512))
            w.references={str(j):dict(ids=[1,2,3],paths=[[1],[2],[3]],native=[.3,.2,.1]) for j in range(512)}
            w.initial_digest='unchanged'; w.started=time.monotonic(); w.fused=np.ones((512,3),dtype=np.float32)
            class Index:
                def conditional_logs(self,x): return x
            w.index=Index(); (root/'gate.pt').write_bytes(b'fixed'); (root/'fit_checks.json').write_text('{}')
            np.save(root/'fused.npy',w.fused)
            prefix=''.join(json.dumps(row(j))+'\n' for j in range(2)); (root/'predictions.jsonl').write_text(prefix)
            def verify(*args):
                np.savez(root/'ranks.npz',ranks=[1]); return dict(state='COMPLETED')
            with patch.object(app.runner,'verify_domain',side_effect=verify): w.run()
            records=prediction_prefix(root/'predictions.jsonl',list(enumerate(w.samples)))
            self.assertEqual(len(records),512); self.assertEqual(w.completed,2*(512-2))
            self.assertEqual((root/'predictions.jsonl').read_text()[:len(prefix)],prefix)
            self.assertEqual(app.read(root/'summary.json')['resumed_users'],2)


if __name__=='__main__': unittest.main()
