"""Resource slowdown must not bypass correctness or stop a valid smoke run."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from experiment.phase21.expert_resource_recovery import RecoveryMixin, read


class FakeWorker(RecoveryMixin):
    def __init__(self, folder, mismatch=False):
        self.folder = Path(folder); self.development = [0, 1, 2]
        self.samples = [dict(user_id=str(i), history=[1, 2]) for i in self.development]
        self.index = SimpleNamespace(conditional_logs=lambda x: x)
        self.fused = np.ones((3, 2)); self.mismatch = mismatch
        self.references = {str(i): dict(paths=[[1]], native=np.array([1.])) for i in self.development}

    def status(self, **kwargs): pass
    def context(self, j): return j

    def decode(self, ctx, q=None, alpha=0., native_entry=False, record=True):
        return dict(paths=[[2]] if self.mismatch and native_entry else [[1]], scores=np.array([1.]),
                    guide=SimpleNamespace(rescore=lambda *args: np.array([1.])))


class ResourceRecoveryTest(unittest.TestCase):
    def test_slow_valid_run_is_warning_not_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            job = FakeWorker(folder)
            # More than 75 minutes even for this tiny fake sample count.
            with patch('experiment.phase21.expert_resource_recovery.time.monotonic',
                       side_effect=[0, 2000, 3000, 5000, 6000, 8000]):
                job.smoke_control()
            result = read(Path(folder) / 'smoke.json')
            self.assertEqual(result['state'], 'PASSED')
            self.assertEqual(result['warning'], 'SLOW_ESTIMATE')
            self.assertEqual(len(result['checks']), 3)

    def test_correctness_mismatch_still_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AssertionError): FakeWorker(folder, mismatch=True).smoke_control()
            self.assertFalse((Path(folder) / 'smoke.json').exists())


if __name__ == '__main__': unittest.main()
