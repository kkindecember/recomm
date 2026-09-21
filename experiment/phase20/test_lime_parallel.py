import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiment.phase20.lime_parallel_core import local_batches, sum_gradients


def gradient_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+root+'/init', rank=rank, world_size=2)
    torch.manual_seed(77)
    model = torch.nn.Linear(3, 2).double()
    reference = torch.nn.Linear(3, 2).double()
    reference.load_state_dict(model.state_dict())
    for n in (8, 3, 1):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        for j in range(n):
            x = torch.arange((j+1)*3, dtype=torch.double).reshape(j+1, 3)/10
            reference(x).square().mean().div(n).backward()
            if j % 2 == rank:
                model(x).square().mean().div(n).backward()
        sum_gradients(model, 2)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)
    dist.destroy_process_group()


class ParallelTest(unittest.TestCase):
    def test_partition_including_partial_update(self):
        for n in range(1, 10):
            batches = [[i] for i in range(n)]
            for world in (1, 2, 4):
                actual = [x[0] for r in range(world) for x in local_batches(batches, r, world)]
                self.assertEqual(sorted(actual), list(range(n)))

    def test_distributed_matches_global_gradient(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(gradient_worker, args=(root,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
