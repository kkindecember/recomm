"""Exercise scheduler concurrency and domain failure isolation without GPUs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiment.phase20.guided_decoding_job import ROOT, CONFIG, read, write
from experiment.phase20.guided_decoding_parallel import supervise_parallel


class SchedulerTests(unittest.TestCase):
    def simulate(self,fail_toys=False):
        config=read(CONFIG); launches=[]; processes=[]
        with TemporaryDirectory(prefix='guided_scheduler_test_',dir=ROOT/'artifacts') as directory:
            out=Path(directory)
            def prepare(c,o):
                for d in c['domains']: (o/d).mkdir()
            class FakeProcess:
                def __init__(self,command,**kwargs):
                    self.domain=command[command.index('--domain')+1]; self.mode=command[command.index('--mode')+1]
                    self.pid=900000+len(launches); self.returncode=None; self.polls=0
                    self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
                    assert all(p.gpu!=self.gpu for p in processes if p.returncode is None)
                    launches.append((self.domain,self.mode)); processes.append(self)
                    if self.mode=='smoke': write(out/self.domain/'smoke.json',dict(projected_domain_seconds=100,peak_allocated_mib=3000))
                    else: write(out/self.domain/'summary.json',dict(metrics={a+'.softmax':dict(N10=.1,H10=.2) for a in ('vanilla','real_cf','shuffled_cf','popularity')}))
                def poll(self):
                    self.polls+=1
                    if self.polls>=2:
                        self.returncode=1 if fail_toys and self.domain=='Toys' else 0
                    return self.returncode
            with patch('experiment.phase20.guided_decoding_parallel.prepare',prepare), \
                 patch('experiment.phase20.guided_decoding_parallel.verify',lambda _:None), \
                 patch('experiment.phase20.guided_decoding_parallel.gpu_candidates',lambda:[dict(uuid='GPU-A',free_mib=14000),dict(uuid='GPU-B',free_mib=12000)]), \
                 patch('experiment.phase20.guided_decoding_parallel.subprocess.Popen',FakeProcess), \
                 patch('experiment.phase20.guided_decoding_parallel.time.sleep',lambda _:None):
                supervise_parallel(config,out)
            result=json.loads((out/'status.json').read_text())
            self.assertEqual(launches[:2],[('Toys','smoke'),('Beauty','smoke')])
            self.assertEqual(result['domains']['Beauty']['state'],'COMPLETED')
            if fail_toys:
                self.assertEqual(result['state'],'FINISHED_WITH_FAILURES')
                self.assertNotIn(('Toys','run'),launches)
                self.assertEqual(result['domains']['Toys']['state'],'FAILED')
                self.assertEqual(list(read(out/'summary.json')['domains']),['Beauty'])
            else:
                self.assertEqual(result['state'],'COMPLETED')
                self.assertEqual(len(launches),4)

    def test_two_domains_use_distinct_gpus(self): self.simulate()
    def test_failed_domain_does_not_cancel_other_domain(self): self.simulate(fail_toys=True)


if __name__=='__main__': unittest.main()
