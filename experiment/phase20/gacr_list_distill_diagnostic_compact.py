"""Resource-only Q0 revision: candidate chunks of 2, CPU gradient accumulation."""
import sys
from pathlib import Path

from experiment.phase20 import gacr_list_distill_diagnostic as base

base.OUTPUT=base.ROOT/'artifacts/phase20/gacr_list_distill_toys_q0_compact_v2'


class CompactDiagnosis(base.Diagnosis):
    def setup(self):
        super().setup()
        # Chunking changes peak memory only: all 50 scores still enter each KL.
        # The unchanged profile checks same cached parent scores and original loss.
        self.work.settings['candidate_chunk']=2
        base.write(base.OUTPUT/'resource_contract.json',dict(
            candidate_chunk=2,full_list_size=50,minimum_free_mib=15000,memory_fraction=.25,
            optimizer_steps=0,parent_parity_atol=1e-6,
            previous_wait_only_attempt='artifacts/phase20/gacr_list_distill_toys_q0_v1'))


def main():
    if '--prepare' in sys.argv:
        previous=base.read(base.ROOT/'artifacts/phase20/gacr_list_distill_toys_q0_v1/status.json')
        assert previous['state']=='STOPPED' and previous['stage']=='post_cuda_resource_check'
        base.prepare()
        path=base.OUTPUT/'config.json';config=base.read(path)
        config.update(minimum_free_mib=15000,memory_fraction=.25,candidate_chunk=2,
            resource_revision='Prior attempt stopped while waiting, no model loaded. Q0 chunk2; 15000 MiB admission; 25% allocator cap; same 512 samples, 32-gradient panel, complete-50 loss and parity checks.')
        for p in (Path(__file__).resolve(),Path(__file__).with_name('run_gacr_list_distill_diagnostic_compact.sh')):
            config['frozen_hashes'][str(p.relative_to(base.ROOT))]=base.sha256(p)
        base.write(path,config)
    else:
        base.Diagnosis=CompactDiagnosis
        base.main()


if __name__=='__main__':main()
