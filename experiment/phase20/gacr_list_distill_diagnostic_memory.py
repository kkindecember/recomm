"""Q0 resource revision: original chunk4 arithmetic, bounded CUDA allocator."""
import sys
from pathlib import Path
from experiment.phase20 import gacr_list_distill_diagnostic as base

base.OUTPUT=base.ROOT/'artifacts/phase20/gacr_list_distill_toys_q0_memory_v3'


def main():
    if '--prepare' in sys.argv:
        previous=base.read(base.ROOT/'artifacts/phase20/gacr_list_distill_toys_q0_compact_v2/status.json')
        assert previous['state']=='FAILED' and 'parent_score_parity' in previous['error']
        base.prepare()
        path=base.OUTPUT/'config.json';config=base.read(path)
        config.update(minimum_free_mib=15000,memory_fraction=.25,candidate_chunk=4,
            resource_revision='Restore original chunk4 after chunk2 score parity failed at 7.51e-6. Keep 15000 MiB admission and 25% allocator cap; same samples and original 1e-6 tolerance; no optimizer.')
        p=Path(__file__).resolve();config['frozen_hashes'][str(p.relative_to(base.ROOT))]=base.sha256(p)
        base.write(path,config)
    else:base.main()


if __name__=='__main__':main()
