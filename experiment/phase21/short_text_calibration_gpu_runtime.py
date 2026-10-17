"""GPU entrypoint: let the existing GRAM worker initialize threads once."""
import argparse
import json
import signal
import sys
import traceback

import torch
from experiment.phase21 import short_text_calibration as S

MODULE = 'experiment.phase21.short_text_calibration_gpu_runtime'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('worker','generate','thread-probe'))
    parser.add_argument('--job',choices=S.JOBS)
    parser.add_argument('--lease-pid',type=int)
    args = parser.parse_args()
    def interrupted(signum,frame):
        raise InterruptedError('Short calibration GPU worker interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    if args.mode=='thread-probe':
        # Reproduce the GRAM constructor's initialization in a fresh process.
        # No earlier main() has called set_num_interop_threads here.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        print(json.dumps(dict(state='PASSED',interop_initialized_once=True)))
        return 0
    try:
        if args.mode=='worker':
            S.F.OUT,S.F.CFG,S.F.MODULE,S.F.JOBS = S.OUT,S.CFG,MODULE,S.JOBS
            return S.F.worker(args.job)
        # S.Generation -> CLEAN.Worker sets intra/inter-op threads itself.
        # Avoid S.main(), which also sets inter-op threads for CPU modes.
        S.Generation(args.job,args.lease_pid).run()
    except BaseException as exc:
        p = S.OUT/'jobs'/args.job
        value = dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=S.stamp(),test_read=False)
        S.write(p/'failure.json',value)
        S.write(p/'status.json',value)
        raise
    return 0


if __name__=='__main__':
    sys.exit(main())
