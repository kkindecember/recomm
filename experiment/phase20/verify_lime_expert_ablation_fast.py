"""Run the unchanged independent auditor with eager, once-only NPZ reads.

The original auditor is frozen and remains on disk. Its mathematics, sorting,
hash checks and assertions run unchanged; only its numpy.load access is adapted.
"""
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from experiment.phase20 import verify_lime_expert_ablation as original

OUT = original.OUT


def write_progress(details):
    path = OUT / 'audit_progress.json'
    temporary = path.with_name(f'audit_progress.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(details, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')),
                                    ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class EagerNumpy:
    """Delegate all numerical operations, eagerly unpack only the two inputs."""
    def __init__(self, progress):
        self.progress = progress

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path, **kwargs):
        path = Path(path)
        assert path.name == 'inputs.npz' and path.parent.parent == OUT
        self.progress.update(stage='loading_arrays_once', domain=path.parent.name)
        with np.load(path, **kwargs) as archive:
            arrays = {name: archive[name] for name in archive.files}
        self.progress.update(stage='reconstructing_scores_and_ranks', domain=path.parent.name,
            arrays_loaded=len(arrays), array_bytes=sum(a.nbytes for a in arrays.values()))
        return arrays


def main():
    begin = time.monotonic()
    progress = dict(state='VERIFYING', stage='checking_frozen_hashes', pid=os.getpid(),
                    io_fix='Materialize each NPZ member once instead of on every subscript.')
    stop = threading.Event()
    def heartbeat():
        while not stop.is_set():
            write_progress(dict(progress, elapsed_seconds=time.monotonic() - begin))
            stop.wait(5)
    thread = threading.Thread(target=heartbeat, daemon=True)
    previous_numpy = original.np
    original.np = EagerNumpy(progress)
    thread.start()
    try:
        original.main()
        progress.update(state='VERIFIED', stage='done')
    except Exception:
        progress.update(state='FAILED', stage='audit_error')
        raise
    finally:
        original.np = previous_numpy
        stop.set(); thread.join()
        write_progress(dict(progress, elapsed_seconds=time.monotonic() - begin))


if __name__ == '__main__':
    main()
