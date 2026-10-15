"""Move only the waiting CPU follow-up to a detached process; keep GPU jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
out = root/'artifacts/phase20/lime_partial_state_20260922_v1'
old_pid = int(sys.argv[1])
command = Path(f'/proc/{old_pid}/cmdline').read_bytes().split(b'\0')
assert command[1:-1] == [b'-u', b'-m', b'experiment.phase20.finish_lime_partial_state'], command[:4]
previous = json.loads((out/'cpu_pipeline_status.json').read_text())
assert previous['stage'] == 'cache_and_feature_review' and not previous['feature_review_completed']
assert all(json.loads((out/d/'status.json').read_text())['state'] == 'RUNNING' for d in ('Toys','Beauty'))
(out/'background_handoff.json').write_text(json.dumps(dict(previous_status=previous, previous_host_pid=old_pid,
    gpu_jobs_unchanged=True, preference='Runs over 10 minutes continue in background; inspect status files without assistant polling.'),indent=2)+'\n')
os.kill(old_pid, signal.SIGTERM)
with (out/'cpu_background.log').open('x') as log:
    proc = subprocess.Popen([sys.executable,'-u','-m','experiment.phase20.finish_lime_partial_state'],
        cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'cpu_background_launch.json').write_text(json.dumps(dict(pid=proc.pid,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),detached=True),indent=2)+'\n')
print(json.dumps(dict(cpu_background_pid=proc.pid,gpu_jobs_unchanged=True)))
