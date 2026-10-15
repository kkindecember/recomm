"""Monitor this run, audit each finished cache, then fit and verify on CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase20/lime_partial_state_20260922_v1'


def read(path):
    for _ in range(5):
        try: return json.loads(path.read_text())
        except json.JSONDecodeError: time.sleep(.05)
    raise RuntimeError(str(path))


def status(**kw):
    (OUT/'cpu_pipeline_status.json').write_text(json.dumps(dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        pid=os.getpid(), **kw), indent=2)+'\n')


def main():
    started = time.monotonic(); reviews = {}; completed = set()
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    while len(completed) < 2 or read(OUT/'status.json')['state'] != 'CACHE_COMPLETED':
        top = read(OUT/'status.json'); assert top['state'] not in ('FAILED', 'COMPLETED'), top
        progress = {}
        for domain in ('Toys', 'Beauty'):
            s = read(OUT/domain/'status.json'); assert s['state'] != 'FAILED', s
            progress[domain] = {k:s.get(k) for k in ('state','users_completed','users_total')}
            if s['state'] == 'COMPLETED' and domain not in reviews:
                with (OUT/domain/'feature_review.log').open('x') as log:
                    reviews[domain] = subprocess.Popen([sys.executable,'-u','-m','experiment.phase20.review_lime_partial_state','features','--domain',domain],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if domain in reviews and reviews[domain].poll() is not None:
                assert reviews[domain].returncode == 0, (domain, reviews[domain].returncode)
                completed.add(domain)
        status(state='RUNNING', stage='cache_and_feature_review', progress=progress, feature_review_completed=sorted(completed))
        if time.monotonic()-started > 3300: raise TimeoutError('CPU follow-up wait/review budget')
        time.sleep(15)
    # Check the CPU scientific code was frozen while raw generation was ongoing.
    import hashlib
    for rel, digest in read(OUT/'analysis_code_frozen.json')['sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    for name, args in [('analysis',['experiment.phase20.analyze_lime_partial_state']),
                       ('fit_review',['experiment.phase20.review_lime_partial_state','fits']),
                       ('report',['experiment.phase20.summarize_lime_partial_state'])]:
        status(state='RUNNING',stage=name)
        with (OUT/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    status(state='COMPLETED',seconds=time.monotonic()-started)
    print(json.dumps(dict(state='COMPLETED',output=str(OUT))),flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as e:
        (OUT/'cpu_pipeline_failure.txt').write_text(traceback.format_exc())
        status(state='FAILED',error=repr(e)); raise
