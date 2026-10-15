"""Resume the fixed 21.19B cohort without changing scientific settings."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiment.phase21 import candidate_gate_generation as G

ROOT = G.ROOT
SOURCE = G.OUT
OUT = ROOT / 'artifacts/phase21/candidate_gate_generation_resume_20261014_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.20_候选权重恢复与双专家简化探索_20261014.md'


def validated_prefix(folder):
    samples = G.read(folder / 'samples.json')
    references = {r['user_id']: r for r in G.read(folder / 'references.json')}
    records = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    assert len(records) <= len(samples)
    for row, s in zip(records, samples):
        assert (row['user_id'], row['target'], row['history']) == (s['user_id'], s['target'], s['history'])
        assert set(row) == {'user_id', 'target', 'history', 'native', 'original', 'user_late', 'user_guided', 'score_late', 'score_guided'}
        assert row['original'] == references[s['user_id']]['guided']
        assert row['native']['ids'] == references[s['user_id']]['native']['ids']
        np.testing.assert_allclose(row['native']['scores'], references[s['user_id']]['native']['scores'], atol=2e-6, rtol=1e-5)
        for key in ('native', 'original', 'user_late', 'user_guided', 'score_late', 'score_guided'):
            assert len(row[key]['ids']) == len(row[key]['scores']) == 50
            assert len(set(row[key]['ids'])) == 50 and np.isfinite(row[key]['scores']).all()
    return len(records)


def prepare():
    old_manifest = G.read(SOURCE / 'manifest.json')['sha256']
    for rel, digest in old_manifest.items():
        assert G.sha(ROOT / rel) == digest, rel
    counts = {d: validated_prefix(SOURCE / d) for d in ('Toys', 'Beauty')}
    OUT.mkdir(exist_ok=False)
    cfg = G.read(SOURCE / 'config.json')
    cfg.update(worker_timeout_seconds=21600, gpus=dict(Toys='GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc', Beauty='GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253'))
    G.write(OUT / 'config.json', cfg)
    for d in counts:
        folder = OUT / d
        folder.mkdir()
        for name in ('samples.json', 'references.json', 'catalog.json', 'data_config.json', 'signals.npz'):
            (folder / name).symlink_to(SOURCE / d / name)
        shutil.copy2(SOURCE / d / 'predictions.jsonl', folder / 'predictions.jsonl')
        G.write(folder / 'status.json', dict(state='PREPARED', reused_users=counts[d], users_total=2048))
    shutil.copy2(PLAN, OUT / 'protocol.md')
    shutil.copy2(__file__, OUT / Path(__file__).name)
    manifest = dict(old_manifest)
    for p in (Path(__file__), PLAN, OUT / 'config.json'):
        manifest[str(p.relative_to(ROOT))] = G.sha(p)
    for d in counts:
        p = SOURCE / d / 'predictions.jsonl'
        manifest[str(p.relative_to(ROOT))] = G.sha(p)
    G.write(OUT / 'manifest.json', dict(sha256=manifest))
    G.write(OUT / 'recovery.json', dict(source=str(SOURCE.relative_to(ROOT)), original_failure='Domain time budget', reused_users=counts, scientific_settings_unchanged=True))
    G.write(OUT / 'status.json', dict(state='PREPARED', reused_users=counts))
    print(counts, flush=True)


class ResumeWorker(G.Worker):
    def run(self):
        count = validated_prefix(self.folder)
        self.users_completed = count
        self.stage = 'resume_parity'
        checks = []
        for j in sorted(set([0, len(self.samples)//2, len(self.samples)-1, min(count, len(self.samples)-1)])):
            ctx = self.context(j)
            native = self.decode(ctx)
            q = self.index.conditional_logs(self.signals['original'][j])
            zero = self.decode(ctx, q, 0.)
            old = self.decode(ctx, q, .3)
            assert native['paths'] == zero['paths']
            np.testing.assert_array_equal(native['scores'], zero['scores'])
            ref = self.reference[self.samples[j]['user_id']]
            for value, key in ((native, 'native'), (old, 'guided')):
                assert value['ids'] == ref[key]['ids'], (j, key)
                np.testing.assert_allclose(value['scores'], ref[key]['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user=self.samples[j]['user_id'], alpha0_exact=True, old_outputs_reproduced=True))
        G.write(self.folder / 'smoke.json', dict(state='PASSED', checks=checks, reused_users=count))
        self.stage = 'matched_new_generation'
        self.status()
        with (self.folder / 'predictions.jsonl').open('a', buffering=1) as f:
            for j in range(count, len(self.samples)):
                s = self.samples[j]
                ctx = self.context(j)
                native = self.decode(ctx)
                ref = self.reference[s['user_id']]['native']
                assert native['ids'] == ref['ids']
                np.testing.assert_allclose(native['scores'], ref['scores'], atol=2e-6, rtol=1e-5)
                row = dict(user_id=s['user_id'], target=s['target'], history=s['history'], native=dict(ids=native['ids'], scores=native['scores'].tolist()), original=self.reference[s['user_id']]['guided'])
                for arm in ('user', 'score'):
                    q = self.index.conditional_logs(self.signals[arm][j])
                    late = native['guide'].rescore(native['paths'], q, .3)
                    guided = self.decode(ctx, q, .3)
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                f.write(json.dumps(row)+'\n')
                self.users_completed = j+1
                if (j+1)%16 == 0 or j+1 == len(self.samples):
                    os.fsync(f.fileno())
                    self.status()
                del ctx, native, guided
        assert validated_prefix(self.folder) == len(self.samples)
        assert self.digest() == self.initial_digest
        self.stage = 'complete'
        self.status('COMPLETED', model_digest_unchanged=True, reused_users=count)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'worker', 'summarize'])
    p.add_argument('domain', nargs='?')
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare()
    else:
        G.OUT = OUT
        if a.mode == 'worker':
            G.Worker = ResumeWorker
            G.worker(a.domain)
        else:
            G.summarize()
