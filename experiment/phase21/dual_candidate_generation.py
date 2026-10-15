"""Fixed seed0 / 512-user true-generation comparison after stage 21.22.

Reuse the established GRAM decoder and AGRec-inspired probability mixture.
SASRec/BGE and frozen fusion are LIME-Rec-derived; no new fitting here.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment.phase21 import dual_candidate_gate as D
from experiment.phase21 import candidate_gate_generation as G
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = D.ROOT
OUT = ROOT/'artifacts/phase21/dual_candidate_generation_20261014_v1'
PLAN = ROOT/'plan/第二十一阶段/21.23_双专家候选权重真实生成_20261014.md'
REPORT = ROOT/'report/第二十一阶段/双专家候选权重真实生成_20261014.md'
ARMS = ('baseline', 'user', 'candidate', 'scalar')
DOMAINS = D.DOMAINS
read, write, sha, stamp = D.read, D.write, D.sha, D.stamp
CFG = dict(evaluation_users=512, head_seed=0, alpha=.3, score_scale=20., beam_size=50,
           length_penalty=1., optimizer_steps=0, test_read=False,
           worker_timeout_seconds=21600, resource_wait_seconds=86400,
           total_lease_mib=30720, expected_workload_peak_mib=12288,
           minimum_free_mib=32768, allocation_limit_gib=11, max_parallel=2,
           gpu_pool='all visible GPUs; free-memory admission',
           arms=list(ARMS), replay_original_dual=True, scientific_auto_retry=False,
           seed_selection='fixed seed0, not best validation seed',
           cohort='same historical512 development users per domain; exploratory')


def record_rank(value, sample):
    ids, scores = value['ids'], value['scores']
    assert len(ids) == len(scores) == len(set(ids)) == 50
    assert np.isfinite(scores).all()
    history = set(sample['history'])
    order = sorted([(i, s) for i, s in zip(ids, scores) if i not in history], key=lambda v: (-v[1], v[0]))
    return next((j+1 for j, (i, _) in enumerate(order) if i == sample['target']), 2**30)


def prepare_domain(domain, use):
    import torch
    folder = OUT/domain; folder.mkdir()
    old = D.DUAL/domain; screen = D.OUT/domain
    old_samples = read(old/'samples.json')
    selected = [j for j, s in enumerate(old_samples) if s['role'] == 'development']
    full = read(D.FULL/domain/'samples.json'); full_by = {s['user_id']: j for j, s in enumerate(full)}
    chosen = []; indices = []
    for j in selected:
        s = old_samples[j]; k = full_by[s['user_id']]; t = full[k]
        assert (s['target'], s['history'], s['full_history']) == (t['target'], t['history'], t['full_history'])
        assert t['previously_evaluated']
        chosen.append(dict(s, validation_index=len(chosen), old_dual_index=j, full_validation_index=k))
        indices.append(k)
    assert len(chosen) == len({s['user_id'] for s in chosen}) == 512
    ix = np.asarray(indices)
    write(folder/'samples.json', chosen)
    for name in ('data_config.json', 'catalog.json'):
        shutil.copy2(old/name, folder/name)
    dc = read(folder/'data_config.json'); use(ROOT/dc['parent_checkpoint'])
    assert sha(ROOT/dc['parent_checkpoint']) == dc['parent_sha256']
    assert read(D.SOURCE/domain/'data_config.json')['parent_sha256'] == dc['parent_sha256']
    old_rr = np.load(old/'ranks.npz')
    np.testing.assert_array_equal(old_rr['users'], [s['user_id'] for s in chosen])
    three_rr = np.load(D.SOURCE/domain/'ranks.npz')
    np.testing.assert_array_equal(three_rr['development_users'], old_rr['users'])
    np.testing.assert_array_equal(three_rr['dev_GRAM'], old_rr['GRAM'])
    # Apply frozen heads to cached inputs; all candidate proposals were target-free.
    data = D.Data.__new__(D.Data)
    keys = ('user', 'candidate', 'mask', 'ids', 'scores', 'weights', 'fused')
    data.a = {k: np.load(screen/'validation'/(k+'.npy'), mmap_mode='r') for k in keys}
    saved = np.load(screen/'ranks.npz')
    baseline = np.array(data.a['fused'][ix], copy=True)
    legacy = np.load(old/'fused.npy', mmap_mode='r')[selected]
    np.testing.assert_allclose(baseline, legacy, atol=2e-6, rtol=1e-5)
    signals = dict(baseline=baseline, legacy=legacy)
    checks = dict(baseline_max_abs_difference=float(np.abs(baseline-legacy).max()), heads={})
    users = {s['user_id'] for s in chosen}
    anchor = torch.load(D.DUAL/domain/'gate.pt', map_location='cpu')
    assert users.isdisjoint(anchor['fit_users'])
    for arm in D.ARMS:
        path = screen/f'{arm}_0.pt'; use(path)
        ck = torch.load(path, map_location='cpu')
        assert ck['seed'] == 0 and ck['arm'] == arm and ck['config'] == D.CFG
        assert users.isdisjoint(ck['fit_users'])
        model = D.Head(); model.load_state_dict(ck['state_dict']); model.eval()
        f = baseline.copy()
        with torch.no_grad():
            for start in range(0, len(ix), 128):
                jj = ix[start:start+128]
                delta = data.delta(model, jj, arm, (ck['mean'], ck['std']))[0].numpy()
                for j, k in enumerate(jj):
                    mask = data.a['mask'][k]; ids = data.a['ids'][k, mask]
                    f[start+j, ids] += delta[j, mask]
                    if start+j in (0, 256, 511):
                        expected = D.numpy_delta(data, model, (ck['mean'], ck['std']), arm, k)
                        np.testing.assert_allclose(delta[j, mask], expected[mask], atol=3e-6, rtol=2e-5)
        signals[arm] = f
        checks['heads'][arm] = dict(seed=0, fit_users_disjoint=True, selected_epochs=ck['selected_epochs'])
    rank_checks = {}
    for arm in ARMS:
        f = signals[arm]; expected = saved['baseline' if arm == 'baseline' else arm+'_0'][ix]
        actual = []
        for j, s in enumerate(chosen):
            eligible = np.ones(f.shape[1], bool); eligible[np.asarray(s['history'])-1] = False
            ids = np.flatnonzero(eligible); order = ids[np.lexsort((ids, -f[j, ids]))]
            pos = np.flatnonzero(order == s['target']-1)
            actual.append(int(pos[0])+1 if len(pos) else f.shape[1]+1)
        np.testing.assert_array_equal(actual, expected)
        rank_checks[arm] = 512
    np.savez(folder/'signals.npz', **signals)
    dual_rows = {r['user_id']: r for r in G.clean.rows(old/'predictions.jsonl')}
    three_rows = {r['user_id']: r for r in G.clean.rows(D.SOURCE/domain/'candidates.jsonl') if r['role'] == 'development'}
    references = []
    for j, s in enumerate(chosen):
        old_r, three = dual_rows[s['user_id']], three_rows[s['user_id']]
        assert (old_r['target'], old_r['history']) == (s['target'], s['history'])
        for value, expected in ((old_r['native'], old_rr['GRAM'][j]), (old_r['guided'], old_rr['guided'][j]), (three['guided'], three_rr['dev_fixed_decode'][j])):
            assert record_rank(value, s) == expected
        references.append(dict(user_id=s['user_id'], native=old_r['native'], guided=old_r['guided'], three=three['guided']))
    write(folder/'references.json', references)
    write(folder/'preparation_checks.json', dict(state='PASSED', score_rank_checks=rank_checks, **checks))
    for path in [*[screen/'validation'/(k+'.npy') for k in keys], screen/'ranks.npz', D.FULL/domain/'samples.json',
                 old/'samples.json', old/'fused.npy', old/'predictions.jsonl', old/'ranks.npz', old/'gate.pt',
                 D.SOURCE/domain/'candidates.jsonl', D.SOURCE/domain/'ranks.npz',
                 *[folder/n for n in ('samples.json', 'data_config.json', 'catalog.json', 'signals.npz', 'references.json', 'preparation_checks.json')]]:
        use(path)
    write(folder/'status.json', dict(state='PREPARED', users_total=512, users_completed=0, updated_at=stamp()))


def prepare():
    import torch
    torch.set_num_threads(2)
    assert read(D.OUT/'status.json')['state'] == 'COMPLETED'
    assert read(D.OUT/'review/verification.json')['state'] == 'VERIFIED_SAVED_OUTPUTS'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG); shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}
    def use(path):
        manifest[str(path.relative_to(ROOT))] = sha(path)
    for domain in DOMAINS:
        for rel, digest in read(D.OUT/domain/'output_manifest.json')['sha256'].items():
            assert sha(ROOT/rel) == digest, rel
        prepare_domain(domain, use)
    paths = [Path(__file__), PLAN, OUT/'config.json', ROOT/'experiment/phase21/run_dual_candidate_generation.sh',
             ROOT/'experiment/phase21/dual_candidate_gate.py', ROOT/'experiment/phase21/candidate_gate_generation.py',
             ROOT/'experiment/phase21/candidate_gate_core.py', ROOT/'experiment/phase21/candidate_gate_screen.py',
             ROOT/'experiment/phase20/lime_clean_benefit.py', ROOT/'experiment/phase20/lime_decode_screen.py',
             ROOT/'experiment/phase20/lime_decode_screen_core.py', ROOT/'experiment/phase20/lime_clean_upstream_core.py',
             ROOT/'experiment/phase20/lime_clean_gram_core.py', ROOT/'experiment/phase20/guided_decoding_core.py',
             ROOT/'experiment/phase20/catalog_scope_core.py', ROOT/'experiment/phase18/core/diff_data.py',
             ROOT/'experiment/gpu_memory_lease.py', ROOT/'experiment/GPU_MEMORY_LEASE_POLICY.md',
             D.OUT/'summary.json', D.OUT/'review/verification.json']
    paths += list((ROOT/'GRAM/src').rglob('*.py'))
    from experiment.phase20.lime_clean_gram_core import T5
    paths += [T5/n for n in ('tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'config.json', 'pytorch_model.bin')]
    for path in paths:
        use(path)
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 双专家候选权重真实生成\n\n状态：PREPARED。固定seed0、同批512用户，尚无新生成结果。\n')
    print(json.dumps(dict(state='PREPARED', checks=4096, domains=list(DOMAINS))), flush=True)


class Worker(G.Worker):
    def __init__(self, domain, lease):
        G.OUT = OUT
        self.row_seconds = []
        super().__init__(domain, lease)

    def status(self, state='RUNNING', **values):
        done = getattr(self, 'users_completed', 0)
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        super().status(state, remaining_seconds_estimate=rate*(512-done) if rate else None,
                       median_seconds_per_user=rate, gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       lease_status_path=str((self.folder/'lease_status.json').relative_to(ROOT)), **values)

    def run(self):
        self.stage = 'native_zero_and_legacy_parity'; self.status()
        longest = max(range(len(self.samples)), key=lambda j: len(self.samples[j]['history']))
        js = list(dict.fromkeys([longest, 0, 256, 511]))[:3]; checks = []
        for j in js:
            ctx = self.context(j); native = self.decode(ctx)
            plain = self.decode(ctx, native_entry=True, record=False)
            q = self.index.conditional_logs(self.signals['legacy'][j])
            zero = self.decode(ctx, q, 0.); legacy = self.decode(ctx, q, .3)
            for other in (plain, zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'], other['scores'])
            ref = self.reference[self.samples[j]['user_id']]
            for output, key in ((native, 'native'), (legacy, 'guided')):
                assert output['ids'] == ref[key]['ids'], (j, key)
                np.testing.assert_allclose(output['scores'], ref[key]['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user_id=self.samples[j]['user_id'], alpha0_exact=True, legacy_reproduced=True))
            del ctx, native, plain, zero, legacy
        write(self.folder/'smoke.json', dict(state='PASSED', checks=checks, longest_history_included=True))
        self.stage = 'matched_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('x') as handle:
            for j, s in enumerate(self.samples):
                started = time.monotonic(); ctx = self.context(j); native = self.decode(ctx)
                ref = self.reference[s['user_id']]
                assert native['ids'] == ref['native']['ids']
                np.testing.assert_allclose(native['scores'], ref['native']['scores'], atol=2e-6, rtol=1e-5)
                row = dict(user_id=s['user_id'], target=s['target'], history=s['history'],
                           native=dict(ids=native['ids'], scores=native['scores'].tolist()),
                           legacy_dual=ref['guided'], three=ref['three'])
                for arm in ARMS:
                    q = self.index.conditional_logs(self.signals[arm][j])
                    late = native['guide'].rescore(native['paths'], q, .3)
                    guided = self.decode(ctx, q, .3)
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                    del guided
                handle.write(json.dumps(row)+'\n'); handle.flush(); os.fsync(handle.fileno())
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-started)
                if j == 0 or (j+1) % 8 == 0 or j == 511:
                    self.status()
                del ctx, native
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', native_reproduced_users=512,
              model_digest_unchanged=True, model_digest=self.initial_digest, optimizer_steps=0))
        self.stage = 'complete'; self.status('COMPLETED', model_digest_unchanged=True)


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    result = []
    for line in output.splitlines():
        index, uuid, free, util = [v.strip() for v in line.split(',')]
        result.append(dict(index=int(index), uuid=uuid, free_mib=int(free), utilization=int(util)))
    return result


def available_gpus(inventory, occupied, slots):
    return sorted([g for g in inventory if g['uuid'] not in occupied and g['free_mib'] >= CFG['minimum_free_mib']],
                  key=lambda g: (-g['free_mib'], g['utilization'], g['index']))[:slots]


def worker(domain):
    folder = OUT/domain; lease = None; gpu = os.environ['CUDA_VISIBLE_DEVICES']
    def interrupted(signum, frame):
        raise InterruptedError('Worker interrupted; durable predictions retained')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        assert read(OUT/'config.json') == CFG
        for rel, digest in read(OUT/'manifest.json')['sha256'].items():
            assert sha(ROOT/rel) == digest, rel
        free = next(g['free_mib'] for g in gpu_inventory() if g['uuid'] == gpu)
        if free < CFG['minimum_free_mib']:
            write(folder/'status.json', dict(state='ADMISSION_DEFERRED', gpu_uuid=gpu, free_mib=free,
                  required_free_mib=CFG['minimum_free_mib'], updated_at=stamp(), experiment_started=False))
            return 75
        with (folder/'lease.log').open('x') as log:
            lease_path = folder/'lease_status.json'
            lease = subprocess.Popen([sys.executable, str(ROOT/'experiment/gpu_memory_lease.py'), '--gpu', '0',
                    '--total-lease-mib', '30720', '--expected-workload-peak-mib', '12288', '--status-path', str(lease_path)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            begin = time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state') != 'holding':
                assert lease.poll() is None, 'GPU lease startup failed'
                assert time.monotonic()-begin < 180, 'GPU lease startup timeout'
                time.sleep(1)
            Worker(domain, lease).run()
        return 0
    except BaseException as exc:
        previous = read(folder/'status.json')
        write(folder/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(),
              previous=previous, updated_at=stamp(), test_read=False))
        raise
    finally:
        if lease is not None and lease.poll() is None:
            lease.terminate()
            try:
                lease.wait(timeout=15)
            except subprocess.TimeoutExpired:
                lease.kill(); lease.wait()


def summarize():
    results = {}; lines = ['# 双专家候选权重真实生成', '',
        '状态：COMPLETED。同批512开发用户、固定小头seed0与每域一个上游seed，探索性结果。', '',
        'GRAM逐用户重放，双专家投影基线重新生成；旧双专家和三专家来自已核验同批输出。没有重新训练或选择小头种子。', '',
        '| 域 | GRAM N10 | 旧双专家 N10 | 重放双专家 N10 | 用户权重 N10 | 候选权重 N10 | 分数修正 N10 | 三专家 N10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for domain in DOMAINS:
        folder = OUT/domain
        assert read(folder/'status.json')['state'] == 'COMPLETED'
        assert read(folder/'runtime_checks.json')['model_digest_unchanged']
        assert read(folder/'lease_status.json')['state'] == 'released'
        samples = read(folder/'samples.json')
        records = G.clean.rows(folder/'predictions.jsonl')
        assert len(records) == len(samples) == 512
        keys = ['native', 'legacy_dual', 'three']+[a+suffix for a in ARMS for suffix in ('_late', '_guided')]
        rr = {key: [] for key in keys}; parity = dict(identical_item_lists=0, top10_rank_identical=0, target_rank_identical=0)
        for row, s in zip(records, samples):
            assert (row['user_id'], row['target'], row['history']) == (s['user_id'], s['target'], s['history'])
            for key in keys:
                rr[key].append(record_rank(row[key], s))
            parity['identical_item_lists'] += row['baseline_guided']['ids'] == row['legacy_dual']['ids']
            a, b = rr['baseline_guided'][-1], rr['legacy_dual'][-1]
            parity['target_rank_identical'] += a == b
            parity['top10_rank_identical'] += (a == b or min(a, b) > 10)
        rr = {k: np.array(v, dtype=np.int64) for k, v in rr.items()}
        old = np.load(D.DUAL/domain/'ranks.npz')
        np.testing.assert_array_equal(old['users'], [s['user_id'] for s in samples])
        np.testing.assert_array_equal(rr['native'], old['GRAM'])
        np.testing.assert_array_equal(rr['legacy_dual'], old['guided'])
        three = np.load(D.SOURCE/domain/'ranks.npz')
        np.testing.assert_array_equal(rr['three'], three['dev_fixed_decode'])
        np.savez_compressed(folder/'ranks.npz', users=np.array([s['user_id'] for s in samples]), **rr)
        pairs = [('candidate_guided', b) for b in ('user_guided', 'scalar_guided', 'baseline_guided', 'native', 'three', 'candidate_late')]
        pairs += [(a+'_guided', 'baseline_guided') for a in ('user', 'scalar')]
        result = dict(users=512, head_seed=0, metrics={k: metrics(v) for k, v in rr.items()},
                      comparisons={a+'_vs_'+b: contrast(rr[a], rr[b], repeats=2000, seed=20261014) for a, b in pairs},
                      baseline_replay_parity=parity, test_read=False, optimizer_steps=0)
        # Match the original motivation without silently equating expert H50 with beam coverage.
        result['coverage'] = {}
        for a in ARMS:
            r = rr[a+'_guided']; g = rr['native']
            result['coverage'][a] = dict(beam_H50=float((r <= 50).mean()),
                new_top10_outside_native50=int(((r <= 10) & (g > 50)).sum()),
                top50_gained=int(((r <= 50) & (g > 50)).sum()), top50_lost=int(((r > 50) & (g <= 50)).sum()))
        results[domain] = result; write(folder/'summary.json', result)
        m = result['metrics']
        names = ('native', 'legacy_dual', 'baseline_guided', 'user_guided', 'candidate_guided', 'scalar_guided', 'three')
        lines.append('| '+domain+' | '+' | '.join(f'{m[n]["N10"]:.6f}' for n in names)+' |')
        write(folder/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', rows=512,
              rank_arrays=len(rr), old_reference_rank_arrays_matched=3, model_unchanged=True,
              test_read=False, baseline_replay_parity=parity))
        outputs = [folder/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json', 'runtime_checks.json', 'smoke.json')]
        write(folder/'output_manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs}))
    write(OUT/'summary.json', dict(domains=results, test_read=False, head_seed=0))
    lines += ['', 'H10/H50、逐用户新增/丢失和配对描述性区间均在 summary.json。对照顺序为：候选相对用户级、直接分数、重放双专家、GRAM、三专家，以及同分数只做后重排。', '',
              '不能因超过GRAM便归因于候选粒度；需要与用户级和分数对照一起判断。512用户反复用于探索，只有一个固定小头seed，区间不包含训练随机性和研究选择，不能当作最终论文确认。若重放基线与旧基线不同，以本轮同流程重放基线为主并公开差异。', '',
              '本轮只验证当前候选机制。没有重做专家消融；旧专家消融不能直接证明调整权重后两专家仍各自必要。后续是否继续由结果决定，不自动追加实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    start = time.monotonic(); pending = list(DOMAINS); active = {}; finished = {}; attempts = {d: 0 for d in DOMAINS}
    try:
        write(OUT/'status.json', dict(state='VERIFYING_INPUTS', pid=os.getpid(), updated_at=stamp()))
        for rel, digest in read(OUT/'manifest.json')['sha256'].items():
            assert sha(ROOT/rel) == digest, rel
        while pending or active:
            for domain, job in list(active.items()):
                code = job['process'].poll()
                if code is None and time.monotonic()-job['started'] > CFG['worker_timeout_seconds']+120:
                    job['process'].terminate()
                    try:
                        code = job['process'].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        job['process'].kill(); code = job['process'].wait()
                    write(OUT/domain/'status.json', dict(state='TIME_LIMIT', updated_at=stamp()))
                if code is not None:
                    job['log'].close(); state = read(OUT/domain/'status.json')
                    if code == 75 and state['state'] == 'ADMISSION_DEFERRED':
                        pending.append(domain)  # no CUDA/scientific job had started
                    else:
                        lease = OUT/domain/'lease_status.json'
                        finished[domain] = dict(exit_code=code, state=state['state'],
                            lease_released=lease.exists() and read(lease)['state'] == 'released')
                    del active[domain]
            if pending and time.monotonic()-start > CFG['resource_wait_seconds']:
                for domain in pending:
                    write(OUT/domain/'status.json', dict(state='RESOURCE_WAIT_LIMIT', updated_at=stamp()))
                    finished[domain] = dict(exit_code=124, state='RESOURCE_WAIT_LIMIT', lease_released=True)
                pending = []
            inventory = gpu_inventory()
            free = available_gpus(inventory, {j['gpu']['uuid'] for j in active.values()}, CFG['max_parallel']-len(active))
            for gpu in free:
                if not pending:
                    break
                domain = pending.pop(0); attempts[domain] += 1; folder = OUT/domain
                log = (folder/f'worker_attempt{attempts[domain]}.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                write(folder/'status.json', dict(state='STARTING', gpu_index=gpu['index'], gpu_uuid=gpu['uuid'], updated_at=stamp()))
                process = subprocess.Popen([sys.executable, '-u', '-m', 'experiment.phase21.dual_candidate_generation', 'worker', '--domain', domain],
                                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[domain] = dict(process=process, log=log, started=time.monotonic(), gpu=gpu)
            write(OUT/'status.json', dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                  pid=os.getpid(), updated_at=stamp(), elapsed_seconds=time.monotonic()-start, pending=pending, finished=finished,
                  active={d: dict(pid=j['process'].pid, gpu_index=j['gpu']['index'], gpu_uuid=j['gpu']['uuid'],
                          worker=read(OUT/d/'status.json')) for d, j in active.items()},
                  gpu_inventory=inventory, required_free_mib=CFG['minimum_free_mib'], test_read=False))
            if pending or active:
                time.sleep(10)
        complete = len(finished) == 2 and all(v['exit_code'] == 0 and v['state'] == 'COMPLETED' and v['lease_released'] for v in finished.values())
        if complete:
            summarize()
        write(OUT/'status.json', dict(state='COMPLETED' if complete else 'INCOMPLETE', finished=finished,
              elapsed_seconds=time.monotonic()-start, updated_at=stamp(), test_read=False, report=str(REPORT.relative_to(ROOT))))
    finally:
        for job in active.values():
            if job['process'].poll() is None:
                job['process'].terminate()
            job['log'].close()


def tests():
    sample = dict(target=3, history=[2]); value = dict(ids=list(range(1, 51)), scores=[1.]*50)
    assert record_rank(value, sample) == 2
    assert record_rank(value, dict(target=2, history=[2])) == 2**30
    assert record_rank(value, dict(target=100, history=[])) == 2**30
    pool = [dict(index=i, uuid=str(i), free_mib=m, utilization=0) for i, m in [(0, 20000), (5, 40960), (6, 36000), (7, 42000)]]
    assert [g['index'] for g in available_gpus(pool, {'7'}, 2)] == [5, 6]
    assert [g['index'] for g in available_gpus(pool, {'7'}, 1)] == [5]
    return dict(history_and_tie_ranking=True, missing_target_retained=True, all_gpu_pool_and_admission=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['test', 'prepare', 'supervise', 'worker', 'summarize'])
    parser.add_argument('--domain', choices=DOMAINS); args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(tests()))
        elif args.mode == 'prepare':
            checks = tests(); prepare(); write(OUT/'tests.json', checks)
        elif args.mode == 'worker':
            assert args.domain
            return worker(args.domain)
        elif args.mode == 'supervise':
            supervise()
        else:
            summarize()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'worker':
            write(OUT/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(), updated_at=stamp()))
        raise


if __name__ == '__main__':
    sys.exit(main())
