"""A/B/C text pooling screen, followed by predeclared conditional generation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from experiment.phase21 import text_contribution as runner
from experiment.phase21 import text_adaptation_core as core
from experiment.phase21.expert_mechanism_core import ScoreOverride
from experiment.phase21.expert_resource_recovery import RecoveryMixin, GPU_POOL, stop_owned
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = runner.ROOT
SOURCE = runner.SOURCE
TEXT = ROOT / 'artifacts/phase21/text_contribution_20260923_v1'
RECOVERY = ROOT / 'artifacts/phase21/expert_resource_recovery_20260923_v1'
OUT = ROOT / 'artifacts/phase21/text_adaptation_20260923_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.4B_文本专家ABC冻结执行_20260923.md'
REPORT = ROOT / 'report/第二十一阶段/文本专家ABC优化_20260923.md'
MODULE = 'experiment.phase21.text_adaptation'
DOMAINS = ('Toys', 'Beauty')
read, write, sha, stamp = runner.read, runner.write, runner.sha, runner.stamp


def status(path, state='RUNNING', **data):
    write(path / 'status.json', dict(state=state, pid=os.getpid(), updated_at=stamp(), test_read=False, **data))


def prepare():
    OUT.mkdir(exist_ok=False); status(OUT, 'PREPARING')
    assert read(TEXT / 'status.json')['state'] == 'COMPLETED'
    cfg = dict(seed=0, epochs=4, batch_size=256, learning_rate=.02, weight_decay=.001,
               score_scale=20., alpha=.3, beam_size=50, development_users=512, tau=.1,
               candidate_block=1024, allocation_limit_gib=11, worker_timeout_seconds=14400,
               supervisor_timeout_seconds=172800, minimum_free_mib=16384, max_parallel=2,
               shared_resource_queue=str(RECOVERY), test_read=False, forecast_policy='warning_only')
    write(OUT / 'config.json', cfg); shutil.copy2(PLAN, OUT / 'protocol.md')
    (OUT / 'shared').mkdir(); (OUT / 'A').mkdir()
    for arm in core.ARMS:
        (OUT / arm).mkdir(); write(OUT / arm / 'config.json', cfg)
    frozen = dict(read(TEXT / 'manifest.json')['sha256'])
    upstream = read(SOURCE / 'manifest.json')['sha256']
    for domain in DOMAINS:
        old = TEXT / 'real' / domain
        (OUT / 'A' / domain).symlink_to(old, target_is_directory=True)
        shared = OUT / 'shared' / domain; shared.mkdir()
        (shared / 'donors.npy').symlink_to(TEXT / 'shared' / domain / 'donors.npy')
        samples = read(old / 'samples.json')
        assert sum(s['role'] == 'development' for s in samples) == 512
        assert sum(s['role'] == 'gate_calibration' for s in samples) == dict(Toys=721, Beauty=775)[domain]
        cfg_d = read(old / 'data_config.json')
        assert cfg_d['embedding'] in upstream
        frozen[cfg_d['embedding']] = upstream[cfg_d['embedding']]
        for name in ('samples.json', 'data_config.json', 'fused.npy', 'gate.pt', 'summary.json', 'ranks.npz'):
            frozen[str((old / name).relative_to(ROOT))] = sha(old / name)
        for arm in core.ARMS:
            folder = OUT / arm / domain; folder.mkdir()
            for name in ('samples.json', 'data_config.json', 'catalog.json'):
                (folder / name).symlink_to(old / name)
            status(folder, 'NOT_STARTED', arm=arm, domain=domain)
        for kind in ('cache_main', 'cache_D'):
            p = OUT / kind / domain; p.mkdir(parents=True)
            status(p, 'QUEUED' if kind == 'cache_main' else 'CONDITIONAL_NOT_STARTED')
    paths = [Path(__file__), ROOT / 'experiment/phase21/text_adaptation_core.py',
             ROOT / 'experiment/phase21/test_text_adaptation.py', ROOT / 'experiment/phase21/run_text_adaptation.sh',
             ROOT / 'experiment/phase21/expert_resource_recovery.py', ROOT / 'experiment/phase21/expert_mechanism_core.py',
             ROOT / 'experiment/phase20/lime_comparison_core.py', PLAN, OUT / 'config.json']
    paths.extend(OUT.glob('*/config.json'))
    for p in paths:
        frozen[str(p.relative_to(ROOT))] = sha(p)
        if p.suffix in ('.py', '.sh', '.md'):
            dest = OUT / 'sources' / p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    for count, (rel, digest) in enumerate(frozen.items(), 1):
        assert sha(ROOT / rel) == digest, rel
        if count % 10 == 0: status(OUT, 'VERIFYING_INPUTS', verified=count, total=len(frozen))
    write(OUT / 'manifest.json', dict(sha256=frozen, created_at=stamp()))
    status(OUT, 'PREPARED')
    REPORT.write_text('# 文本专家ABC优化\n\n状态：PREPARED。配置已冻结；A复用已完成结果，B/C尚无新效果结论。\n')


def bind():
    runner.OUT = OUT; runner.transform_scores = core.transform_scores; runner.fuse = core.fuse


class SignalFit:
    """Use the exact original fusion trainer without loading GRAM."""
    fit_signals = runner.Worker.fit_signals

    def __init__(self, arm, domain, reporter):
        self.arm, self.domain = arm, domain
        self.folder, self.src = OUT / arm / domain, SOURCE / domain
        self.samples = read(self.folder / 'samples.json')
        self.source_indices = np.asarray([s['source_sample_index'] for s in self.samples])
        raw = np.load(self.src / 'experts.npy', mmap_mode='r')
        replacement = np.load(self.folder / 'semantic.npy', mmap_mode='r')
        self.raw = ScoreOverride(raw, self.source_indices, replacement, 2)
        self.development = [i for i, s in enumerate(self.samples) if s['role'] == 'development']
        self.donors = np.load(OUT / 'shared' / domain / 'donors.npy')
        self.device, self.fit_steps = 'cuda:0', 0
        self.reporter = reporter

    def status(self, **values):
        self.reporter(stage=self.stage, arm=self.arm, fusion_optimizer_steps=self.fit_steps, **values)


def references(domain):
    with (SOURCE / domain / 'candidates.jsonl').open() as handle:
        return {r['user_id']: r for r in map(json.loads, handle) if r['role'] == 'development'}


def evaluate_cache(domain, arms, reporter):
    samples = read(OUT / 'B' / domain / 'samples.json'); refs = references(domain)
    selected = [(i, s) for i, s in enumerate(samples) if s['role'] == 'development']
    original_scores = np.load(SOURCE / domain / 'experts.npy', mmap_mode='r')
    result = {}
    for arm in arms:
        fused = np.load(OUT / arm / domain / 'fused.npy', mmap_mode='r')
        text_scores = None if arm == 'A' else np.load(OUT / arm / domain / 'semantic.npy', mmap_mode='r')
        ids = np.arange(1, fused.shape[1] + 1)
        rr = dict(text=[], fusion_catalog=[], fixed50_fusion=[])
        for count, (j, sample) in enumerate(selected, 1):
            semantic = original_scores[sample['source_sample_index'], 2] if arm == 'A' else text_scores[j]
            target, history = sample['target'], sample['history']
            rr['text'].append(runner.verified_rank(ids, semantic, target, history))
            rr['fusion_catalog'].append(runner.verified_rank(ids, fused[j], target, history))
            candidates = np.asarray(refs[sample['user_id']]['ids'])
            rr['fixed50_fusion'].append(runner.verified_rank(candidates, fused[j][candidates-1], target, history))
            if count % 64 == 0: reporter(stage='cache_evaluation', arm=arm, users=count, total=512)
        if arm == 'A':
            np.testing.assert_array_equal(rr['fusion_catalog'], runner.base.arrays(TEXT / 'real' / domain / 'ranks.npz')['LIME'])
        kind = 'cache_D' if arms == ('D',) else 'cache_main'
        np.savez_compressed(OUT / kind / domain / f'{arm}_ranks.npz',
                            users=np.asarray([s['user_id'] for _, s in selected]), **rr)
        result[arm] = dict(metrics={k: metrics(v) for k, v in rr.items()}, users=512,
                          rank_verified_by_sort_and_count=True)
    return result


def cache(domain, only_D=False):
    import torch
    sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
    from lime_rec.evaluation import normalize_score_rows
    from lime_rec.experts import LLMSemanticExpert
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    kind = 'cache_D' if only_D else 'cache_main'; folder = OUT / kind / domain
    started = time.monotonic()
    reporter = lambda **kw: status(folder, **kw)
    reporter(stage='load_embeddings')
    bind(); cfg = read(OUT / 'config.json'); cfg_d = read(OUT / 'B' / domain / 'data_config.json')
    samples = read(OUT / 'B' / domain / 'samples.json')
    catalog = read(OUT / 'B' / domain / 'catalog.json')
    with np.load(ROOT / cfg_d['embedding'], allow_pickle=False) as data:
        assert data['item_ids'].tolist() == catalog['items'][1:]
        embeddings = data['embeddings'].astype(np.float32)
    assert embeddings.ndim == 2 and np.isfinite(embeddings).all()
    author = LLMSemanticExpert(catalog['items'][1:], {}, embeddings=embeddings)
    e = torch.from_numpy(embeddings).to('cuda:0'); v = core.unit_rows(e)
    donors = np.load(OUT / 'shared' / domain / 'donors.npy')
    mapping = torch.from_numpy(donors).to('cuda:0')
    raw = np.load(SOURCE / domain / 'experts.npy', mmap_mode='r')
    arms = ('D',) if only_D else ('B', 'C')
    arrays = {arm: np.lib.format.open_memmap(OUT / arm / domain / 'semantic.npy', mode='w+',
                    dtype='float32', shape=(len(samples), len(e))) for arm in arms}
    maximum_error, dispersion, examples = 0., [], []
    with torch.no_grad():
        for j, sample in enumerate(samples):
            if time.monotonic() - started > cfg['worker_timeout_seconds']: raise TimeoutError('Cache worker actual 4h limit')
            history = sample['full_history']
            a = core.score_user(e, v, history, 'A').cpu().numpy()
            a = normalize_score_rows(a[None])[0]
            expected = raw[sample['source_sample_index'], 2]
            np.testing.assert_allclose(a, expected, atol=5e-6, rtol=1e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(a - expected))))
            if j in (0, len(samples)//2, len(samples)-1):
                old = author.score(sample['user_id'], [catalog['items'][x] for x in history])
                np.testing.assert_allclose(a, normalize_score_rows(old[None])[0], atol=5e-6, rtol=1e-5)
            for arm in arms:
                value = core.score_user(e, v, history, arm, tau=cfg['tau'], block=cfg['candidate_block'],
                                        query_map=mapping if arm == 'D' else None).cpu().numpy()
                arrays[arm][j] = normalize_score_rows(value[None])[0]
            h, r, m = core.pool_inputs(e, history)
            spread = float(1 - ((core.unit_rows(h) @ core.unit_rows(m)) * r/r.sum()).sum())
            dispersion.append(dict(user_id=sample['user_id'], role=sample['role'], length=len(history), value=spread))
            if j in (0, len(samples)//2, len(samples)-1):
                weights = core.query_weights(e[:3], h, r, cfg['tau']).cpu().numpy().tolist()
                example = dict(user_id=sample['user_id'], history=history, candidate_ids=[1,2,3],
                               real_candidate_history_weights=weights, targets_used=False)
                if only_D:
                    wrong = core.query_weights(e[mapping[:3]], h, r, cfg['tau']).cpu().numpy()
                    example.update(donor_ids=(donors[:3]+1).tolist(), donor_history_weights=wrong.tolist(),
                                   mean_abs_weight_difference=float(np.abs(wrong-np.asarray(weights)).mean()))
                examples.append(example)
            if (j+1) % 16 == 0 or j == 0:
                reporter(stage='text_scores', rows=j+1, total=len(samples), elapsed_seconds=time.monotonic()-started)
    for array in arrays.values(): array.flush()
    write(folder / 'score_audit.json', dict(state='PASSED', original_text_max_abs_error=maximum_error,
          original_author_parity_rows=[0,len(samples)//2,len(samples)-1], identical_embeddings_and_full_history=True,
          query_only_permutation=only_D, donor_changed_fraction=float(np.mean(donors!=np.arange(len(donors)))),
          other_expert_channels_unchanged=True, extra_trainable_parameters=0,
          text_scoring_seconds=time.monotonic()-started))
    train_values = [s['value'] for s in dispersion if s['role']=='gate_calibration' and s['length'] > 1]
    cuts = np.quantile(train_values, [1/3,2/3]).tolist()
    for row in dispersion:
        row['group'] = 'single' if row['length']==1 else ['low','middle','high'][int(np.searchsorted(cuts,row['value'],side='right'))]
    write(folder / 'history_dispersion.json', dict(training_only_cutpoints=cuts, rows=dispersion))
    write(folder / 'weight_examples.json', examples)
    del e, v, h, r, m, mapping, author, embeddings
    torch.cuda.empty_cache()
    for arm in arms:
        fit = SignalFit(arm, domain, reporter); fit.fit_signals(); del fit
        cache_hashes = {name: sha(OUT / arm / domain / name) for name in
                        ('semantic.npy','fused.npy','gate.pt','gate_prediction.npz','fit_checks.json')}
        write(OUT / arm / domain / 'cache_manifest.json', dict(sha256=cache_hashes, created_at=stamp()))
        status(OUT / arm / domain, 'CACHE_READY', arm=arm, domain=domain)
    evaluated = evaluate_cache(domain, ('D',) if only_D else ('A','B','C'), reporter)
    write(folder / 'summary.json', dict(state='COMPLETED', arms=evaluated, domain=domain,
          source='full_catalog_scores_and_fixed_original50', actual_generation_performed=False,
          fixed50_rule='rank_by_fused_score_only_not_AGRec_probability_mixture', seconds=time.monotonic()-started,
          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    status(folder, 'COMPLETED', seconds=time.monotonic()-started)


class GenerationWorker(runner.Worker):
    smoke_control = RecoveryMixin.smoke_control

    def fit_signals(self):
        import torch
        self.fused = np.load(self.folder / 'fused.npy', mmap_mode='r')
        gate = torch.load(self.folder / 'gate.pt', map_location='cpu')
        assert gate['arm'] == self.arm and not gate['development_labels_used']
        expected = sorted(s['user_id'] for s in self.samples if s['role']=='gate_calibration')
        assert gate['fit_users'] == expected
        self.fit_steps = gate['optimizer_steps']
        assert self.fit_steps == dict(Toys=12,Beauty=16)[self.domain]
        assert read(self.folder / 'fit_checks.json')['state'] == 'PASSED'
        self.stage = 'reuse_screened_fusion'; self.status()


def job_folder(job):
    return OUT / (job['arm'] if job['kind']=='decode' else job['kind']) / job['domain']


def worker(job):
    folder = job_folder(job); stop = threading.Event()
    def beat():
        while not stop.is_set():
            write(folder / 'heartbeat.json', dict(pid=os.getpid(), updated_at=stamp(), liveness_only=True))
            stop.wait(30)
    thread = threading.Thread(target=beat, daemon=True); thread.start()
    try:
        for rel, digest in read(OUT / 'manifest.json')['sha256'].items():
            if rel.startswith(('experiment/','GRAM/src/','third_party/')): assert sha(ROOT / rel)==digest, rel
        cfg_d = read(OUT / 'B' / job['domain'] / 'data_config.json')
        manifest = read(OUT / 'manifest.json')['sha256']
        if job['kind'] != 'decode':
            assert sha(ROOT / cfg_d['embedding']) == manifest[cfg_d['embedding']]
            cache(job['domain'], only_D=job['kind']=='cache_D')
        else:
            for name, digest in read(folder / 'cache_manifest.json')['sha256'].items(): assert sha(folder / name)==digest
            assert sha(ROOT / cfg_d['parent_checkpoint']) == cfg_d['parent_sha256']
            bind(); GenerationWorker(job['arm'], job['domain']).run()
    except BaseException as error:
        (folder / 'failure.txt').write_text(traceback.format_exc())
        status(folder, 'RESOURCE_TIME_LIMIT' if isinstance(error,TimeoutError) else 'FAILED', error=repr(error))
        raise
    finally:
        stop.set(); thread.join(timeout=2)


def external_capacity():
    """The older immutable scheduler may fill any vacancy while jobs remain."""
    path = RECOVERY / 'status.json'
    if not path.exists(): return 0, set(), 'recovery_status_unavailable'
    d = read(path)
    if d['state'] in ('PREPARING','PREPARED','VERIFYING_INPUTS') or d.get('pending'):
        return 0, set(), 'existing_queue_has_priority'
    occupied = set()
    launch = RECOVERY / 'launch.json'
    if launch.exists():
        for item in read(launch)['launches']:
            cmd = Path(f"/proc/{item['pid']}/cmdline")
            try: alive = 'experiment.phase21.expert_resource_recovery' in cmd.read_bytes().replace(b'\0',b' ').decode()
            except (FileNotFoundError, ProcessLookupError): alive = False
            if alive: occupied.add(item['gpu'])
    return max(0,2-len(occupied)), occupied, None


def run_stage(jobs, phase, pipeline_start):
    cfg = read(OUT / 'config.json'); pending = list(jobs); active = {}; completed = []
    launches = read(OUT/'launch.json')['launches'] if (OUT/'launch.json').exists() else []
    try:
        while pending or active:
            now = time.monotonic()
            if now-pipeline_start > cfg['supervisor_timeout_seconds']: raise TimeoutError('Pipeline actual 48h limit including queue wait')
            for key, rec in list(active.items()):
                process = rec['process']; folder = job_folder(rec['job'])
                if process.poll() is None and now-rec['start'] > cfg['worker_timeout_seconds']:
                    status(folder,'RESOURCE_TIME_LIMIT',reason='Actual worker wall time exceeded 4h')
                    stop_owned(process)
                if process.poll() is not None:
                    del active[key]
                    child_status = read(folder/'status.json')
                    if process.returncode != 0 or child_status['state'] != 'COMPLETED':
                        if child_status['state'] == 'RESOURCE_TIME_LIMIT':
                            raise TimeoutError(f'Worker resource limit: {key}: {child_status}')
                        raise RuntimeError(f'Worker stopped: {key}: {child_status}')
                    completed.append(rec['job'])
            capacity, external_gpus, reason = external_capacity()
            free = {}; query_error = None
            if pending and len(active) < capacity:
                try:
                    output = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=60)
                    free = {x.split(',')[0].strip():int(x.split(',')[1]) for x in output.splitlines()}
                except (subprocess.SubprocessError,ValueError) as error: query_error=repr(error)
            busy = external_gpus | {r['gpu'] for r in active.values()}
            available = sorted([g for g in GPU_POOL if g not in busy and free.get(g,0)>=cfg['minimum_free_mib']],key=lambda g:-free[g])
            while pending and available and len(active)<capacity:
                job=pending.pop(0); gpu=available.pop(0); folder=job_folder(job)
                key=f"{job['kind']}/{job.get('arm','')}/{job['domain']}"
                env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
                cmd=[sys.executable,'-u','-m',MODULE,'worker','--kind',job['kind'],'--domain',job['domain']]
                if 'arm' in job: cmd += ['--arm',job['arm']]
                status(folder,'STARTING',gpu=gpu)
                with (folder/'worker.log').open('x') as handle:
                    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=dict(process=process,job=job,gpu=gpu,start=time.monotonic())
                launches.append(dict(job=job,pid=process.pid,gpu=gpu,launched_at=stamp()))
                write(OUT/'launch.json',dict(supervisor_pid=os.getpid(),launches=launches))
            status(OUT,'RUNNING' if active else 'WAITING_FOR_RESOURCES',phase=phase,pending=pending,completed_in_stage=completed,
                active={k:dict(pid=r['process'].pid,gpu=r['gpu'],job=r['job'],worker=read(job_folder(r['job'])/'status.json')) for k,r in active.items()},
                waiting_reason=reason,resource_query_error=query_error,elapsed_seconds=now-pipeline_start)
            if pending or active: time.sleep(15)
    except BaseException:
        for rec in active.values():
            stop_owned(rec['process']); status(job_folder(rec['job']),'SUPERVISOR_STOPPED')
        raise


def write_screen_report(screen, verdict):
    lines=['# 文本专家ABC优化','','状态：CACHE_SCREEN_COMPLETED。缓存评分筛查完成；不是新真实生成结果。','',
           '| 域 | 组 | 文本N10 | 融合全目录N10 | 融合H10 | 原GRAM50按融合分排序N10 |','|---|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        for arm in ('A','B','C'):
            m=screen[domain]['arms'][arm]['metrics']
            lines.append(f'| {domain} | {arm} | {m["text"]["N10"]:.6f} | {m["fusion_catalog"]["N10"]:.6f} | {m["fusion_catalog"]["H10"]:.6f} | {m["fixed50_fusion"]["N10"]:.6f} |')
    lines += ['', '按预设规则进入真实生成：'+str(verdict['pass_for_actual_generation'])+'。', '',
        '固定50列只按融合分排序，未使用新前缀概率混合；原缓存缺少逐token原生概率，不能将其称为同规则AGRec后重排。该列不参与升级门槛。',
        'A复算核对原文本评分并复用旧融合器；B/C使用同一冻结BGE、full_history、相同融合训练及评价名单。',
        '单上游seed、反复使用开发池；未读test。未通过筛查不等同于实际生成已失败，也不穷尽否定候选相关聚合。', '',
        '[冻结协议](../../plan/第二十一阶段/21.4B_文本专家ABC冻结执行_20260923.md)']
    REPORT.write_text('\n'.join(lines)+'\n')


def generation_summary(arms):
    result={}; over_a={}; over_b={}
    lines=[REPORT.read_text(), '\n## 真实生成结果\n',
           '| 域 | 组 | N10 | H10 | H50 | 同规则后重排N10 |','|---|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        rr={arm:runner.base.arrays(OUT/arm/domain/'ranks.npz') for arm in arms}
        for arm in arms: np.testing.assert_array_equal(rr[arm]['users'],rr['A']['users'])
        m={arm:metrics(rr[arm]['guided']) for arm in arms}
        over_a[domain]=bool(m['C']['N10']>m['A']['N10'] and m['C']['H10']>=m['A']['H10'])
        over_b[domain]=m['C']['N10']-m['B']['N10']
        pairs={name:contrast(rr[left]['guided'],rr[right]['guided'],repeats=2000,seed=20260923)
               for name,left,right in [('C_minus_A','C','A'),('C_minus_B','C','B')]}
        if 'D' in arms: pairs['C_minus_D']=contrast(rr['C']['guided'],rr['D']['guided'],repeats=2000,seed=20260923)
        changes={}
        for arm in arms:
            np.testing.assert_array_equal(rr[arm]['GRAM'],rr['A']['GRAM'])
            current=m[arm]; late=metrics(rr[arm]['late'])
            lines.append(f'| {domain} | {arm} | {current["N10"]:.6f} | {current["H10"]:.6f} | {current["H50"]:.6f} | {late["N10"]:.6f} |')
            changes[arm]=dict(outside_original50_to_guided10=int(((rr[arm]['GRAM']>50)&(rr[arm]['guided']<=10)).sum()),
                              original50_lost=int(((rr[arm]['GRAM']<=50)&(rr[arm]['guided']>50)).sum()))
        dispersion=read(OUT/'cache_main'/domain/'history_dispersion.json')
        by_user={r['user_id']:r['group'] for r in dispersion['rows']}
        strata={}
        for group in ('single','low','middle','high'):
            mask=np.asarray([by_user[str(u)]==group for u in rr['A']['users']])
            if mask.any(): strata[group]=dict(users=int(mask.sum()), metrics={a:metrics(rr[a]['guided'][mask]) for a in arms})
        result[domain]=dict(metrics=m,contrasts=pairs,coverage=changes,history_dispersion=strata)
    supported=all(over_a.values()) and all(v>=0 for v in over_b.values()) and any(v>0 for v in over_b.values())
    lines += ['', 'C相对A的两域净收益方向：'+str(over_a)+'。',
              'C相对B的N10差：'+str(over_b)+'。',
              '区间、前十新增/丢失、覆盖和按训练切点分组见JSON；这些是探索性结果。']
    REPORT.write_text('\n'.join(lines)+'\n')
    output=dict(state='COMPLETED',domains=result,continue_to_D=bool(supported),test_read=False,
                new_actual_generation_arms=[a for a in arms if a!='A'], A_reused_verified=True)
    write(OUT/('generation_with_D.json' if 'D' in arms else 'generation_ABC.json'),output)
    return output


def supervise():
    started=time.monotonic()
    try:
        run_stage([dict(kind='cache_main',domain=d) for d in DOMAINS],'CACHE_SCREEN',started)
        screen={d:read(OUT/'cache_main'/d/'summary.json') for d in DOMAINS}
        verdict=core.screen_rule(screen)
        write(OUT/'screen_summary.json',dict(domains=screen,verdict=verdict,test_read=False))
        write_screen_report(screen,verdict)
        if not verdict['pass_for_actual_generation']:
            status(OUT,'COMPLETED_SCREEN_ONLY',reason='Predeclared generation admission not met',verdict=verdict,
                   actual_generation_performed=False)
            return
        run_stage([dict(kind='decode',arm=a,domain=d) for d in DOMAINS for a in ('B','C')],'ACTUAL_GENERATION_ABC',started)
        result=generation_summary(('A','B','C'))
        if result['continue_to_D']:
            run_stage([dict(kind='cache_D',domain=d) for d in DOMAINS],'CONDITIONAL_D_CACHE',started)
            run_stage([dict(kind='decode',arm='D',domain=d) for d in DOMAINS],'CONDITIONAL_D_GENERATION',started)
            result=generation_summary(('A','B','C','D'))
        status(OUT,'COMPLETED',D_performed='D' in result['new_actual_generation_arms'],
               duration_seconds=time.monotonic()-started)
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        status(OUT,'RESOURCE_TIME_LIMIT' if isinstance(error,TimeoutError) else 'FAILED',error=repr(error))
        raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','supervise','start','worker'])
    parser.add_argument('--kind',choices=['cache_main','cache_D','decode']); parser.add_argument('--domain',choices=DOMAINS)
    parser.add_argument('--arm',choices=core.ARMS); args=parser.parse_args()
    if args.action in ('prepare','start'): prepare()
    if args.action in ('supervise','start'): supervise()
    if args.action=='worker':
        assert args.kind and args.domain
        job=dict(kind=args.kind,domain=args.domain)
        if args.kind=='decode': assert args.arm; job['arm']=args.arm
        worker(job)


if __name__=='__main__': main()
