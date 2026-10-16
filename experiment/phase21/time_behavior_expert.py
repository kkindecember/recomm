"""SASRec vs TiSASRec vs equally-spaced-time control, frozen BGE and GRAM.

TiSASRec: Li et al., WSDM 2020. Existing expert normalization/history handling
derives from LIME-Rec; unchanged GRAM probability guidance is AGRec-inspired.
This is an exploratory model substitution, not a claim of a new architecture.
"""
import argparse
from collections import defaultdict
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import torch

from experiment.phase21 import single_fusion as S
from experiment.phase21 import single_fusion_core as C
from experiment.phase21 import candidate_text_source as T
from experiment.phase21 import tisasrec_core as B
from experiment.phase21.candidate_gate_core import fold
from experiment.phase21.train_time_behavior import train as fit_behavior

ROOT, OLD, SOURCE = S.ROOT, S.OUT, S.SOURCE
OUT = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
PLAN = ROOT/'plan/第二十一阶段/21.34_SASRec与TiSASRec并行对照_20261015.md'
REPORT = ROOT/'report/第二十一阶段/SASRec与TiSASRec并行对照_20261015.md'
MODULE = 'experiment.phase21.time_behavior_expert'
DOMAINS, NEW = ('Toys', 'Beauty'), ('tisas', 'tisas_order')
ARMS = ('full',)+NEW
BEHAVIOR = dict(seed=0, hidden=64, maxlen=50, num_layers=2, num_heads=2, dropout=.2,
    time_span=256, epochs=200, batch_size=128, lr=.001, weight_decay=0.,
    label_smoothing=.1, clip_norm=1., objective='full catalog CE incl class0, as existing SASRec',
    selection='fixed epoch200; no heldout labels or dates',
    normalization='per-query visible-prefix minimum positive gap; rounded coordinates; cap256',
    control='same TiSASRec, replace visible dates with equally spaced ordinal positions',
    precision='float32, TF32 off')
CFG = dict(S.CFG, arms=list(ARMS), experts=['SASRec or TiSASRec', 'fixed BGE'],
    behavior=BEHAVIOR, external_behavior_training=True, gram_optimizer_steps=0,
    question='Can time-aware behavior replace the current SASRec expert?',
    primary_comparisons=['tisas_guided vs full_guided', 'tisas_guided vs tisas_order_guided'],
    baseline='Existing seed0 fixed200 SASRec + verified user fuser; exact512 replay',
    architecture_and_parameter_counts_equal=False, tisas_control_parameters_equal=True,
    alpha_learned=False, all_arms_generate=True,
    timestamp_semantics='Amazon2014 unixReviewTime; day resolution, not purchase time',
    target_timestamps_used=False, future_time_normalization=False,
    stop_after='Two domains, fixed512 development users each, seed0; no automatic combined Qwen run')
CFG.pop('upstream_optimizer_steps')
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp
RAW = dict(Toys=Path('/tmp/recomm_toys_5_timeaudit_20261015.json.gz'),
           Beauty=Path('/tmp/recomm_beauty_5_timeaudit_20261015.json.gz'))
RAW_HASH = dict(Toys='f9f0cc56ad6b5de15a1b0dd1c07fcfc9e7367efc73481f5d94d8065f91149286',
                Beauty='e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe')
LEASE, VARIANT = None, None


def progress(domain, stage, **values):
    if LEASE is not None:
        assert LEASE.poll() is None, 'GPU lease exited'
    row = dict(state='RUNNING', stage=stage, variant=VARIANT, pid=os.getpid(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), updated_at=stamp(), test_read=False, **values)
    write(OUT/domain/'status.json', row)
    print(json.dumps(dict(domain=domain, **row)), flush=True)


def bind():
    S.OUT, S.REPORT, S.MODULE, S.CFG = OUT, REPORT, MODULE, CFG
    S.progress, S.summarize = progress, summarize
    T.OUT, T.OLD, T.CFG, T.ARMS = OUT, OLD, CFG, ARMS


def materialize_times(domain, folder, catalog, training, samples):
    source = ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1'/domain/'upstream_sequences_without_terminal.jsonl'
    upstream = sorted([json.loads(line) for line in source.open()], key=lambda s: s['user_id'])
    histories = [(r['user_id'], r['history']) for r in upstream]
    histories += [(r['user_id'], r['full_history']) for r in training+samples]
    wanted = {(u, catalog[i]) for u, hist in histories for i in hist}
    found = defaultdict(list)
    assert sha(RAW[domain]) == RAW_HASH[domain]
    with gzip.open(RAW[domain], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['reviewerID'], row['asin'])
            if key in wanted:
                found[key].append(row['unixReviewTime'])
    assert set(found) == wanted and all(len(v) == 1 for v in found.values())
    assert all(isinstance(v[0], int) and v[0] > 0 and v[0] % 86400 == 0 for v in found.values())
    dates = {k: v[0]//86400 for k, v in found.items()}
    for u, hist in histories:
        values = [dates[(u, catalog[i])] for i in hist]
        assert all(b >= a for a, b in zip(values, values[1:])), (domain, u)
    def pack(rows, upstream_training):
        n, length = len(rows), BEHAVIOR['maxlen']
        x, days, targets = [np.zeros((n, length), dtype=np.int64) for _ in range(3)]
        for j, row in enumerate(rows):
            full = row['history'] if upstream_training else row['full_history']
            if upstream_training:
                assert len(full) >= 2
                seq = full[-(length+1):]
                visible, labels = seq[:-1], seq[1:]
                targets[j, -len(labels):] = labels
                # Exactly the existing SeqDataset input/target alignment.
                assert len(visible) == min(len(full)-1, length)
            else:
                visible = full[-length:]
            assert len(visible)
            x[j, -len(visible):] = visible
            days[j, -len(visible):] = [dates[(row['user_id'], catalog[i])] for i in visible]
        content = dict(items=x, days=days, users=np.asarray([r['user_id'] for r in rows]), num_items=len(catalog)-1)
        if upstream_training:
            content['targets'] = targets
        return content
    np.savez_compressed(folder/'upstream.npz', **pack(upstream, True))
    np.savez_compressed(folder/'fit_histories.npz', **pack(training, False))
    np.savez_compressed(folder/'validation_histories.npz', **pack(samples, False))
    # Internal terminal labels used for fuser fitting are absent from the expert's
    # training histories, matching the existing frozen SASRec protocol.
    by_user = {r['user_id']: r['history'] for r in upstream}
    assert all(r['full_history'] == by_user[r['user_id']] for r in training)
    assert all(r['target'] not in r['full_history'] for r in training+samples)
    checks = dict(state='PASSED', raw_sha256=RAW_HASH[domain], raw_url=
        'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_'+
        ('Toys_and_Games' if domain == 'Toys' else 'Beauty')+'_5.json.gz',
        retained_visible_user_item_pairs=len(wanted), missing=0, ambiguous=0, nonmonotonic=0,
        upstream_users=len(upstream), upstream_supervised_positions=sum(min(len(r['history'])-1, 50) for r in upstream),
        fit_users=len(training), validation_users=len(samples), training_target_dates_not_packed=True,
        inference_target_dates_not_read=True, preserved_original_order=True, timestamp_resolution='days',
        source_scan='Raw public file scanned; only declared upstream/visible history pairs retained')
    write(folder/'time_alignment.json', checks)
    return source


def prepare():
    assert read(OLD/'status.json')['state'] == 'COMPLETED'
    assert read(OLD/'review/verification.json')['state'] == 'VERIFIED_SAVED_PREDICTIONS'
    tests = dict(behavior=B.tests(), fusion=C.tests())
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', tests)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}
    def use(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        manifest[str(path.relative_to(ROOT))] = digest
    for relative, digest in read(OLD/'manifest.json')['sha256'].items():
        if relative.startswith(('experiment/', 'GRAM/src/', '.cache/')):
            use(ROOT/relative, digest)
    for d in DOMAINS:
        p = OUT/d; p.mkdir()
        for name in ('samples.json', 'catalog.json', 'data_config.json'):
            use(OLD/d/name); shutil.copy2(OLD/d/name, p/name); use(p/name)
        training = [s for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        samples, dc = read(p/'samples.json'), read(p/'data_config.json')
        ck = torch.load(OLD/d/'user.pt', map_location='cpu')
        assert ck['fit_users'] == [r['user_id'] for r in training]
        assert len(samples) == 512 and set(ck['fit_users']).isdisjoint(s['user_id'] for s in samples)
        for key in ('seed', 'epochs', 'checkpoints', 'batch_size', 'lr', 'weight_decay', 'score_scale'):
            assert ck['config'][key] == CFG[key]
        source = materialize_times(d, p, read(p/'catalog.json')['items'], training, samples)
        use(source)
        old_behavior = torch.load(ROOT/dc['sasrec'], map_location='cpu')
        assert old_behavior['train_complete'] and old_behavior['architecture'] == 'legacy'
        for key in ('seed', 'hidden', 'maxlen', 'num_layers', 'num_heads', 'dropout', 'epochs', 'batch_size', 'weight_decay', 'label_smoothing', 'clip_norm'):
            assert old_behavior['config'][key] == BEHAVIOR[key], key
        assert old_behavior['config']['learning_rate'] == BEHAVIOR['lr']
        assert old_behavior['epoch'] == BEHAVIOR['epochs']
        for rel, digest in old_behavior['input_sha256'].items():
            if rel == str(source.relative_to(ROOT)):
                assert digest == sha(source)
        refs = [json.loads(line) for line in (OLD/d/'predictions.jsonl').read_text().splitlines()]
        assert len(refs) == len(samples)
        for sample, ref in zip(samples, refs):
            for k in ('user_id', 'target', 'history'):
                assert sample[k] == ref[k]
        write(p/'references.json', refs)
        for name in ('upstream.npz', 'fit_histories.npz', 'validation_histories.npz', 'time_alignment.json', 'references.json'):
            use(p/name)
        for name in ('samples.json', 'frequencies.npy'):
            use(SOURCE/d/name)
        for name in ('user.pt', 'signals.npz', 'ranks.npz', 'predictions.jsonl', 'train/item.npy', 'train/users.json',
                     'validation/item.npy', 'validation/user.npy'):
            use(OLD/d/name)
        use(ROOT/dc['parent_checkpoint'], dc['parent_sha256'])
        use(ROOT/dc['sasrec']); use(ROOT/dc['embedding'])
        write(p/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))
    for path in (PLAN, Path(__file__), OUT/'config.json', ROOT/'experiment/phase21/tisasrec_core.py',
                 ROOT/'experiment/phase21/train_time_behavior.py', ROOT/'experiment/phase21/run_time_behavior_expert.sh',
                 ROOT/'experiment/phase21/candidate_text_source.py'):
        use(path)
    for relative in manifest:
        path = ROOT/relative
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# SASRec 与 TiSASRec 并行对照\n\n状态：PREPARED，尚无 TiSASRec 效果结论。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest))), flush=True)


class Data(S.Data):
    def __init__(self, domain, training, variant, model):
        self.domain = domain
        self.samples = ([s for s in read(SOURCE/domain/'samples.json') if s['role'] == 'ranker_fit']
                        if training else read(OUT/domain/'samples.json'))
        self.n = len(self.samples)
        part = 'train' if training else 'validation'
        old = np.load(OLD/domain/part/'item.npy', mmap_mode='r')
        histories = np.load(OUT/domain/('fit_histories.npz' if training else 'validation_histories.npz'))
        assert histories['users'].tolist() == [s['user_id'] for s in self.samples]
        self.v = old.shape[1]
        assert old.shape == (self.n, self.v, C.DC)
        if training:
            assert [s['user_id'] for s in self.samples] == [s['user_id'] for s in read(OLD/domain/'train/users.json')]
        self.folder = OUT/domain/'features'/variant/part
        self.folder.mkdir(parents=True)
        self.user = np.lib.format.open_memmap(self.folder/'user.npy', mode='w+', dtype='float32', shape=(self.n, C.DU))
        self.item = np.lib.format.open_memmap(self.folder/'item.npy', mode='w+', dtype='float32', shape=old.shape)
        self.m1, self.m2 = np.zeros((self.n, C.DC)), np.zeros((self.n, C.DC))
        counts = np.load(SOURCE/domain/'frequencies.npy')
        model.eval()
        with torch.no_grad():
            for start in range(0, self.n, 64):
                stop = min(start+64, self.n)
                x, days = histories['items'][start:stop], histories['days'][start:stop]
                rel = B.intervals(x, days, BEHAVIOR['time_span'], variant == 'tisas_order')
                scores = model.predict(torch.from_numpy(x).cuda(), torch.from_numpy(rel).cuda())[:, 1:].cpu().numpy()
                assert np.isfinite(scores).all() and scores.shape == (stop-start, self.v)
                span = scores.max(1, keepdims=True) - scores.min(1, keepdims=True)
                scores = (scores-scores.min(1, keepdims=True)) / (span + 1e-8)
                for k, j in enumerate(range(start, stop)):
                    s = self.samples[j]
                    u, c = C.features(np.stack((scores[k], old[j, :, 1])), s['full_history'], counts)
                    np.testing.assert_array_equal(c[:, 1], old[j, :, 1])
                    self.user[j], self.item[j] = u, c
                    self.m1[j], self.m2[j] = c.mean(0, dtype=np.float64), np.square(c.astype(np.float64)).mean(0)
                progress(domain, 'features_'+part, completed=stop, total=self.n)
        self.user.flush(); self.item.flush()
        self.targets = np.asarray([s['target']-1 for s in self.samples], dtype=np.int64)
        self.folds = np.asarray([fold(s['user_id'], domain) for s in self.samples])
        write(self.folder/'metadata.json', dict(users=self.n, variant=variant, training=training,
            bge_scores_identical=True, target_used_in_features=False, target_dates_used=False))


def train(domain):
    global VARIANT
    p = OUT/domain
    samples = read(p/'samples.json')
    ck = torch.load(OLD/domain/'user.pt', map_location='cpu')
    signals = dict(full=np.load(OLD/domain/'signals.npz')['user'])
    old_u, old_c = [np.load(OLD/domain/'validation'/name, mmap_mode='r') for name in ('user.npy', 'item.npy')]
    for j in range(512):
        f, _ = C.numpy_score(ck['state_dict'], old_u[j], old_c[j], ck['mean'], ck['std'], 'user')
        np.testing.assert_allclose(f, signals['full'][j], atol=3e-6, rtol=2e-5)
    fits, checks = dict(full=dict(reused=True, selected_epochs=ck['selected_epochs'])), {}
    behavior_ranks = dict(full=np.array([C.target_rank(old_c[j, :, 0], s['target'], s['history']) for j, s in enumerate(samples)]))
    for variant in NEW:
        VARIANT = variant
        fit_behavior(p, variant, BEHAVIOR, lambda stage, **v: progress(domain, stage, **v), write)
        checkpoint = torch.load(p/'behavior'/variant/'model.pt', map_location='cpu')
        assert checkpoint['train_complete'] and checkpoint['epoch'] == BEHAVIOR['epochs']
        behavior = B.TiSASRec(checkpoint['num_items'], **{k: BEHAVIOR[k] for k in
            ('hidden', 'maxlen', 'num_layers', 'num_heads', 'dropout', 'time_span')}).cuda()
        behavior.load_state_dict(checkpoint['state_dict'])
        training, validation = Data(domain, True, variant, behavior), Data(domain, False, variant, behavior)
        assert [s['user_id'] for s in training.samples] == ck['fit_users']
        # Retain scores from the expert independently of the small fusion head.
        behavior_ranks[variant] = np.array([C.target_rank(validation.item[j, :, 0], s['target'], s['history']) for j, s in enumerate(samples)])
        del behavior, checkpoint
        gc.collect(); torch.cuda.empty_cache()
        S.OUT = OUT/'models'/variant; (S.OUT/domain).mkdir(parents=True)
        model, norm, fits[variant] = S.fit(training, 'user', torch.device('cuda:0'))
        signals[variant], checks[variant] = S.export(validation, model, norm, 'user', torch.device('cuda:0'))
        saved = torch.load(S.OUT/domain/'user.pt', map_location='cpu')
        assert saved['config'] == CFG and saved['fit_users'] == ck['fit_users'] and saved['parameters'] == 610
        for j in (0, 127, 255, 383, 511):
            f, w = C.numpy_score(saved['state_dict'], validation.user[j], validation.item[j], saved['mean'], saved['std'], 'user')
            np.testing.assert_allclose(f, signals[variant][j], atol=3e-6, rtol=2e-5)
            np.testing.assert_allclose(w[:, 0], np.full(len(w), w[0, 0]), atol=2e-7, rtol=1e-6)
        del model, training, validation
        gc.collect(); torch.cuda.empty_cache()
    S.OUT, VARIANT = OUT, None
    matched = [read(p/'behavior'/a/'summary.json') for a in NEW]
    for key in ('initial_state_sha256', 'shuffle_rng_final_sha256', 'parameters', 'optimizer_steps'):
        assert matched[0][key] == matched[1][key], key
    pure = {a: np.array([C.target_rank(signals[a][j], s['target'], s['history']) for j, s in enumerate(samples)]) for a in ARMS}
    np.savez(p/'signals.npz', **signals)
    np.savez_compressed(p/'pure_ranks.npz', **pure)
    np.savez_compressed(p/'behavior_ranks.npz', **behavior_ranks)
    write(p/'training.json', dict(fits=fits, checks=checks, full_score_replayed_users=512,
        pure_metrics={a: S.metrics(r) for a, r in pure.items()},
        behavior_metrics={a: S.metrics(r) for a, r in behavior_ranks.items()}, test_read=False))
    paths = [p/name for name in ('signals.npz', 'pure_ranks.npz', 'behavior_ranks.npz', 'training.json')]
    paths += list((p/'behavior').glob('*/*.pt')) + list((p/'behavior').glob('*/*.json'))
    paths += list((OUT/'models').glob('*/'+domain+'/*.pt'))
    write(p/'training_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))


def summarize():
    keys = ('native',)+tuple(a+s for a in ARMS for s in ('_late', '_guided'))
    results = {}
    lines = ['# SASRec 与 TiSASRec 并行对照', '', '状态：COMPLETED；两域各512名复用开发用户，seed0探索。', '',
             '| 域 | GRAM N10 | SAS+BGE | TiSAS+BGE | 等间隔TiSAS+BGE |', '|---|---:|---:|---:|---:|']
    for d in DOMAINS:
        p = OUT/d
        assert read(p/'status.json')['state'] == 'COMPLETED'
        assert read(p/'lease_status.json')['state'] == 'released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
        samples = read(p/'samples.json')
        rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        ranks = {key: [] for key in keys}
        for sample, row in zip(samples, rows):
            for k in ('user_id', 'target', 'history'):
                assert sample[k] == row[k]
            for key in keys:
                ids, values = np.array(row[key]['ids']), np.array(row[key]['scores'])
                keep = ~np.isin(ids, sample['history'])
                ordered = ids[keep][np.lexsort((ids[keep], -values[keep]))]
                positions = np.flatnonzero(ordered == sample['target'])
                rank = int(positions[0])+1 if len(positions) else 2**30
                assert rank == S.OLD.record_rank(row[key], sample)
                ranks[key].append(rank)
        ranks = {k: np.array(v) for k, v in ranks.items()}
        previous = np.load(OLD/d/'ranks.npz')
        for key, origin in [('native', 'native'), ('full_late', 'user_late'), ('full_guided', 'user_guided')]:
            np.testing.assert_array_equal(ranks[key], previous[origin])
        np.savez_compressed(p/'ranks.npz', users=np.array([s['user_id'] for s in samples]), **ranks)
        pure, signals = np.load(p/'pure_ranks.npz'), np.load(p/'signals.npz')
        for arm in ARMS:
            recomputed = [C.target_rank(signals[arm][j], s['target'], s['history']) for j, s in enumerate(samples)]
            np.testing.assert_array_equal(pure[arm], recomputed)
        pairs = [(a+'_guided', 'full_guided') for a in NEW]+[('tisas_guided', 'tisas_order_guided')]
        pairs += [(a+'_guided', b) for a in ARMS for b in ('native', a+'_late')]
        behavior_ranks = np.load(p/'behavior_ranks.npz')
        dates = np.load(p/'validation_histories.npz')
        unique_dates = np.array([len(set(t[x != 0].tolist())) for x, t in zip(dates['items'], dates['days'])])
        strata = {name: dict(users=int(mask.sum()), metrics={k: S.metrics(v[mask]) for k, v in ranks.items()})
                  for name, mask in [('one_date', unique_dates == 1), ('two_dates', unique_dates == 2),
                                     ('at_least_three_dates', unique_dates >= 3)] if mask.any()}
        result = dict(users=512, metrics={k: S.metrics(v) for k, v in ranks.items()},
            comparisons={a+'_vs_'+b: S.contrast(ranks[a], ranks[b], repeats=2000, seed=20261015) for a, b in pairs},
            behavior_comparisons={a+'_vs_'+b: S.contrast(behavior_ranks[a], behavior_ranks[b], repeats=2000, seed=20261015)
                                  for a, b in [('tisas', 'full'), ('tisas', 'tisas_order')]},
            visible_history_date_strata=strata,
            training=read(p/'training.json'), behavior_training={a: read(p/'behavior'/a/'summary.json') for a in NEW},
            time_alignment=read(p/'time_alignment.json'), test_read=False)
        results[d] = result; write(p/'summary.json', result)
        write(p/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', users=512,
            old_rank_arrays_matched=3, alternative_sort_implementation=True, generation_ranks=len(keys)*512))
        paths = [p/name for name in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json',
                                    'runtime_checks.json', 'smoke.json', 'training_manifest.json')]
        write(p/'output_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))
        lines.append('| '+d+' | '+' | '.join(f'{result["metrics"][k]["N10"]:.6f}' for k in
            ('native', 'full_guided', 'tisas_guided', 'tisas_order_guided'))+' |')
    write(OUT/'summary.json', dict(domains=results, seed=0, test_read=False))
    lines += ['', 'TiSASRec与等间隔对照：同结构、同初始化种子、同用户批次顺序、同200轮全目录CE；后者完全不使用真实日期。',
        'SASRec沿用现有CPU训练固定200轮checkpoint；TiSASRec为GPU训练，注意力/FFN/初始化不同，不能将二者差值直接归因于时间。',
        'TiSASRec按每个查询的可见前缀归一化，区别于公开实现的全序列归一化；不是原论文训练配方原样复现。',
        'BGE分数、用户融合器训练规则、GRAM及原相似商品提示、alpha=.3、beam50、评估用户保持一致。',
        '报告独立行为专家排序、融合后排序、同候选重排和完整引导生成；不自动与Qwen组合。',
        '一个seed、复用开发集；配对区间是描述性的未校正区间，不能当作正式显著性结论。',
        '只在TiSASRec优于现有SAS且优于等间隔对照时，才有理由将真实时间作为替换动机；单域收益不支持双域统一收益。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    global LEASE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['test', 'prepare', 'supervise', 'worker', 'train', 'generate', 'summarize'])
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args(); bind()
    def interrupted(signum, frame):
        raise InterruptedError('Time behavior experiment interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(dict(behavior=B.tests(), fuser=C.tests())))
        elif args.mode == 'prepare':
            prepare()
        elif args.mode == 'worker':
            return S.worker(args.domain)
        elif args.mode == 'supervise':
            S.supervise()
        elif args.mode in ('train', 'generate'):
            assert args.domain and args.lease_pid
            LEASE = S.LeaseHandle(args.lease_pid)
            assert LEASE.poll() is None
            if args.mode == 'train':
                torch.set_num_threads(2); torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.cuda.set_per_process_memory_fraction(CFG['allocation_limit_gib']*2**30 / torch.cuda.get_device_properties(0).total_memory)
                train(args.domain)
            else:
                T.Generation(args.domain, LEASE).run()
        else:
            summarize()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'worker':
            folder = OUT/args.domain if args.domain else OUT
            write(folder/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(), updated_at=stamp()))
        raise


if __name__ == '__main__':
    sys.exit(main())
