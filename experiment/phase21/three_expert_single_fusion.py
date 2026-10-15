"""Two vs three experts in the current user-level single fuser.

One new three-channel head and full replay of the exact current dual head.
Expert scoring remains LIME-Rec-derived; probability guidance AGRec-inspired.
No change to alpha, decoding, data splits or score normalization.
"""
import argparse
import gc
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
from experiment.phase21 import single_fusion_core as TWO
from experiment.phase21 import three_expert_single_fusion_core as C

ROOT, OLD, SOURCE, FULL = S.ROOT, S.OUT, S.SOURCE, S.FULL
OUT = ROOT/'artifacts/phase21/three_expert_single_fusion_20261015_v1'
PLAN = ROOT/'plan/第二十一阶段/21.30_用户级单融合器双三专家匹配对照_20261015.md'
REPORT = ROOT/'report/第二十一阶段/用户级单融合器双三专家匹配对照_20261015.md'
MODULE = 'experiment.phase21.three_expert_single_fusion'
DOMAINS, ARMS = ('Toys', 'Beauty'), ('two', 'three')
CFG = dict(S.CFG, experts=['SASRec', 'ItemCF', 'BGE'], arms=list(ARMS),
    user_features=C.USER_NAMES, item_features=C.ITEM_NAMES, parameters_two=610, parameters_three=835,
    active_parameters_two=418, active_parameters_three=579,
    question='incremental ItemCF contribution in the current single user-level fusion',
    primary_comparison='three_guided_vs_two_guided', alpha_learned=False,
    two_reference='stage21.24 exact user head; full512 NumPy and fresh generation replay',
    initialization='same seed0 original random shared columns/bias; new CF columns; zero output',
    matching='same architecture and training recipe; dimensions extend with expert count, not equal parameter count')
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp
LEASE = None


def progress(domain, stage, **values):
    if LEASE is not None:
        assert LEASE.poll() is None, 'GPU lease exited'
    row = dict(state='RUNNING', stage=stage, variant='three', pid=os.getpid(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), updated_at=stamp(), test_read=False, **values)
    write(OUT/domain/'status.json', row)
    print(json.dumps(dict(domain=domain, **row)), flush=True)


def bind():
    S.OUT, S.REPORT, S.MODULE, S.CFG, S.C = OUT, REPORT, MODULE, CFG, C
    S.progress, S.summarize = progress, summarize


def prepare():
    assert read(OLD/'status.json')['state'] == 'COMPLETED'
    assert read(OLD/'review/verification.json')['state'] == 'VERIFIED_SAVED_PREDICTIONS'
    checks = C.tests()
    assert checks['parameters_two'] == CFG['parameters_two'] and checks['parameters_three'] == CFG['parameters_three']
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}

    def use(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        manifest[str(path.relative_to(ROOT))] = digest

    old_manifest = read(OLD/'manifest.json')['sha256']
    for relative, digest in old_manifest.items():
        if relative.startswith(('experiment/', 'GRAM/src/', '.cache/')):
            use(ROOT/relative, digest)
    use(OLD/'review/verification.json')
    for d in DOMAINS:
        p = OUT/d; p.mkdir()
        for name in ('samples.json', 'catalog.json', 'data_config.json'):
            use(OLD/d/name)
            shutil.copy2(OLD/d/name, p/name); use(p/name)
        samples = read(p/'samples.json')
        training = [s for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        ck = torch.load(OLD/d/'user.pt', map_location='cpu')
        assert ck['fit_users'] == [s['user_id'] for s in training]
        assert set(ck['fit_users']).isdisjoint(s['user_id'] for s in samples)
        assert len(samples) == len({s['user_id'] for s in samples}) == 512
        assert len(training) == dict(Toys=2773, Beauty=3310)[d]
        assert ck['selected_epochs'] == dict(Toys=4, Beauty=12)[d]
        for key in ('seed', 'epochs', 'checkpoints', 'batch_size', 'lr', 'weight_decay', 'score_scale', 'alpha', 'beam_size', 'length_penalty'):
            assert ck['config'][key] == CFG[key]
        refs = [json.loads(line) for line in (OLD/d/'predictions.jsonl').read_text().splitlines()]
        assert len(refs) == len(samples)
        for s, r in zip(samples, refs):
            assert all(s[k] == r[k] for k in ('user_id', 'target', 'history'))
        write(p/'references.json', refs); use(p/'references.json')
        # Small real-data check before any GPU admission. The worker checks all
        # rows again after building its full three-channel feature tables.
        counts = np.load(SOURCE/d/'frequencies.npy')
        fit_index = read(OLD/d/'train/users.json')
        cache_checks = []
        for part, origin, selected in (('train', SOURCE, training), ('validation', FULL, samples)):
            raw = np.load(origin/d/'experts.npy', mmap_mode='r')
            old_u = np.load(OLD/d/part/'user.npy', mmap_mode='r')
            old_c = np.load(OLD/d/part/'item.npy', mmap_mode='r')
            for j in (0, len(selected)//2, len(selected)-1):
                idx = fit_index[j]['source_index'] if part == 'train' else selected[j]['full_validation_index']
                u, c = C.features(raw[idx], selected[j]['full_history'], counts)
                np.testing.assert_array_equal(u[C.TWO_USER_COLUMNS], old_u[j])
                np.testing.assert_array_equal(c[:, C.TWO_ITEM_COLUMNS], old_c[j])
                cache_checks.append(dict(part=part, row=j, shared_features_exact=True))
        write(p/'preflight.json', dict(state='PASSED', cached_features=cache_checks))
        use(p/'preflight.json')
        for name in ('user.pt', 'signals.npz', 'ranks.npz', 'pure_ranks.npz', 'predictions.jsonl', 'training.json',
                     'train/user.npy', 'train/item.npy', 'train/users.json', 'validation/user.npy', 'validation/item.npy'):
            use(OLD/d/name)
        for origin in (SOURCE, FULL):
            for name in ('samples.json', 'experts.npy'):
                path = origin/d/name
                use(path, old_manifest[str(path.relative_to(ROOT))])
        use(SOURCE/d/'frequencies.npy')
        dc = read(p/'data_config.json')
        use(ROOT/dc['parent_checkpoint'], dc['parent_sha256'])
        # The raw CF file is provenance; scores are taken from the exact old cache.
        use(ROOT/dc['itemcf'])
        assert read(ROOT/dc['itemcf'])['training_only']
        write(p/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))
    for path in (PLAN, Path(__file__), ROOT/'experiment/phase21/three_expert_single_fusion_core.py',
                 ROOT/'experiment/phase21/run_three_expert_single_fusion.sh', OUT/'config.json'):
        use(path)
    for relative in manifest:
        path = ROOT/relative
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 用户级单融合器双三专家匹配对照\n\n状态：PREPARED。两组同训练规则、同生成设置；尚无三专家新效果结论。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), checks=checks)), flush=True)


class Data(S.Data):
    def __init__(self, domain, training):
        super().__init__(domain, training)
        part = 'train' if training else 'validation'
        old_user = np.load(OLD/domain/part/'user.npy', mmap_mode='r')
        old_item = np.load(OLD/domain/part/'item.npy', mmap_mode='r')
        np.testing.assert_array_equal(self.user[:, C.TWO_USER_COLUMNS], old_user)
        for start in range(0, self.n, 32):
            stop = min(start+32, self.n)
            np.testing.assert_array_equal(np.take(self.item[start:stop], C.TWO_ITEM_COLUMNS, axis=-1),
                                          old_item[start:stop])
        # Also verify explicit row slices to make the column convention clear.
        for j in (0, self.n//2, self.n-1):
            np.testing.assert_array_equal(self.item[j][:, C.TWO_ITEM_COLUMNS], old_item[j])
        write(self.folder/'shared_feature_checks.json', dict(state='PASSED', users=self.n,
            shared_user_columns=C.TWO_USER_COLUMNS, shared_item_columns=C.TWO_ITEM_COLUMNS,
            all_shared_features_exact=True, target_used_in_features=False))


def export_three(data, model, norm):
    result = np.empty((data.n, data.v), dtype=np.float32)
    weights = np.empty((data.n, 3), dtype=np.float32)
    nt = tuple(torch.from_numpy(x).cuda() for x in norm)
    with torch.no_grad():
        for start in range(0, data.n, CFG['batch_size']):
            stop = min(start+CFG['batch_size'], data.n)
            u = torch.from_numpy(np.array(data.user[start:stop])).cuda()
            c = torch.from_numpy(np.array(data.item[start:stop])).cuda()
            f, w = C.score(model, u, c, *nt, 'user')
            assert torch.isfinite(f).all() and w.shape == (stop-start, 1, 3)
            torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]))
            assert ((f >= c[..., :3].min(-1).values-2e-6) & (f <= c[..., :3].max(-1).values+2e-6)).all()
            result[start:stop], weights[start:stop] = f.cpu().numpy(), w[:, 0].cpu().numpy()
    return result, weights


def train(domain):
    p = OUT/domain
    training, validation = Data(domain, True), Data(domain, False)
    old_ck = torch.load(OLD/domain/'user.pt', map_location='cpu')
    assert [s['user_id'] for s in training.samples] == old_ck['fit_users']
    assert set(old_ck['fit_users']).isdisjoint(s['user_id'] for s in validation.samples)
    shared_norm = training.norm(np.arange(training.n))
    for actual, expected in zip(shared_norm, (old_ck['mean'], old_ck['std'])):
        np.testing.assert_allclose(actual[C.TWO_INPUT_COLUMNS], expected, atol=1e-6, rtol=2e-6)
    signals = dict(two=np.load(OLD/domain/'signals.npz')['user'])
    old_u = np.load(OLD/domain/'validation/user.npy', mmap_mode='r')
    old_c = np.load(OLD/domain/'validation/item.npy', mmap_mode='r')
    two_weights = []
    progress(domain, 'replay_two_scores', completed=0, total=512)
    for j in range(512):
        f, w = TWO.numpy_score(old_ck['state_dict'], old_u[j], old_c[j], old_ck['mean'], old_ck['std'], 'user')
        np.testing.assert_allclose(f, signals['two'][j], atol=3e-6, rtol=2e-5)
        two_weights.append(w[0].tolist())
    model, norm, fit = S.fit(training, 'user', torch.device('cuda:0'))
    signals['three'], weights = export_three(validation, model, norm)
    saved = torch.load(p/'user.pt', map_location='cpu')
    assert saved['fit_users'] == old_ck['fit_users'] and saved['config'] == CFG
    for j in (0, 127, 255, 383, 511):
        actual, w = C.numpy_score(saved['state_dict'], validation.user[j], validation.item[j], saved['mean'], saved['std'], 'user')
        np.testing.assert_allclose(actual, signals['three'][j], atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(w, np.broadcast_to(weights[j], w.shape), atol=3e-6, rtol=2e-5)
    pure = {a: np.asarray([C.target_rank(signals[a][j], s['target'], s['history'])
            for j, s in enumerate(validation.samples)]) for a in ARMS}
    np.testing.assert_array_equal(pure['two'], np.load(OLD/domain/'pure_ranks.npz')['user'])
    np.savez(p/'signals.npz', **signals)
    np.savez_compressed(p/'pure_ranks.npz', users=np.asarray([s['user_id'] for s in validation.samples]), **pure)
    np.savez(p/'weights.npz', users=np.asarray([s['user_id'] for s in validation.samples]),
             two=np.asarray(two_weights), three=weights)
    write(p/'training.json', dict(fits=dict(two=dict(reused=True, selected_epochs=old_ck['selected_epochs'],
        parameters=610, refit_users=len(old_ck['fit_users'])), three=fit),
        checks=dict(two_scores_replayed_users=512, three_numpy_forward_users=[0, 127, 255, 383, 511],
            shared_features_exact=True, shared_full_training_norm_matched=True, user_shared_weights=True),
        pure_metrics={a: S.metrics(v) for a, v in pure.items()},
        three_weight_statistics=dict(order=list(C.EXPERTS), mean=weights.mean(0).tolist(),
            p10=np.quantile(weights, .1, axis=0).tolist(), p90=np.quantile(weights, .9, axis=0).tolist()),
        note='Weights are not proof of expert utility; the matched two-vs-three contrast tests ItemCF.', test_read=False))
    paths = [p/n for n in ('user.pt', 'signals.npz', 'pure_ranks.npz', 'weights.npz', 'training.json')]
    paths += [p/part/name for part in ('train', 'validation') for name in ('manifest.json', 'shared_feature_checks.json')]
    write(p/'training_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))
    del model, training, validation
    gc.collect(); torch.cuda.empty_cache()


class Generation(S.Generation):
    def run(self):
        self.stage = 'native_zero_two_smoke'; self.status()
        checks = []
        for j in (0, 256, 511):
            ctx = self.context(j)
            native, plain = self.decode(ctx), self.decode(ctx, native_entry=True, record=False)
            q = self.index.conditional_logs(self.signals['two'][j], temperature_scale=CFG['score_scale'])
            zero, two = self.decode(ctx, q, 0.), self.decode(ctx, q, CFG['alpha'])
            for other in (plain, zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'], other['scores'])
            ref = self.reference[self.samples[j]['user_id']]
            assert two['ids'] == ref['user_guided']['ids']
            np.testing.assert_allclose(two['scores'], ref['user_guided']['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user_id=self.samples[j]['user_id'], alpha0_exact=True, original_two_reproduced=True))
            del ctx, native, plain, zero, two
        write(self.folder/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'matched_generation'
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j, sample in enumerate(self.samples):
                begin = time.monotonic()
                ctx, ref = self.context(j), self.reference[sample['user_id']]
                native = self.decode(ctx)
                assert native['ids'] == ref['native']['ids']
                np.testing.assert_allclose(native['scores'], ref['native']['scores'], atol=2e-6, rtol=1e-5)
                row = dict(user_id=sample['user_id'], target=sample['target'], history=sample['history'],
                    native=dict(ids=native['ids'], scores=native['scores'].tolist()))
                for arm in ARMS:
                    q = self.index.conditional_logs(self.signals[arm][j], temperature_scale=CFG['score_scale'])
                    late = native['guide'].rescore(native['paths'], q, CFG['alpha'])
                    guided = self.decode(ctx, q, CFG['alpha'])
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                    if arm == 'two':
                        for suffix in ('late', 'guided'):
                            a, b = row[arm+'_'+suffix], ref['user_'+suffix]
                            assert a['ids'] == b['ids']
                            np.testing.assert_allclose(a['scores'], b['scores'], atol=2e-6, rtol=1e-5)
                    del guided
                stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1) % 8 == 0 or j == 511:
                    self.status()
                del ctx, native
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', native_reproduced_users=512,
            two_guided_reproduced_users=512, two_late_reproduced_users=512, gram_model_digest_unchanged=True,
            gram_optimizer_steps=0))
        self.stage = 'complete'; self.status('COMPLETED')


def summarize():
    results = {}
    keys = ('native', 'two_late', 'two_guided', 'three_late', 'three_guided')
    lines = ['# 用户级单融合器双三专家匹配对照', '', '## Material Passport', '',
        '状态：COMPLETED；保存预测独立排序重算通过。两域各512个反复使用的开发用户，小头seed0；探索性对照。', '',
        '| 域 | GRAM N10 | 双专家 N10 | 三专家 N10 | 三−双绝对差 | 三对双相对变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        p = OUT/d
        assert read(p/'status.json')['state'] == 'COMPLETED'
        assert read(p/'lease_status.json')['state'] == 'released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
        for manifest in (OUT/'manifest.json', p/'training_manifest.json', p/'train/manifest.json', p/'validation/manifest.json'):
            for path, digest in read(manifest)['sha256'].items():
                assert sha(ROOT/path) == digest, path
        samples = read(p/'samples.json')
        rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        rr = {k: [] for k in keys}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            for key in keys:
                pred = row[key]
                assert len(pred['ids']) == len(set(pred['ids'])) == len(pred['scores']) == 50
                actual = S.OLD.record_rank(pred, sample)
                values = {int(i): float(v) for i, v in zip(pred['ids'], pred['scores']) if i not in sample['history']}
                y = sample['target']
                other = 2**30 if y not in values else 1+sum(v > values[y] or (v == values[y] and i < y) for i, v in values.items())
                assert actual == other
                rr[key].append(actual)
        rr = {k: np.asarray(v) for k, v in rr.items()}
        old = np.load(OLD/d/'ranks.npz')
        for a, b in (('native', 'native'), ('two_late', 'user_late'), ('two_guided', 'user_guided')):
            np.testing.assert_array_equal(rr[a], old[b])
        np.savez_compressed(p/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **rr)
        pure, signals = np.load(p/'pure_ranks.npz'), np.load(p/'signals.npz')
        for arm in ARMS:
            expected = []
            for j, s in enumerate(samples):
                values = signals[arm][j]; ids = np.arange(1, len(values)+1)
                keep = ~np.isin(ids, s['history'])
                order = ids[keep][np.lexsort((ids[keep], -values[keep]))]
                pos = np.flatnonzero(order == s['target'])
                expected.append(int(pos[0])+1 if len(pos) else len(values)+1)
            np.testing.assert_array_equal(expected, pure[arm])
        pairs = [('three_guided', 'two_guided'), ('three_late', 'two_late')]
        pairs += [(a+'_guided', b) for a in ARMS for b in ('native', a+'_late')]
        short = np.asarray([len(s['full_history']) <= 20 for s in samples])
        results[d] = dict(users=512, metrics={k: S.metrics(v) for k, v in rr.items()},
            comparisons={a+'_vs_'+b: S.contrast(rr[a], rr[b], repeats=2000, seed=20261015) for a, b in pairs},
            pure_three_vs_two=S.contrast(pure['three'], pure['two'], repeats=2000, seed=20261015),
            history_at_most20=dict(n=int(short.sum()), metrics={k: S.metrics(v[short]) for k, v in rr.items()}),
            training=read(p/'training.json'), test_read=False)
        write(p/'summary.json', results[d])
        write(p/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', users=512,
            generation_ranks=len(keys)*512, pure_ranks=len(ARMS)*512, old_rank_arrays_matched=3,
            independent_count_greater_rank=True, test_read=False))
        paths = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json',
                              'runtime_checks.json', 'smoke.json', 'training_manifest.json')]
        write(p/'output_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))
        m = results[d]['metrics']; delta = m['three_guided']['N10']-m['two_guided']['N10']
        lines.append(f'| {d} | {m["native"]["N10"]:.6f} | {m["two_guided"]["N10"]:.6f} | {m["three_guided"]["N10"]:.6f} | {delta:+.6f} | {100*delta/m["two_guided"]["N10"]:+.2f}% |')
    write(OUT/'summary.json', dict(domains=results, test_read=False, seed=0))
    write(OUT/'interpretation_checks.json', dict(covered=11, total=11, status='CAUTION',
        items=['domains separate', 'user is analysis unit', 'reused development cohort disclosed',
               'history subgroup descriptive only', 'counts and all512 denominators retained',
               'not independent replication', 'all arms and missing-target users retained',
               'no added outcome-driven comparisons', 'frozen protocol and no scientific retry',
               'conditional contribution not unique mechanism', 'no causal direction inferred']))
    lines += ['', '两组均为一个用户级单融合器；双专家模型复用21.24，并在本轮全量重算分数与重放生成。三专家独立拟合，同预算内部CE选轮，不读取开发指标选轮。', '',
        '加入ItemCF带来必要的输入输出扩展，总参数610→835，隐藏宽32不变；共同输入的随机初始化和特征已核对。此为同结构专家扩展，不是严格等参数量比较。', '',
        '外层alpha=.3、scale20、beam50、过滤与GRAM均固定。所有预定组完整真实生成，未按纯排序筛选。主比较为三专家−双专家；完整指标、配对描述性区间和救回/误伤见summary.json。', '',
        '区间未作多重比较校正，不包含上游训练及研究选择不确定性；一个seed、反复使用开发池不能代替正式确认。权重非零不等于专家必要，本轮只检验ItemCF条件贡献，不证明其余两个专家在三专家结构中分别必要。', '',
        '完成后停止，等待结果分析确定组合，不自动追加alpha学习、其他消融或论文实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    global LEASE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'train', 'generate', 'summarize'))
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args(); bind()
    def interrupted(signum, frame):
        raise InterruptedError('Three-expert single-fusion process interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(C.tests()))
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
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            if args.mode == 'train':
                torch.set_num_threads(2); torch.set_num_interop_threads(1)
                torch.cuda.set_per_process_memory_fraction(CFG['allocation_limit_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
                train(args.domain)
            else:
                Generation(args.domain, LEASE).run()
        else:
            summarize()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'worker':
            p = OUT/args.domain if args.mode in ('train', 'generate') else OUT
            write(p/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(), updated_at=stamp()))
        raise


if __name__ == '__main__':
    sys.exit(main())
