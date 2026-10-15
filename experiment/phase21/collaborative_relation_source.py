"""Matched CF support outside GRAM's static hints; frozen sources, real decoding.

ItemCF/BGE follow the project's LIME-Rec-derived expert pipeline. The existing
probability guide is AGRec-inspired. This is a mechanism screen, not a novelty
claim. Reuses the previously verified user fitter and GPU lifecycle unchanged.
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
from types import SimpleNamespace

import numpy as np
import torch

from experiment.phase21 import single_fusion as S
from experiment.phase21 import single_fusion_core as C
from experiment.phase21 import collaborative_relation_source_core as R
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20.lime_clean_gram_core import T5

ROOT, OLD, SOURCE, FULL = S.ROOT, S.OUT, S.SOURCE, S.FULL
FIRST = ROOT/'artifacts/phase21/candidate_text_source_20261015_v1'
OUT = ROOT/'artifacts/phase21/collaborative_relation_source_20261015_v1'
PLAN = ROOT/'plan/第二十一阶段/21.28_协同关系超出静态邻居的匹配验证_20261015.md'
REPORT = ROOT/'report/第二十一阶段/协同关系超出静态邻居的匹配验证_20261015.md'
MODULE = 'experiment.phase21.collaborative_relation_source'
DOMAINS = ('Toys', 'Beauty')
NEW = ('cf_full', 'cf_prompt', 'cf_matched')
ARMS = NEW+('text_only', 'cf_only')
CFG = dict(S.CFG, experts=['ItemCF', 'BGE'], arms=list(ARMS),
    user_features=[s.replace('sas', 'cf') for s in C.USER_NAMES],
    item_features=[s.replace('sas', 'cf') for s in C.ITEM_NAMES],
    question='weighted collaborative relations beyond visible GRAM neighbor hints',
    candidate_gate=False, alpha_learned=False, history_policy='CF last20 linear; BGE original full_history; <=20 subgroup',
    primary_comparison='cf_full_guided_vs_cf_prompt_guided',
    source_control='per-source equal edge counts and total CF weight; overlap allowed',
    sas_text_reference='reuse verified stage21.24 user head outputs; three-user fresh parity',
    single_expert_controls=['text_only', 'cf_only'])
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp
LEASE = None
VARIANT = None


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


def audit_hints(catalog, hints, tokenizer):
    """All catalog prefixes checked; actual collator checked at fixed positions."""
    assert len(hints) == len(catalog['items'])-1
    prefixes = []
    for i, near in enumerate(hints, 1):
        assert len(set(near)) == len(near) and i not in near
        prefix = 'item: '+catalog['lexical_ids'][i]+'; similar items: '+', '.join(catalog['lexical_ids'][int(j)] for j in near)+';'
        assert catalog['passages'][i].startswith(prefix), ('hint/catalog mismatch', i)
        prefixes.append(prefix)
    encoded = tokenizer(prefixes, add_special_tokens=False)['input_ids']
    filtered = [[t for t in row if t not in (1820, 9175)] for row in encoded]
    lengths = np.asarray([len(row) for row in filtered])
    assert lengths.max() <= 127, 'Hint truncation requires a revised visible-edge protocol'
    sys.path.insert(0, str(ROOT/'GRAM/src'))
    from processor import CollatorGRAM
    collator = CollatorGRAM(tokenizer, SimpleNamespace(item_prompt_max_len=128,
        target_max_len=32, max_his=20, item_id_type='split', hierarchical_id_type='unused'))
    chosen = sorted(set([1, len(hints)//2, len(hints), int(lengths.argmax())+1]))
    for i in chosen:
        actual, mask = collator.encode_texts_split([['What would user purchase ?', catalog['passages'][i]]], tokenizer)
        actual = actual[0, 1][mask[0, 1]].tolist()
        assert filtered[i-1] == actual[:len(filtered[i-1])], ('visible-prefix parity', i)
    return dict(all_catalog_hint_texts_identical=True, all_hint_prefixes_fit=True,
        catalog_items=len(hints), max_prefix_tokens=int(lengths.max()), prompt_limit=128,
        collator_checked_item_ids=chosen, array_row='catalog ID minus1', array_values='1-based catalog IDs',
        targets_accessed=False)


def prepare():
    assert read(FIRST/'status.json')['state'] == 'COMPLETED'
    assert read(FIRST/'review/verification.json')['state'] == 'VERIFIED_SAVED_OUTPUTS'
    tests = dict(relations=R.tests(), fusion=C.tests())
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'tests.json', tests)
    write(OUT/'config.json', CFG)
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
    use(FIRST/'review/verification.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(T5), local_files_only=True)
    for d in DOMAINS:
        p = OUT/d
        p.mkdir()
        for name in ('samples.json', 'catalog.json', 'data_config.json'):
            use(OLD/d/name)
            shutil.copy2(OLD/d/name, p/name)
            use(p/name)
        samples = read(p/'samples.json')
        train = [s for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        ck = torch.load(OLD/d/'user.pt', map_location='cpu')
        assert ck['fit_users'] == [s['user_id'] for s in train]
        assert set(ck['fit_users']).isdisjoint(s['user_id'] for s in samples)
        assert len(samples) == len({s['user_id'] for s in samples}) == 512
        assert len(train) == dict(Toys=2773, Beauty=3310)[d]
        refs = [json.loads(line) for line in (OLD/d/'predictions.jsonl').read_text().splitlines()]
        assert len(refs) == len(samples)
        for sample, ref in zip(samples, refs):
            assert all(sample[k] == ref[k] for k in ('user_id', 'target', 'history'))
        write(p/'references.json', refs)
        use(p/'references.json')
        catalog, dc = read(p/'catalog.json'), read(p/'data_config.json')
        nearpath = ROOT/f'artifacts/phase20/lime_clean_sasrec_{d.lower()}_20260920_v1/neighbor_ids.npy'
        hints = np.load(nearpath)
        checks = audit_hints(catalog, hints, tokenizer)
        g, stats = R.graphs(read(ROOT/dc['itemcf']), catalog['items'][1:], hints)
        np.save(p/'hint_ids.npy', hints)
        write(p/'hint_checks.json', checks)
        write(p/'graph_summary.json', stats)
        for arm, graph in g.items():
            R.save(p/(arm+'_graph.npz'), graph)
            use(p/(arm+'_graph.npz'))
        for path in (nearpath, ROOT/dc['itemcf'], p/'hint_ids.npy', p/'hint_checks.json', p/'graph_summary.json'):
            use(path)
        use(ROOT/dc['parent_checkpoint'], dc['parent_sha256'])
        for name in ('user.pt', 'signals.npz', 'ranks.npz', 'predictions.jsonl', 'train/users.json',
                     'train/item.npy', 'validation/item.npy', 'validation/user.npy'):
            use(OLD/d/name)
        for name in ('samples.json', 'frequencies.npy', 'experts.npy'):
            use(SOURCE/d/name)
        for name in ('samples.json', 'experts.npy'):
            use(FULL/d/name)
        write(p/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))
        print(json.dumps(dict(domain=d, state='PREPARED', hint_checks=checks, graph=stats)), flush=True)
    for path in (PLAN, Path(__file__), ROOT/'experiment/phase21/collaborative_relation_source_core.py',
                 ROOT/'experiment/phase21/candidate_text_source_core.py', ROOT/'experiment/phase21/run_collaborative_relation_source.sh',
                 ROOT/'third_party/LIME-Rec/lime_rec/experts.py', OUT/'config.json'):
        use(path)
    for relative in manifest:
        path = ROOT/relative
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 协同关系超出静态邻居的匹配验证\n\n状态：PREPARED，五组预定真实生成对照；尚无新效果结论。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), tests=tests)), flush=True)


def build_scores(domain):
    """Author/raw/cache parity, target-free scores, and separate dev diagnostics."""
    p = OUT/domain
    graph = {a: R.load(p/(a+'_graph.npz')) for a in R.GRAPH_ARMS}
    catalog, dc = read(p/'catalog.json'), read(p/'data_config.json')
    hints = np.load(p/'hint_ids.npy')
    sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
    from lime_rec.experts import ItemCFExpert
    author = ItemCFExpert(str(ROOT/dc['itemcf']), catalog['items'][1:])
    receipts, diagnostics = {}, []
    for part in ('train', 'validation'):
        raw_samples = read((SOURCE if part == 'train' else FULL)/domain/'samples.json')
        if part == 'train':
            ix = [j for j, s in enumerate(raw_samples) if s['role'] == 'ranker_fit']
            samples = [raw_samples[j] for j in ix]
            assert [s['user_id'] for s in samples] == [s['user_id'] for s in read(OLD/domain/'train/users.json')]
        else:
            samples = read(p/'samples.json')
            ix = [s['full_validation_index'] for s in samples]
        cache = np.load((SOURCE if part == 'train' else FULL)/domain/'experts.npy', mmap_mode='r')
        target = p/'scores'/part
        target.mkdir(parents=True)
        arrays = {a: np.lib.format.open_memmap(target/(a+'.npy'), mode='w+', dtype='float32',
                  shape=(len(samples), len(catalog['items'])-1)) for a in NEW}
        err, decomposition = 0., 0.
        for j, (sample, index) in enumerate(zip(samples, ix)):
            assert all(sample[k] == raw_samples[index][k] for k in ('user_id', 'full_history', 'history', 'target'))
            assert sample['history'] == sample['full_history'][-20:]
            raw = {a: R.score(sample['full_history'], graph[a]) for a in R.GRAPH_ARMS}
            expected = author.score(str(sample['user_id']), [catalog['items'][i] for i in sample['full_history']])
            np.testing.assert_array_equal(raw['cf_full'], expected)
            delta = float(np.max(np.abs(raw['cf_full']-raw['cf_prompt']-raw['cf_outside'])))
            decomposition = max(decomposition, delta)
            np.testing.assert_allclose(raw['cf_full'], raw['cf_prompt']+raw['cf_outside'], atol=2e-6, rtol=2e-6)
            for arm in NEW:
                arrays[arm][j] = R.normalize(raw[arm])
            err = max(err, float(np.max(np.abs(arrays['cf_full'][j]-cache[index, 1]))))
            np.testing.assert_allclose(arrays['cf_full'][j], cache[index, 1], atol=3e-6, rtol=2e-5)
            if part == 'validation':
                y = sample['target']-1
                listed = set(hints[np.asarray(sample['history'])-1].ravel().tolist())
                diagnostics.append(dict(user_id=sample['user_id'], target=sample['target'],
                    target_listed_any_history_hint=sample['target'] in listed,
                    target_raw_support={a: float(s[y]) for a, s in raw.items()},
                    total_raw_mass={a: float(s.sum(dtype=np.float64)) for a, s in raw.items()},
                    supported_candidates={a: int((s > 0).sum()) for a, s in raw.items()},
                    pure_cf_ranks={a: C.target_rank(s, sample['target'], sample['history']) for a, s in raw.items()}))
            if j == 0 or (j+1) % 256 == 0 or j+1 == len(samples):
                progress(domain, 'relation_scores_'+part, completed=j+1, total=len(samples))
        for a in arrays.values():
            a.flush()
        receipts[part] = dict(users=len(samples), author_raw_exact=True, max_cached_normalized_error=err,
            max_decomposition_error=decomposition, graph_and_scores_target_free=True)
    write(p/'score_checks.json', receipts)
    write(p/'support_diagnostics.json', diagnostics)


class Data(S.Data):
    def __init__(self, domain, training, variant):
        self.domain = domain
        self.samples = ([s for s in read(SOURCE/domain/'samples.json') if s['role'] == 'ranker_fit']
                        if training else read(OUT/domain/'samples.json'))
        self.n = len(self.samples)
        part = 'train' if training else 'validation'
        old = np.load(OLD/domain/part/'item.npy', mmap_mode='r')
        cf = np.load(OUT/domain/'scores'/part/(variant+'.npy'), mmap_mode='r')
        self.v = old.shape[1]
        assert cf.shape == (self.n, self.v) and old.shape == (self.n, self.v, C.DC)
        self.folder = OUT/domain/'features'/variant/part
        self.folder.mkdir(parents=True)
        self.user = np.lib.format.open_memmap(self.folder/'user.npy', mode='w+', dtype='float32', shape=(self.n, C.DU))
        self.item = np.lib.format.open_memmap(self.folder/'item.npy', mode='w+', dtype='float32', shape=old.shape)
        self.m1, self.m2 = np.zeros((self.n, C.DC)), np.zeros((self.n, C.DC))
        counts = np.load(SOURCE/domain/'frequencies.npy')
        for j, sample in enumerate(self.samples):
            u, c = C.features(np.stack((cf[j], old[j, :, 1])), sample['full_history'], counts)
            np.testing.assert_array_equal(c[:, 1], old[j, :, 1])
            self.user[j], self.item[j] = u, c
            self.m1[j] = c.mean(0, dtype=np.float64)
            self.m2[j] = np.square(c.astype(np.float64)).mean(0)
            if j == 0 or (j+1) % 256 == 0 or j+1 == self.n:
                progress(domain, 'features_'+part, completed=j+1, total=self.n)
        self.user.flush(); self.item.flush()
        self.targets = np.asarray([s['target']-1 for s in self.samples], dtype=np.int64)
        self.folds = np.asarray([fold(s['user_id'], domain) for s in self.samples])
        write(self.folder/'metadata.json', dict(users=self.n, variant=variant, training=training,
            bge_scores_identical=True, target_used_in_features=False, experts=['ItemCF', 'BGE'],
            user_features=CFG['user_features'], item_features=CFG['item_features'],
            users_order=[s['user_id'] for s in self.samples]))


def train(domain):
    global VARIANT
    build_scores(domain)
    p = OUT/domain
    samples = read(p/'samples.json')
    old_item = np.load(OLD/domain/'validation/item.npy', mmap_mode='r')
    signals = dict(sas_text=np.load(OLD/domain/'signals.npz')['user'],
        text_only=np.array(old_item[:, :, 1]), cf_only=np.load(p/'scores/validation/cf_full.npy'))
    ck = torch.load(OLD/domain/'user.pt', map_location='cpu')
    old_user = np.load(OLD/domain/'validation/user.npy', mmap_mode='r')
    for j in range(512):
        f, _ = C.numpy_score(ck['state_dict'], old_user[j], old_item[j], ck['mean'], ck['std'], 'user')
        np.testing.assert_allclose(f, signals['sas_text'][j], atol=3e-6, rtol=2e-5)
    fits = {arm: dict(parameters=0, weight=1.) for arm in ('text_only', 'cf_only')}
    checks = {}
    for variant in NEW:
        VARIANT = variant
        training, validation = Data(domain, True, variant), Data(domain, False, variant)
        assert [s['user_id'] for s in training.samples] == ck['fit_users']
        assert set(ck['fit_users']).isdisjoint(s['user_id'] for s in validation.samples)
        S.OUT = OUT/'models'/variant
        (S.OUT/domain).mkdir(parents=True)
        model, norm, fits[variant] = S.fit(training, 'user', torch.device('cuda:0'))
        signals[variant], checks[variant] = S.export(validation, model, norm, 'user', torch.device('cuda:0'))
        saved = torch.load(S.OUT/domain/'user.pt', map_location='cpu')
        ww = []
        for j in (0, 127, 255, 383, 511):
            f, w = C.numpy_score(saved['state_dict'], validation.user[j], validation.item[j], saved['mean'], saved['std'], 'user')
            np.testing.assert_allclose(f, signals[variant][j], atol=3e-6, rtol=2e-5)
            np.testing.assert_allclose(w[:, 0], np.full(len(w), w[0, 0]), atol=2e-7)
            ww.append(float(w[0, 0]))
        checks[variant]['sample_user_cf_weights'] = ww
        del model, training, validation
        gc.collect(); torch.cuda.empty_cache()
    S.OUT = OUT
    VARIANT = None
    pure = {a: np.asarray([C.target_rank(signals[a][j], s['target'], s['history']) for j, s in enumerate(samples)]) for a in ARMS+('sas_text',)}
    np.savez(p/'signals.npz', **signals)
    np.savez_compressed(p/'pure_ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **pure)
    write(p/'training.json', dict(fits=fits, checks=checks, reference_scores_replayed_users=512,
        pure_metrics={a: S.metrics(r) for a, r in pure.items()}, test_read=False))
    paths = [p/name for name in ('signals.npz', 'pure_ranks.npz', 'training.json', 'score_checks.json', 'support_diagnostics.json')]
    paths += list((OUT/'models').glob('*/'+domain+'/*.pt'))
    write(p/'training_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))


class Generation(S.Generation):
    def run(self):
        self.stage = 'native_zero_reference_smoke'; self.status()
        checks = []
        for j in (0, 256, 511):
            ctx = self.context(j)
            native = self.decode(ctx)
            plain = self.decode(ctx, native_entry=True, record=False)
            q = self.index.conditional_logs(self.signals['sas_text'][j], temperature_scale=CFG['score_scale'])
            zero, reference = self.decode(ctx, q, 0.), self.decode(ctx, q, CFG['alpha'])
            for other in (plain, zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'], other['scores'])
            ref = self.reference[self.samples[j]['user_id']]
            assert reference['ids'] == ref['user_guided']['ids']
            np.testing.assert_allclose(reference['scores'], ref['user_guided']['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user_id=self.samples[j]['user_id'], alpha0_exact=True, sas_text_reproduced=True))
            del ctx, native, plain, zero, reference
        write(self.folder/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'matched_generation'
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j, sample in enumerate(self.samples):
                begin = time.monotonic()
                ctx = self.context(j)
                native = self.decode(ctx)
                ref = self.reference[sample['user_id']]
                assert native['ids'] == ref['native']['ids']
                np.testing.assert_allclose(native['scores'], ref['native']['scores'], atol=2e-6, rtol=1e-5)
                row = dict(user_id=sample['user_id'], target=sample['target'], history=sample['history'],
                    native=dict(ids=native['ids'], scores=native['scores'].tolist()),
                    sas_text_guided=ref['user_guided'], sas_text_late=ref['user_late'])
                for arm in ARMS:
                    q = self.index.conditional_logs(self.signals[arm][j], temperature_scale=CFG['score_scale'])
                    late = native['guide'].rescore(native['paths'], q, CFG['alpha'])
                    guided = self.decode(ctx, q, CFG['alpha'])
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                    del guided
                stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                self.users_completed = j+1
                self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1) % 8 == 0 or j == 511:
                    self.status()
                del ctx, native
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', gram_model_digest_unchanged=True,
            native_reproduced_users=512, sas_text_fresh_reproduced_users=3, sas_text_saved_reused_users=512,
            gram_optimizer_steps=0))
        self.stage = 'complete'; self.status('COMPLETED')


def summarize():
    results = {}
    keys = ('native', 'sas_text_late', 'sas_text_guided')+tuple(a+s for a in ARMS for s in ('_late', '_guided'))
    lines = ['# 协同关系超出静态邻居的匹配验证', '', '状态：COMPLETED；保存预测独立排序重算通过。两域各512个反复使用的开发用户，单个小头seed；探索性结果。', '',
        '| 域 | GRAM N10 | SAS+文本参照 | 全图+文本 | 提示内+文本 | 匹配图+文本 | 仅文本 | 仅CF |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        p = OUT/d
        assert read(p/'status.json')['state'] == 'COMPLETED'
        assert read(p/'lease_status.json')['state'] == 'released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
        for manifest in (OUT/'manifest.json', p/'training_manifest.json'):
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
                scores = {int(i): float(v) for i, v in zip(pred['ids'], pred['scores']) if i not in sample['history']}
                y = sample['target']
                alternative = 2**30 if y not in scores else 1+sum(v > scores[y] or (v == scores[y] and i < y) for i, v in scores.items())
                assert actual == alternative
                rr[key].append(actual)
        rr = {k: np.asarray(v) for k, v in rr.items()}
        old = np.load(OLD/d/'ranks.npz')
        for a, b in (('native', 'native'), ('sas_text_late', 'user_late'), ('sas_text_guided', 'user_guided')):
            np.testing.assert_array_equal(rr[a], old[b])
        np.savez_compressed(p/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **rr)
        pure, signals = np.load(p/'pure_ranks.npz'), np.load(p/'signals.npz')
        for arm in ARMS+('sas_text',):
            expected = []
            for j, s in enumerate(samples):
                values = signals[arm][j]
                ids = np.arange(1, len(values)+1)
                keep = ~np.isin(ids, s['history'])
                order = ids[keep][np.lexsort((ids[keep], -values[keep]))]
                pos = np.flatnonzero(order == s['target'])
                expected.append(int(pos[0])+1 if len(pos) else len(values)+1)
            np.testing.assert_array_equal(expected, pure[arm])
        pairs = [('cf_full_guided', a+'_guided') for a in ('cf_prompt', 'cf_matched', 'text_only', 'cf_only')]
        pairs += [('cf_matched_guided', 'cf_prompt_guided')]
        pairs += [(a+'_guided', b) for a in ARMS for b in ('native', a+'_late', 'sas_text_guided')]
        short = np.asarray([len(s['full_history']) <= 20 for s in samples])
        diagnostics = read(p/'support_diagnostics.json')
        cf_r = np.asarray([s['pure_cf_ranks']['cf_full'] for s in diagnostics])
        opportunities = (cf_r <= 10) & (rr['native'] > 10)
        results[d] = dict(users=512, metrics={k: S.metrics(v) for k, v in rr.items()},
            comparisons={a+'_vs_'+b: S.contrast(rr[a], rr[b], repeats=2000, seed=20261015) for a, b in pairs},
            history_at_most20=dict(n=int(short.sum()), metrics={k: S.metrics(v[short]) for k, v in rr.items()}),
            relation_opportunities=dict(cf_hit10_native_miss10=int(opportunities.sum()),
                among_these_target_not_listed_any_hint=sum(bool(opportunities[j]) and not r['target_listed_any_history_hint'] for j, r in enumerate(diagnostics)),
                target_with_prompt_support=sum(r['target_raw_support']['cf_prompt'] > 0 for r in diagnostics),
                target_with_outside_support=sum(r['target_raw_support']['cf_outside'] > 0 for r in diagnostics)),
            training=read(p/'training.json'), graph=read(p/'graph_summary.json'),
            score_checks=read(p/'score_checks.json'), test_read=False)
        write(p/'summary.json', results[d])
        write(p/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', users=512,
            generation_ranks=len(keys)*512, pure_ranks=(len(ARMS)+1)*512,
            old_rank_arrays_matched=3, independent_count_greater_rank=True, test_read=False))
        paths = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json',
                              'runtime_checks.json', 'smoke.json', 'training_manifest.json')]
        write(p/'output_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))
        names = ('native', 'sas_text_guided')+tuple(a+'_guided' for a in ARMS)
        lines.append('| '+d+' | '+' | '.join(f'{results[d]["metrics"][k]["N10"]:.6f}' for k in names)+' |')
    write(OUT/'summary.json', dict(domains=results, test_read=False, seed=0))
    lines += ['', '前三个CF+文本版本均按相同预算独立训练一个用户级融合器；两个单专家组无融合器。SAS+文本参照复用已验证保存预测，并在本轮3人新生成核对；原GRAM每人重新生成。', '',
        '主比较为全图−提示内；全图−仅文本检验CF条件贡献，全图−仅CF检验文本条件贡献。匹配图−提示内控制逐源边数与权重总和，但不能同时控制逐用户归一化后的尺度和图拓扑。', '',
        '提示外边不等于候选完全未被提示，也不等于模型参数不知道该关系。共现不能称为功能兼容或因果关系。完整逐用户支持与提示覆盖见support_diagnostics.json。', '',
        '所有预定组和正负变化完整保留，未按纯评分筛掉真实生成。2000次配对bootstrap为未经多重比较校正的描述性区间，不包含上游训练及研究选择的不确定性；不是论文最终确认结果。', '',
        '同一融合器学习外部影响强度尚未启动，等待专家依据判断。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    global LEASE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'train', 'generate', 'summarize'))
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args()
    bind()
    def interrupted(signum, frame):
        raise InterruptedError('Collaborative relation process interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(dict(relations=R.tests(), fusion=C.tests())))
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
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
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
