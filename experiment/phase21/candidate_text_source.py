"""Candidate text vs lexical IDs, with matched user fusion and real GRAM decoding.

Reuses stage21.24 fitting/supervision and its frozen decoder, not its candidate
gate. Expert scoring is LIME-Rec-derived; probability guidance is AGRec-inspired.
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
from experiment.phase21 import single_fusion_core as C
from experiment.phase21 import candidate_text_source_core as T
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20.lime_clean_gram_core import T5

ROOT, OLD, SOURCE = S.ROOT, S.OUT, S.SOURCE
OUT = ROOT/'artifacts/phase21/candidate_text_source_20261015_v1'
PLAN = ROOT/'plan/第二十一阶段/21.27_候选文本信息来源匹配验证_20261015.md'
REPORT = ROOT/'report/第二十一阶段/候选文本信息来源匹配验证_20261015.md'
MODULE = 'experiment.phase21.candidate_text_source'
ASSETS = ROOT/'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/assets'
BGE = ASSETS/'bge-base-en-v1.5'
ARMS = ('full', 'candidate_id', 'both_id', 'sas_only')
NEW = ('candidate_id', 'both_id')
DOMAINS = ('Toys', 'Beauty')
CFG = dict(S.CFG, arms=list(ARMS), question='candidate text information vs lexical IDs',
    candidate_gate=False, alpha_learned=False, embedding_batch=32, bge_max_length=512,
    bge_revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
    history_policy='same original full_history; prespecified <=20 subgroup',
    full_baseline='reuse stage21.24 user head; all512 score and generation replay',
    sas_only='weight1; no fusion features or trainable parameters',
    primary_comparison='full_guided_vs_candidate_id_guided',
    second_hypothesis='authorized later; not automatically executed by this runner')
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp
LEASE = None
VARIANT = None


def progress(domain, stage, **values):
    if LEASE is not None:
        assert LEASE.poll() is None, 'GPU lease exited'
    write(OUT/domain/'status.json', dict(state='RUNNING', stage=stage, variant=VARIANT,
        pid=os.getpid(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), updated_at=stamp(),
        test_read=False, **values))
    print(json.dumps(dict(domain=domain, stage=stage, variant=VARIANT, **values)), flush=True)


def bind():
    S.OUT, S.REPORT, S.MODULE, S.CFG = OUT, REPORT, MODULE, CFG
    S.progress, S.summarize = progress, summarize


def prepare():
    assert read(OLD/'status.json')['state'] == 'COMPLETED'
    checks = T.tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG)
    write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}

    def use(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        manifest[str(path.relative_to(ROOT))] = digest

    # Pin the existing decoder, training algorithm and tokenizer implementation.
    for relative, digest in read(OLD/'manifest.json')['sha256'].items():
        if relative.startswith(('experiment/', 'GRAM/src/', '.cache/')):
            use(ROOT/relative, digest)
    assets = read(ASSETS/'manifest.json')
    assert assets['revision'] == CFG['bge_revision']
    for name, digest in assets['hashes'].items():
        use(BGE/name, digest)
    use(ASSETS/'manifest.json')
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
        for key in ('seed', 'epochs', 'checkpoints', 'batch_size', 'lr', 'weight_decay', 'score_scale'):
            assert ck['config'][key] == CFG[key]
        assert len(samples) == len({s['user_id'] for s in samples}) == 512
        assert len(train) == dict(Toys=2773, Beauty=3310)[d]
        catalog = read(p/'catalog.json')
        texts = tokenizer.batch_decode(catalog['decoder_paths'], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        assert len(texts) == len(catalog['items'])-1 and all(s.strip() for s in texts)
        write(p/'id_texts.json', dict(item_ids=catalog['items'][1:], texts=texts,
            rendering='T5 batch_decode actual decoder_paths; special tokens removed'))
        use(p/'id_texts.json')
        refs = [json.loads(line) for line in (OLD/d/'predictions.jsonl').read_text().splitlines()]
        for sample, ref in zip(samples, refs):
            assert (sample['user_id'], sample['target'], sample['history']) == (ref['user_id'], ref['target'], ref['history'])
        write(p/'references.json', refs)
        use(p/'references.json')
        dc = read(p/'data_config.json')
        use(ROOT/dc['embedding'])
        use(ROOT/dc['parent_checkpoint'], dc['parent_sha256'])
        use((ROOT/dc['embedding']).parent/'author_metadata.jsonl')
        for name in ('samples.json', 'frequencies.npy'):
            use(SOURCE/d/name)
        for name in ('user.pt', 'signals.npz', 'ranks.npz', 'predictions.jsonl', 'training.json',
                     'train/item.npy', 'train/users.json', 'validation/item.npy', 'validation/user.npy'):
            use(OLD/d/name)
        write(p/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))
    for path in (PLAN, Path(__file__), ROOT/'experiment/phase21/candidate_text_source_core.py',
                 ROOT/'experiment/phase21/run_candidate_text_source.sh', OUT/'config.json',
                 ROOT/'experiment/phase20/lime_comparison_core.py', ROOT/'third_party/LIME-Rec/lime_rec/data.py'):
        use(path)
    for relative in manifest:
        p = ROOT/relative
        if p.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 候选文本信息来源匹配验证\n\n状态：PREPARED，尚无新效果结论。四组预定真实生成对照，第二条协同关系尚未启动。\n')
    print(json.dumps(dict(state='PREPARED', tests=checks, hashes=len(manifest))), flush=True)


def encode_ids(domain):
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
    from lime_rec.data import _read_metadata
    p = OUT/domain
    info = read(p/'id_texts.json')
    dc = read(p/'data_config.json')
    full = np.load(ROOT/dc['embedding'])
    assert full['item_ids'].tolist() == info['item_ids']
    metadata = _read_metadata((ROOT/dc['embedding']).parent/'author_metadata.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(str(BGE), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(BGE), local_files_only=True).eval().requires_grad_(False).cuda()

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            value = torch.nn.functional.normalize(encoder(**batch).last_hidden_state[:, 0], dim=-1)
        return value.cpu().numpy()

    js = [0, len(info['texts'])//4, len(info['texts'])//2, 3*len(info['texts'])//4, len(info['texts'])-1]
    recreated = encode([metadata[info['item_ids'][j]] for j in js])
    np.testing.assert_allclose(recreated, full['embeddings'][js], atol=5e-5, rtol=3e-4)
    values = np.empty_like(full['embeddings'])
    begin = time.monotonic()
    for start in range(0, len(values), CFG['embedding_batch']):
        stop = min(start+CFG['embedding_batch'], len(values))
        values[start:stop] = encode(info['texts'][start:stop])
        if start == 0 or start % 512 == 0 or stop == len(values):
            progress(domain, 'encode_lexical_ids', completed=stop, total=len(values), elapsed_seconds=time.monotonic()-begin)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1., atol=2e-6)
    np.savez(p/'id_embeddings.npz', item_ids=np.asarray(info['item_ids']), embeddings=values)
    write(p/'encoding_checks.json', dict(state='PASSED', reencoded_full_text_items=js,
        full_cache_max_abs_error=float(np.max(np.abs(recreated-full['embeddings'][js]))),
        encoder=str(BGE.relative_to(ROOT)), revision=CFG['bge_revision'], candidate_labels_accessed=False,
        id_text_sha256=sha(p/'id_texts.json'), id_embedding_sha256=sha(p/'id_embeddings.npz')))
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


class Data(S.Data):
    def __init__(self, domain, training, variant, full_emb, id_emb):
        self.domain = domain
        self.samples = ([s for s in read(SOURCE/domain/'samples.json') if s['role'] == 'ranker_fit']
                        if training else read(OUT/domain/'samples.json'))
        self.n = len(self.samples)
        part = 'train' if training else 'validation'
        if training:
            assert [s['user_id'] for s in self.samples] == [s['user_id'] for s in read(OLD/domain/'train/users.json')]
        old = np.load(OLD/domain/part/'item.npy', mmap_mode='r')
        self.v = old.shape[1]
        assert old.shape == (self.n, self.v, C.DC)
        self.folder = OUT/domain/'features'/variant/part
        self.folder.mkdir(parents=True)
        self.user = np.lib.format.open_memmap(self.folder/'user.npy', mode='w+', dtype='float32', shape=(self.n, C.DU))
        self.item = np.lib.format.open_memmap(self.folder/'item.npy', mode='w+', dtype='float32', shape=old.shape)
        self.m1, self.m2 = np.zeros((self.n, C.DC)), np.zeros((self.n, C.DC))
        counts = np.load(SOURCE/domain/'frequencies.npy')
        errors = []
        for start in range(0, self.n, 64):
            stop = min(start+64, self.n)
            histories = [s['full_history'] for s in self.samples[start:stop]]
            original = T.semantic_scores(histories, full_emb, full_emb)
            error = float(np.abs(original-old[start:stop, :, 1]).max())
            errors.append(error)
            np.testing.assert_allclose(original, old[start:stop, :, 1], atol=5e-6, rtol=2e-5)
            text = T.semantic_scores(histories, full_emb if variant == 'candidate_id' else id_emb, id_emb)
            for k, j in enumerate(range(start, stop)):
                sample = self.samples[j]
                assert sample['history'] == sample['full_history'][-20:]
                u, c = C.features(np.stack((old[j, :, 0], text[k])), sample['full_history'], counts)
                np.testing.assert_array_equal(c[:, 0], old[j, :, 0])
                self.user[j], self.item[j] = u, c
                self.m1[j] = c.mean(0, dtype=np.float64)
                self.m2[j] = np.square(c.astype(np.float64)).mean(0)
            progress(domain, 'features_'+part, completed=stop, total=self.n)
        self.user.flush()
        self.item.flush()
        self.targets = np.asarray([s['target']-1 for s in self.samples], dtype=np.int64)
        self.folds = np.asarray([fold(s['user_id'], domain) for s in self.samples])
        write(self.folder/'metadata.json', dict(users=self.n, variant=variant, training=training,
            original_text_max_abs_error=max(errors), sas_scores_identical=True, target_used_in_features=False,
            users_order=[s['user_id'] for s in self.samples]))


def train(domain):
    global VARIANT
    encode_ids(domain)
    p = OUT/domain
    dc = read(p/'data_config.json')
    full = np.load(ROOT/dc['embedding'])['embeddings']
    ids = np.load(p/'id_embeddings.npz')['embeddings']
    samples = read(p/'samples.json')
    old_signals = np.load(OLD/domain/'signals.npz')
    features = np.load(OLD/domain/'validation/item.npy', mmap_mode='r')
    signals = dict(full=old_signals['user'], sas_only=np.array(features[:, :, 0]))
    ck = torch.load(OLD/domain/'user.pt', map_location='cpu')
    assert set(ck['fit_users']).isdisjoint(s['user_id'] for s in samples)
    # Replay all512 original user-head outputs, not only three smoke records.
    old_user = np.load(OLD/domain/'validation/user.npy', mmap_mode='r')
    for j in range(len(samples)):
        f, _ = C.numpy_score(ck['state_dict'], old_user[j], features[j], ck['mean'], ck['std'], 'user')
        np.testing.assert_allclose(f, signals['full'][j], atol=3e-6, rtol=2e-5)
    fits = dict(full=dict(reused=True, selected_epochs=ck['selected_epochs'], fit_users=len(ck['fit_users'])),
                sas_only=dict(parameters=0, weight=1.))
    checks = {}
    for variant in NEW:
        VARIANT = variant
        training = Data(domain, True, variant, full, ids)
        validation = Data(domain, False, variant, full, ids)
        assert [s['user_id'] for s in training.samples] == ck['fit_users']
        S.OUT = OUT/'models'/variant
        (S.OUT/domain).mkdir(parents=True)
        model, norm, fits[variant] = S.fit(training, 'user', torch.device('cuda:0'))
        signals[variant], checks[variant] = S.export(validation, model, norm, 'user', torch.device('cuda:0'))
        saved = torch.load(S.OUT/domain/'user.pt', map_location='cpu')
        for j in (0, 127, 255, 383, 511):
            f, w = C.numpy_score(saved['state_dict'], validation.user[j], validation.item[j], saved['mean'], saved['std'], 'user')
            np.testing.assert_allclose(f, signals[variant][j], atol=3e-6, rtol=2e-5)
            np.testing.assert_allclose(w[:, 0], np.full(len(w), w[0, 0]), atol=2e-7, rtol=1e-6)
        # Also save text-alone ranks as source diagnostics, not new guide arms.
        checks[variant]['text_alone_ranks'] = [C.target_rank(validation.item[j, :, 1], s['target'], s['history'])
                                             for j, s in enumerate(samples)]
        del model, training, validation
        gc.collect()
        torch.cuda.empty_cache()
    S.OUT = OUT
    VARIANT = None
    pure = {a: np.asarray([C.target_rank(signals[a][j], s['target'], s['history']) for j, s in enumerate(samples)]) for a in ARMS}
    np.savez(p/'signals.npz', **signals)
    np.savez_compressed(p/'pure_ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **pure)
    write(p/'training.json', dict(fits=fits, checks=checks, full_score_replayed_users=512,
        pure_metrics={a: S.metrics(r) for a, r in pure.items()}, test_read=False))
    paths = [p/name for name in ('signals.npz', 'pure_ranks.npz', 'training.json', 'id_embeddings.npz', 'encoding_checks.json')]
    paths += list((OUT/'models').glob('*/'+domain+'/*.pt'))
    write(p/'training_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))


class Generation(S.Generation):
    def run(self):
        self.stage = 'native_zero_full_smoke'
        self.status()
        checks = []
        for j in (0, 256, 511):
            ctx = self.context(j)
            native = self.decode(ctx)
            plain = self.decode(ctx, native_entry=True, record=False)
            q = self.index.conditional_logs(self.signals['full'][j], temperature_scale=CFG['score_scale'])
            zero = self.decode(ctx, q, 0.)
            full = self.decode(ctx, q, CFG['alpha'])
            for other in (plain, zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'], other['scores'])
            ref = self.reference[self.samples[j]['user_id']]
            assert full['ids'] == ref['user_guided']['ids']
            np.testing.assert_allclose(full['scores'], ref['user_guided']['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user_id=self.samples[j]['user_id'], alpha0_exact=True, full_reproduced=True))
            del ctx, native, plain, zero, full
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
                           native=dict(ids=native['ids'], scores=native['scores'].tolist()))
                for arm in ARMS:
                    q = self.index.conditional_logs(self.signals[arm][j], temperature_scale=CFG['score_scale'])
                    late = native['guide'].rescore(native['paths'], q, CFG['alpha'])
                    guided = self.decode(ctx, q, CFG['alpha'])
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                    if arm == 'full':
                        for a, b in ((row[arm+'_late'], ref['user_late']), (row[arm+'_guided'], ref['user_guided'])):
                            assert a['ids'] == b['ids']
                            np.testing.assert_allclose(a['scores'], b['scores'], atol=2e-6, rtol=1e-5)
                    del guided
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                self.users_completed = j+1
                self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1) % 8 == 0 or j == 511:
                    self.status()
                del ctx, native
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', gram_model_digest_unchanged=True,
            native_reproduced_users=512, full_reproduced_users=512, gram_optimizer_steps=0))
        self.stage = 'complete'
        self.status('COMPLETED')


def summarize():
    results = {}
    keys = ('native',)+tuple(a+s for a in ARMS for s in ('_late', '_guided'))
    lines = ['# 候选文本信息来源匹配验证', '', '## Material Passport', '',
             '状态：COMPLETED。保存预测重算通过；每域512反复使用的开发用户、单个小头seed，探索性证据。', '',
             '| 域 | GRAM N10 | 完整文本 | 仅候选标识 | 双侧标识 | 无文本 |', '|---|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        p = OUT/d
        assert read(p/'status.json')['state'] == 'COMPLETED'
        assert read(p/'lease_status.json')['state'] == 'released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
        samples = read(p/'samples.json')
        rows = [json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        rr = {k: [] for k in keys}
        for row, sample in zip(rows, samples):
            for k in ('user_id', 'target', 'history'):
                assert row[k] == sample[k]
            for k in keys:
                v = row[k]
                r = S.OLD.record_rank(v, sample)
                ids, scores = np.asarray(v['ids']), np.asarray(v['scores'])
                keep = ~np.isin(ids, sample['history'])
                order = ids[keep][np.lexsort((ids[keep], -scores[keep]))]
                pos = np.flatnonzero(order == sample['target'])
                assert r == (int(pos[0])+1 if len(pos) else 2**30)
                rr[k].append(r)
        rr = {k: np.asarray(v) for k, v in rr.items()}
        old = np.load(OLD/d/'ranks.npz')
        for a, b in (('native', 'native'), ('full_late', 'user_late'), ('full_guided', 'user_guided')):
            np.testing.assert_array_equal(rr[a], old[b])
        np.savez_compressed(p/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **rr)
        signals, pure = np.load(p/'signals.npz'), np.load(p/'pure_ranks.npz')
        for a in ARMS:
            check = []
            for j, s in enumerate(samples):
                values = signals[a][j]
                ids = np.arange(1, len(values)+1)
                keep = ~np.isin(ids, s['history'])
                order = ids[keep][np.lexsort((ids[keep], -values[keep]))]
                pos = np.flatnonzero(order == s['target'])
                check.append(int(pos[0])+1 if len(pos) else len(values)+1)
            np.testing.assert_array_equal(check, pure[a])
        pairs = [('full_guided', a+'_guided') for a in ('candidate_id', 'both_id', 'sas_only')]
        pairs += [('candidate_id_guided', 'both_id_guided')]
        pairs += [(a+'_guided', b) for a in ARMS for b in ('native', a+'_late')]
        short = np.asarray([len(s['full_history']) <= 20 for s in samples])
        results[d] = dict(users=512, metrics={k: S.metrics(v) for k, v in rr.items()},
            comparisons={a+'_vs_'+b: S.contrast(rr[a], rr[b], repeats=2000, seed=20261015) for a, b in pairs},
            history_at_most20=dict(n=int(short.sum()), metrics={k: S.metrics(v[short]) for k, v in rr.items()}),
            training=read(p/'training.json'), encoding=read(p/'encoding_checks.json'), test_read=False)
        write(p/'summary.json', results[d])
        write(p/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', users=512,
            generation_ranks=len(keys)*512, pure_ranks=len(ARMS)*512, old_rank_arrays_matched=3,
            alternative_sort_implementation=True, gram_unchanged=True, test_read=False))
        paths = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json',
                              'runtime_checks.json', 'smoke.json', 'training_manifest.json')]
        write(p/'output_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))
        names = ('native',)+tuple(a+'_guided' for a in ARMS)
        lines.append('| '+d+' | '+' | '.join(f'{results[d]["metrics"][k]["N10"]:.6f}' for k in names)+' |')
    write(OUT/'summary.json', dict(domains=results, test_read=False, seed=0))
    write(OUT/'interpretation_checks.json', dict(covered=11, total=11,
        items=['domains and <=20 subgroup separate', 'user is analysis unit', 'cohort selection disclosed',
               'conditional opportunity not attainable oracle', 'hits and sample sizes retained',
               'reused development and model-selection uncertainty', 'all predefined arms retained',
               'descriptive unadjusted intervals', 'no result-driven branch or retry',
               'representation effect not field-specific causal proof', 'train-only fitting; no test or target features']))
    lines += ['', '完整文本是21.24的用户级参照，本轮逐用户重放；两个标识臂按相同训练预算独立拟合，无文本臂SASRec权重恒为1。', '',
              '主对比为完整文本−仅候选标识；双侧标识检查两侧表示不匹配。完整文本与标识长度和分布不同，不能将差值孤立归因于某个属性。', '',
              '所有H10/H50、救回／误伤、配对描述性区间、历史≤20子群、纯评分和后重排见summary.json。区间不含上游训练和研究选择不确定性。', '',
              '若完整文本未超过双侧标识或无文本组，不宣称完整候选文本必要；相近结果不构成等价证明。所有预定组完整保留，未因纯排序效果筛掉生成。', '',
              '第二条协同关系仍为已授权后续阶段，待本轮分析后实施；本后台进程结束并释放租约，不自动扩大实验。']
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
        raise InterruptedError('Candidate text source process interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(T.tests()))
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
                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
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
