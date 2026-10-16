"""BGE vs Qwen3 0.6B/4B: matched item text and user fuser, frozen GRAM.

Scoring/history aggregation derives from the existing LIME-Rec-based pipeline;
token guidance is the existing AGRec-inspired probability guide. No new mechanism
or model novelty is claimed by this encoder substitution experiment.
"""
import argparse
import gc
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
import torch

from experiment.phase21 import single_fusion as S
from experiment.phase21 import single_fusion_core as C
from experiment.phase21 import candidate_text_source as T
from experiment.phase21 import candidate_text_source_core as TC
from experiment.phase21.candidate_gate_core import fold

ROOT, OLD, SOURCE = S.ROOT, S.OUT, S.SOURCE
OUT = ROOT/'artifacts/phase21/qwen_text_expert_20261015_v1'
PLAN = ROOT/'plan/第二十一阶段/21.33_Qwen文本专家替换对照_20261015.md'
REPORT = ROOT/'report/第二十一阶段/Qwen文本专家替换对照_20261015.md'
MODULE = 'experiment.phase21.qwen_text_expert'
ENCODER_PY = ROOT/'.runtime/qwen_embedding_20261015/env/bin/python'
DOMAINS = ('Toys', 'Beauty')
NEW = ('qwen06', 'qwen4')
ARMS = ('full',)+NEW
CFG = dict(S.CFG, arms=list(ARMS), experts=['SASRec', 'text encoder'],
    expected_workload_peak_mib=16384, allocation_limit_gib=15,
    qwen_models={
        'qwen06': dict(repo='Qwen/Qwen3-Embedding-0.6B',
            revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3', dimension=1024, batch_size=16),
        'qwen4': dict(repo='Qwen/Qwen3-Embedding-4B',
            revision='5cf2132abc99cad020ac570b19d031efec650f2b', dimension=2560, batch_size=8)},
    qwen_max_length=2048, qwen_truncation=False,
    text_policy='Exact BGE-visible document prefix; original BGE input IDs checked for every item',
    qwen_instruction=None, history_policy='Unchanged exp(-0.1*position_distance) mean and cosine',
    question='Text encoder substitution and Qwen capacity under the current user fuser',
    primary_comparisons=['qwen06_guided vs full_guided', 'qwen4_guided vs full_guided',
                         'qwen4_guided vs qwen06_guided'],
    full_baseline='Existing stage21.24 user head; all512 score and generation replay',
    alpha_learned=False, model_size_equal=False, all_arms_generate=True,
    bge_revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
    stop_after='Two domains, 512 users each, seed0; no automatic 8B or additional methods')
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
    T.OUT, T.OLD, T.CFG, T.ARMS = OUT, OLD, CFG, ARMS


def prepare():
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
    from lime_rec.data import _read_metadata
    assert read(OLD/'status.json')['state'] == 'COMPLETED'
    assert read(OLD/'review/verification.json')['state'] == 'VERIFIED_SAVED_PREDICTIONS'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG)
    write(OUT/'tests.json', dict(fuser=C.tests(), semantic_scoring=TC.tests()))
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
    assets = read(T.ASSETS/'manifest.json')
    assert assets['revision'] == CFG['bge_revision']
    for name, digest in assets['hashes'].items():
        use(T.BGE/name, digest)
    use(T.ASSETS/'manifest.json')
    text_audit = ROOT/'artifacts/phase21/candidate_text_source_20261015_v1'
    prior_text_hashes = read(text_audit/'manifest.json')['sha256']
    tokenizer = AutoTokenizer.from_pretrained(str(T.BGE), local_files_only=True, use_fast=True)
    assert tokenizer.is_fast
    for d in DOMAINS:
        p = OUT/d
        p.mkdir()
        for name in ('samples.json', 'catalog.json', 'data_config.json'):
            use(OLD/d/name)
            shutil.copy2(OLD/d/name, p/name)
            use(p/name)
        samples = read(p/'samples.json')
        training = [s for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        ck = torch.load(OLD/d/'user.pt', map_location='cpu')
        assert ck['fit_users'] == [s['user_id'] for s in training]
        assert set(ck['fit_users']).isdisjoint(s['user_id'] for s in samples)
        assert len(samples) == 512 and len(training) == dict(Toys=2773, Beauty=3310)[d]
        for key in ('seed', 'epochs', 'checkpoints', 'batch_size', 'lr', 'weight_decay', 'score_scale'):
            assert ck['config'][key] == CFG[key]
        catalog, dc = read(p/'catalog.json'), read(p/'data_config.json')
        embedding_path = ROOT/dc['embedding']
        original = np.load(embedding_path)
        ids = catalog['items'][1:]
        assert original['item_ids'].tolist() == ids
        metadata_path = embedding_path.parent/'author_metadata.jsonl'
        metadata = _read_metadata(metadata_path)
        raw = [metadata[i] for i in ids]
        texts, shortened, decoded_fallback = [], 0, 0
        for text in raw:
            full_ids = tokenizer(text, truncation=True, max_length=512)['input_ids']
            body = tokenizer(text, add_special_tokens=False, truncation=True, max_length=510,
                             return_offsets_mapping=True)
            end = body['offset_mapping'][-1][1] if body['offset_mapping'] else len(text)
            prefix = text[:end] if len(body['input_ids']) == 510 else text
            if tokenizer(prefix, truncation=True, max_length=512)['input_ids'] != full_ids:
                prefix = tokenizer.decode(body['input_ids'], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
                decoded_fallback += 1
            assert tokenizer(prefix, truncation=True, max_length=512)['input_ids'] == full_ids
            assert prefix.strip()
            shortened += prefix != text
            texts.append(prefix)
        write(p/'encoder_texts.json', dict(item_ids=ids, texts=texts))
        write(p/'text_matching.json', dict(state='PASSED', items=len(ids),
            bge_input_id_matches=len(ids), shortened_items=shortened,
            decoded_prefix_fallbacks=decoded_fallback, fields='same author_metadata rendering',
            targets_used=False, qwen_instructions=None))
        refs = [json.loads(line) for line in (OLD/d/'predictions.jsonl').read_text().splitlines()]
        for sample, ref in zip(samples, refs):
            for k in ('user_id', 'target', 'history'):
                assert sample[k] == ref[k]
        write(p/'references.json', refs)
        for name in ('encoder_texts.json', 'text_matching.json', 'references.json'):
            use(p/name)
        use(embedding_path, prior_text_hashes[str(embedding_path.relative_to(ROOT))])
        use(metadata_path, prior_text_hashes[str(metadata_path.relative_to(ROOT))])
        assert read(text_audit/d/'encoding_checks.json')['state'] == 'PASSED'
        use(text_audit/d/'encoding_checks.json')
        use(ROOT/dc['parent_checkpoint'], dc['parent_sha256'])
        for name in ('samples.json', 'frequencies.npy'):
            use(SOURCE/d/name)
        for name in ('user.pt', 'signals.npz', 'ranks.npz', 'predictions.jsonl',
                     'train/item.npy', 'train/users.json', 'validation/item.npy', 'validation/user.npy'):
            use(OLD/d/name)
        write(p/'status.json', dict(state='PREPARED', users_total=512, items=len(ids), updated_at=stamp()))
    for path in (PLAN, Path(__file__), ROOT/'experiment/phase21/qwen_text_encoder.py',
                 ROOT/'experiment/phase21/run_qwen_text_expert.sh', OUT/'config.json',
                 ROOT/'experiment/phase21/candidate_text_source.py',
                 ROOT/'experiment/phase21/candidate_text_source_core.py'):
        use(path)
    for relative in manifest:
        path = ROOT/relative
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# Qwen 文本专家替换对照\n\n状态：PREPARED，尚无 Qwen 效果结论。\n')
    print(json.dumps(dict(state='PREPARED', input_hashes=len(manifest))), flush=True)


class Data(S.Data):
    def __init__(self, domain, training, variant, full_emb, qwen_emb):
        self.domain = domain
        self.samples = ([s for s in read(SOURCE/domain/'samples.json') if s['role'] == 'ranker_fit']
                        if training else read(OUT/domain/'samples.json'))
        self.n = len(self.samples)
        part = 'train' if training else 'validation'
        old = np.load(OLD/domain/part/'item.npy', mmap_mode='r')
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
        errors = []
        for start in range(0, self.n, 64):
            stop = min(start+64, self.n)
            histories = [s['full_history'] for s in self.samples[start:stop]]
            original = TC.semantic_scores(histories, full_emb, full_emb)
            errors.append(float(np.abs(original-old[start:stop, :, 1]).max()))
            np.testing.assert_allclose(original, old[start:stop, :, 1], atol=5e-6, rtol=2e-5)
            text = TC.semantic_scores(histories, qwen_emb, qwen_emb)
            for k, j in enumerate(range(start, stop)):
                s = self.samples[j]
                assert s['history'] == s['full_history'][-20:]
                u, c = C.features(np.stack((old[j, :, 0], text[k])), s['full_history'], counts)
                np.testing.assert_array_equal(c[:, 0], old[j, :, 0])
                self.user[j], self.item[j] = u, c
                self.m1[j], self.m2[j] = c.mean(0, dtype=np.float64), np.square(c.astype(np.float64)).mean(0)
            progress(domain, 'features_'+part, completed=stop, total=self.n)
        self.user.flush()
        self.item.flush()
        self.targets = np.asarray([s['target']-1 for s in self.samples], dtype=np.int64)
        self.folds = np.asarray([fold(s['user_id'], domain) for s in self.samples])
        write(self.folder/'metadata.json', dict(users=self.n, variant=variant, training=training,
            original_bge_score_max_abs_error=max(errors), sas_scores_identical=True,
            target_used_in_features=False, users_order=[s['user_id'] for s in self.samples]))


def train(domain, lease_pid):
    global VARIANT
    subprocess.run([str(ENCODER_PY), '-u', '-m', 'experiment.phase21.qwen_text_encoder',
                    'encode', '--domain', domain, '--lease-pid', str(lease_pid)], cwd=ROOT, check=True)
    p = OUT/domain
    dc = read(p/'data_config.json')
    with np.load(ROOT/dc['embedding']) as e:
        full, item_ids = e['embeddings'], e['item_ids']
    samples = read(p/'samples.json')
    ck = torch.load(OLD/domain/'user.pt', map_location='cpu')
    signals = dict(full=np.load(OLD/domain/'signals.npz')['user'])
    old_u = np.load(OLD/domain/'validation/user.npy', mmap_mode='r')
    old_c = np.load(OLD/domain/'validation/item.npy', mmap_mode='r')
    for j in range(512):
        f, _ = C.numpy_score(ck['state_dict'], old_u[j], old_c[j], ck['mean'], ck['std'], 'user')
        np.testing.assert_allclose(f, signals['full'][j], atol=3e-6, rtol=2e-5)
    fits, checks = dict(full=dict(reused=True, selected_epochs=ck['selected_epochs'])), {}
    for variant in NEW:
        VARIANT = variant
        with np.load(p/(variant+'_embeddings.npz')) as e:
            np.testing.assert_array_equal(e['item_ids'], item_ids)
            qwen = e['embeddings']
        assert sha(p/(variant+'_embeddings.npz')) == read(p/(variant+'_encoding.json'))['embeddings_sha256']
        training, validation = Data(domain, True, variant, full, qwen), Data(domain, False, variant, full, qwen)
        assert [s['user_id'] for s in training.samples] == ck['fit_users']
        assert set(ck['fit_users']).isdisjoint(s['user_id'] for s in validation.samples)
        S.OUT = OUT/'models'/variant
        (S.OUT/domain).mkdir(parents=True)
        model, norm, fits[variant] = S.fit(training, 'user', torch.device('cuda:0'))
        signals[variant], checks[variant] = S.export(validation, model, norm, 'user', torch.device('cuda:0'))
        saved = torch.load(S.OUT/domain/'user.pt', map_location='cpu')
        assert saved['config'] == CFG and saved['fit_users'] == ck['fit_users']
        assert saved['parameters'] == 610
        for j in (0, 127, 255, 383, 511):
            f, w = C.numpy_score(saved['state_dict'], validation.user[j], validation.item[j],
                                 saved['mean'], saved['std'], 'user')
            np.testing.assert_allclose(f, signals[variant][j], atol=3e-6, rtol=2e-5)
            np.testing.assert_allclose(w[:, 0], np.full(len(w), w[0, 0]), atol=2e-7, rtol=1e-6)
        checks[variant]['text_alone_ranks'] = [C.target_rank(validation.item[j, :, 1], s['target'], s['history'])
                                              for j, s in enumerate(samples)]
        del model, training, validation, qwen
        gc.collect()
        torch.cuda.empty_cache()
    S.OUT, VARIANT = OUT, None
    pure = {a: np.asarray([C.target_rank(signals[a][j], s['target'], s['history'])
                          for j, s in enumerate(samples)]) for a in ARMS}
    np.savez(p/'signals.npz', **signals)
    np.savez_compressed(p/'pure_ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **pure)
    write(p/'training.json', dict(fits=fits, checks=checks, full_score_replayed_users=512,
        pure_metrics={a: S.metrics(r) for a, r in pure.items()}, test_read=False))
    paths = [p/n for n in ('signals.npz', 'pure_ranks.npz', 'training.json')]
    paths += [p/(a+s) for a in NEW for s in ('_embeddings.npz', '_encoding.json')]
    paths += list((OUT/'models').glob('*/'+domain+'/*.pt'))
    write(p/'training_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))


def summarize():
    keys = ('native',)+tuple(a+s for a in ARMS for s in ('_late', '_guided'))
    results = {}
    lines = ['# Qwen 文本专家替换对照', '',
        '状态：COMPLETED。两域各512名反复使用的开发用户、固定小头seed0，探索性对照。', '',
        '| 域 | GRAM N10 | SAS+BGE | SAS+Qwen0.6B | SAS+Qwen4B |', '|---|---:|---:|---:|---:|']
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
            for key in ('user_id', 'target', 'history'):
                assert row[key] == sample[key]
            for key in keys:
                record = row[key]
                r = S.OLD.record_rank(record, sample)
                ids, values = np.asarray(record['ids']), np.asarray(record['scores'])
                keep = ~np.isin(ids, sample['history'])
                ordered = ids[keep][np.lexsort((ids[keep], -values[keep]))]
                positions = np.flatnonzero(ordered == sample['target'])
                assert r == (int(positions[0])+1 if len(positions) else 2**30)
                ranks[key].append(r)
        ranks = {k: np.asarray(v) for k, v in ranks.items()}
        old = np.load(OLD/d/'ranks.npz')
        for key, origin in [('native', 'native'), ('full_late', 'user_late'), ('full_guided', 'user_guided')]:
            np.testing.assert_array_equal(ranks[key], old[origin])
        np.savez_compressed(p/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **ranks)
        pure, signals = np.load(p/'pure_ranks.npz'), np.load(p/'signals.npz')
        for arm in ARMS:
            recomputed = []
            for j, sample in enumerate(samples):
                ids = np.arange(1, len(signals[arm][j])+1)
                keep = ~np.isin(ids, sample['history'])
                ordered = ids[keep][np.lexsort((ids[keep], -signals[arm][j][keep]))]
                position = np.flatnonzero(ordered == sample['target'])
                recomputed.append(int(position[0])+1 if len(position) else len(ids)+1)
            np.testing.assert_array_equal(pure[arm], recomputed)
        pairs = [(a+'_guided', 'full_guided') for a in NEW]+[('qwen4_guided', 'qwen06_guided')]
        pairs += [(a+'_guided', b) for a in ARMS for b in ('native', a+'_late')]
        result = dict(users=512, metrics={k: S.metrics(v) for k, v in ranks.items()},
            comparisons={a+'_vs_'+b: S.contrast(ranks[a], ranks[b], repeats=2000, seed=20261015) for a, b in pairs},
            training=read(p/'training.json'), encoding={a: read(p/(a+'_encoding.json')) for a in NEW},
            text_matching=read(p/'text_matching.json'), test_read=False)
        results[d] = result
        write(p/'summary.json', result)
        write(p/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', users=512,
            old_rank_arrays_matched=3, alternative_sort_implementation=True,
            generation_ranks=len(keys)*512, pure_ranks=len(ARMS)*512, test_read=False))
        paths = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json',
                              'runtime_checks.json', 'smoke.json', 'training_manifest.json')]
        write(p/'output_manifest.json', dict(sha256={str(q.relative_to(ROOT)): sha(q) for q in paths}))
        lines.append('| '+d+' | '+' | '.join(f'{result["metrics"][k]["N10"]:.6f}' for k in
                     ('native', 'full_guided', 'qwen06_guided', 'qwen4_guided'))+' |')
    write(OUT/'summary.json', dict(domains=results, test_read=False, seed=0))
    lines += ['', 'SASRec、输入商品文本内容、历史聚合、用户融合器训练规则、alpha=.3、beam50均固定。',
              'BGE使用已验证头并逐用户重放；两种Qwen各自重训同结构融合器。Qwen规格、编码精度和向量维度不同，不是等参数架构消融。',
              '所有预定模型完整生成；纯排序、同候选重排、完整生成、收益/误伤与描述性未校正区间见summary.json。',
              '一个seed、反复使用的开发集，不能支持正式显著性或全量收益结论；0.6B失败不能代表4B/8B失败。',
              '文本相似度提升不自动等于下一商品偏好提升；本轮不宣称新专家机制或线上性能。',
              '结束释放显存租约，不自动追加8B、时间专家或学习alpha。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    global LEASE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['test', 'prepare', 'supervise', 'worker', 'train', 'generate', 'summarize'])
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args()
    bind()
    def interrupted(signum, frame):
        raise InterruptedError('Qwen text experiment interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(dict(fuser=C.tests(), semantic=TC.tests())))
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
                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.cuda.set_per_process_memory_fraction(CFG['allocation_limit_gib']*2**30 /
                                                         torch.cuda.get_device_properties(0).total_memory)
                train(args.domain, args.lease_pid)
            else:
                T.Generation(args.domain, LEASE).run()
        else:
            summarize()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'worker':
            folder = OUT/args.domain if args.domain else OUT
            write(folder/'status.json', dict(state='FAILED', error=repr(exc),
                  traceback=traceback.format_exc(), updated_at=stamp()))
        raise


if __name__ == '__main__':
    sys.exit(main())
