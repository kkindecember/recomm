"""Matched frozen-backbone description-memory screening, Beauty only."""
import argparse
import gc
import hashlib
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

from experiment.phase21.attribute_memory_core import ARMS, AttributeMemoryResidual, memory_token_banks
from experiment.phase21.attribute_retention import read, write, rows, sha, stamp
from experiment.phase21.attribute_retention_core import install_bank
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/attribute_memory_beauty_20260923_v1'
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
PREVIOUS = ROOT / 'artifacts/phase21/attribute_retention_beauty_recovery_20260923_v1/Beauty'
TRAIN = ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1/Beauty'
PLAN = ROOT / 'plan/第二十一阶段/21.2B_原输入保留的描述记忆实验_20260923.md'
REPORT = ROOT / 'report/第二十一阶段/原输入保留的描述记忆_20260923.md'
GPU = 'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2'
MODULE = 'experiment.phase21.attribute_memory'


def prepare():
    from transformers import AutoTokenizer
    from experiment.phase20.lime_clean_gram_core import T5
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    manifest = {}
    def use(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        manifest[str(path.relative_to(ROOT))] = digest
        return path
    for rel, digest in read(PREVIOUS/'input_manifest.json')['sha256'].items():
        use(ROOT/rel, digest)
    for name in ('predictions_parent.jsonl', 'tokens_original.npz', 'samples.json', 'data_config.json'):
        use(PREVIOUS/name)
    for name in ('catalog.json', 'samples.json'):
        use(SOURCE/'Beauty'/name)
    use(SOURCE/'summary.json')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'config.json'):
        use(T5/name)
    receipt = read(ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1/summary.json')
    path = TRAIN/'upstream_train_without_terminal.jsonl'
    use(path, receipt['output_sha256'][str(path.relative_to(ROOT))])
    for p in [PLAN, Path(__file__), ROOT/'experiment/phase21/attribute_memory_core.py',
              ROOT/'experiment/phase21/test_attribute_memory.py',
              ROOT/'experiment/phase21/run_attribute_memory.sh',
              ROOT/'experiment/phase21/attribute_retention.py',
              ROOT/'experiment/phase21/attribute_retention_core.py',
              ROOT/'experiment/phase20/lime_clean_gram_core.py',
              ROOT/'experiment/phase20/lime_clean_upstream_core.py',
              ROOT/'experiment/phase20/lime_parallel_core.py',
              ROOT/'experiment/phase20/catalog_scope_core.py',
              ROOT/'experiment/phase18/core/diff_data.py'] + list((ROOT/'GRAM/src').rglob('*.py')):
        use(p)
        dest = OUT/'sources'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    catalog = read(SOURCE/'Beauty/catalog.json')
    expected_users = [s['user_id'] for s in read(SOURCE/'Beauty/samples.json') if s['role'] == 'development']
    actual_users = [s['user_id'] for s in read(PREVIOUS/'samples.json') if s['cohort'] == 'existing512']
    assert actual_users == expected_users and len(actual_users) == 512
    native = dict(np.load(PREVIOUS/'tokens_original.npz', allow_pickle=False))
    tokenizer = AutoTokenizer.from_pretrained(str(T5), local_files_only=True)
    banks, lengths, usable, donors, audit = memory_token_banks(catalog, tokenizer, native['tokens'])
    assert audit['usable_items'] > 1000 and audit['changed_items'] > 1000
    np.savez_compressed(OUT/'memory_tokens.npz', **banks, lengths=lengths, usable=usable, donors=donors)
    write(OUT/'packing_audit.json', audit)
    for name in ('samples.json', 'data_config.json', 'predictions_parent.jsonl'):
        shutil.copy2(PREVIOUS/name, OUT/name)
        use(OUT/name)
    use(OUT/'memory_tokens.npz')
    shutil.copy2(PLAN, OUT/'protocol.md')
    write(OUT/'manifest.json', dict(created_at=stamp(), sha256=manifest))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), audit=audit, test_read=False))
    REPORT.write_text('# 原输入保留的描述记忆：Beauty实验进展\n\n状态：PREPARED；'+stamp()+
        '。已冻结三组匹配配置，尚无新效果结果。\n\n'+
        '实时进度见 `artifacts/phase21/attribute_memory_beauty_20260923_v1/status.json`；'+
        '先完成零支路生成一致性与资源试跑，通过后才正式训练。完成后自动更新本报告。\n')
    print(json.dumps(dict(state='PREPARED', audit=audit)), flush=True)


class Worker:
    def __init__(self):
        import torch
        from experiment.phase20.lime_clean_gram_core import T5, epoch_order, update_windows
        from experiment.phase20.lime_clean_upstream_core import new_gram, runtime_args
        from experiment.phase18.core.diff_data import DiffDataset
        sys.path.insert(0, str(ROOT/'GRAM/src'))
        from processor import CollatorGRAM
        from utils import generation_trie
        self.started = time.monotonic()
        self.phase('verify_inputs')
        for rel, digest in read(OUT/'manifest.json')['sha256'].items():
            assert sha(ROOT/rel) == digest, rel
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(12288/(torch.cuda.get_device_properties(0).total_memory/2**20))
        self.device = torch.device('cuda')
        self.cfg = read(OUT/'data_config.json')
        self.catalog = read(SOURCE/'Beauty/catalog.json')
        self.samples = read(OUT/'samples.json')
        self.training = rows(TRAIN/'upstream_train_without_terminal.jsonl')
        assert len(self.training) == 127328 and len(self.samples) == 1024
        self.model, self.tokenizer = new_gram(T5)
        self.model.set_checkpoint(False)
        saved = torch.load(ROOT/self.cfg['parent_checkpoint'], map_location='cpu')
        self.model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        self.model.requires_grad_(False).eval().to(self.device)
        self.initial_digest = self.digest()
        assert self.initial_digest == read(SOURCE/'Beauty/cache_summary.json')['model_digest']
        native = dict(np.load(PREVIOUS/'tokens_original.npz', allow_pickle=False))
        self.collator = install_bank(CollatorGRAM(self.tokenizer, runtime_args()), self.catalog, native['tokens'], native['lengths'])
        self.dataset = DiffDataset(self.training, self.catalog)
        self.eval_data = DiffDataset(self.samples, self.catalog)
        self.windows = list(update_windows(epoch_order(len(self.dataset), 2023, 1), 16, 8))
        assert len(self.windows) == 995
        self.tokens = dict(np.load(OUT/'memory_tokens.npz', allow_pickle=False))
        self.path_to_id = {tuple(p): i for i, p in enumerate(self.catalog['decoder_paths'], 1)}
        self.prefix = generation_trie.prefix_allowed_tokens_fn(generation_trie.Trie(self.catalog['decoder_paths']))
        self.max_length = max(map(len, self.catalog['decoder_paths']))
        self.handle = None
        self.encode_memories()
        self.reset('seen')

    def digest(self):
        result = hashlib.sha256()
        for k, v in self.model.state_dict().items():
            result.update(k.encode()); result.update(v.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    def phase(self, stage, **values):
        write(OUT/'worker_status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(),
            updated_at=stamp(), elapsed_seconds=time.monotonic()-self.started, test_read=False, **values))

    def encode_memories(self):
        import torch
        started = time.monotonic()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.memories = {}
        selected = np.flatnonzero(self.tokens['usable'])
        for arm in ('seen', 'attribute'):
            values = torch.zeros(len(self.catalog['items']), self.model.config.d_model, device=self.device)
            for start in range(0, len(selected), 64):
                ix = selected[start:start+64]
                ids = torch.from_numpy(self.tokens[arm][ix]).to(self.device)
                mask = ids.ne(0)
                with torch.no_grad():
                    hidden = self.model.encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)[0]
                    pooled = (hidden * mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
                values[torch.as_tensor(ix, device=self.device)] = pooled
                if start % 640 == 0:
                    self.phase('encode_memory', arm=arm, completed=min(start+64, len(selected)), total=len(selected))
            self.memories[arm] = values
        self.memories['shuffled'] = self.memories['attribute'][torch.from_numpy(self.tokens['donors']).to(self.device)]
        assert all(torch.isfinite(v).all() for v in self.memories.values())
        torch.save({a: v.cpu() for a, v in self.memories.items()}, OUT/'frozen_memories.pt')
        write(OUT/'memory_encoding.json', dict(seconds=time.monotonic()-started,
            items=len(selected), model_digest=self.initial_digest, encoder_frozen=True,
            raw_width=64, memory_vectors_per_history_item=1))

    def reset(self, arm):
        import torch
        from experiment.phase20.lime_clean_gram_core import seed_all
        if self.handle is not None:
            self.handle.remove()
        seed_all(2023)
        self.arm = arm
        self.branch = AttributeMemoryResidual(self.model.config.d_model, 64).to(self.device)
        assert sum(p.numel() for p in self.branch.parameters()) == 131072
        self.handle = self.branch.attach(self.model.lm_head)
        self.model.eval()
        torch.cuda.empty_cache()

    def set_history(self, records):
        import torch
        ids = np.zeros((len(records), max(1, max(len(r['history']) for r in records))), dtype=np.int64)
        for row, record in enumerate(records):
            history = record['history'][::-1]
            assert len(history) <= 20
            ids[row, :len(history)] = history
        ix = torch.from_numpy(ids).to(self.device)
        valid = torch.from_numpy(self.tokens['usable'][ids]).to(self.device)
        self.branch.set_memory(self.memories[self.arm][ix], valid)

    def optimizer(self):
        import torch
        from transformers import get_linear_schedule_with_warmup
        opt = torch.optim.AdamW(self.branch.parameters(), lr=.001, weight_decay=.01)
        schedule = get_linear_schedule_with_warmup(opt, int(.05*len(self.windows)), len(self.windows))
        return opt, schedule

    def update(self, windows, opt, schedule):
        import torch
        opt.zero_grad(set_to_none=True)
        loss_total = 0.
        for indices in windows:
            records = [self.training[i] for i in indices]
            batch = self.collator([self.dataset[i] for i in indices])
            self.set_history(records)
            loss = self.model(input_ids=batch['item_text_ids'].to(self.device),
                attention_mask=batch['item_text_masks'].to(self.device), labels=batch['target_ids'].to(self.device), use_cache=False).loss
            (loss/len(windows)).backward()
            loss_total += float(loss.detach())/len(windows)
            del loss, batch
        norm = torch.nn.utils.clip_grad_norm_(self.branch.parameters(), 1.)
        assert np.isfinite(loss_total) and torch.isfinite(norm)
        assert all(p.grad is None for p in self.model.parameters())
        opt.step(); schedule.step()
        return dict(loss=loss_total, gradient_norm=float(norm))

    def generate(self, index):
        import torch
        record = self.samples[index]
        sample = dict(self.eval_data[index], output=self.catalog['lexical_ids'][1])
        batch = self.collator([sample])
        self.set_history([record])
        with torch.no_grad():
            output = self.model.generate(input_ids=batch['item_text_ids'].to(self.device),
                attention_mask=batch['item_text_masks'].to(self.device), max_length=self.max_length,
                prefix_allowed_tokens_fn=self.prefix, num_beams=50, num_return_sequences=50,
                return_dict_in_generate=True, output_scores=True, length_penalty=1.)
        paths = [p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids = [self.path_to_id[tuple(p)] for p in paths]
        scores = output.sequences_scores.cpu().double().tolist()
        assert len(set(ids)) == 50 and np.isfinite(scores).all()
        ordered = [i for i, s in sorted(zip(ids, scores), key=lambda v: (-v[1], v[0])) if i not in record['history']]
        rank = ordered.index(record['target'])+1 if record['target'] in ordered else 2**30
        return dict(user_id=record['user_id'], cohort=record['cohort'], target=record['target'], ids=ids, scores=scores, rank=rank)

    def profile(self):
        import torch
        from experiment.phase20.lime_clean_gram_core import update_windows
        self.phase('profile')
        parent = rows(OUT/'predictions_parent.jsonl')
        parity = []
        for i in range(3):
            actual = self.generate(i)
            assert actual['ids'] == parent[i]['ids']
            np.testing.assert_array_equal(actual['scores'], parent[i]['scores'])
            parity.append(dict(user_id=actual['user_id'], paths_exact=True, scores_exact=True))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        opt, schedule = self.optimizer()
        longest = sorted(range(len(self.training)), key=lambda i: len(self.training[i]['history']), reverse=True)[:128]
        windows = [list(update_windows(longest, 16, 8))[0]] + self.windows[:3]
        profile = []
        for i, window in enumerate(windows):
            start = time.monotonic()
            result = self.update(window, opt, schedule)
            torch.cuda.synchronize()
            result.update(seconds=time.monotonic()-start, stress=i == 0)
            profile.append(result)
            self.phase('profile', completed=i+1, total=4)
        estimate = float(np.median([p['seconds'] for p in profile[1:]]))*995*3+3600
        write(OUT/'resource_profile.json', dict(records=profile, parent_parity=parity,
            projected_seconds=estimate, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            trial_updates_discarded=4, memory_encoding_seconds=read(OUT/'memory_encoding.json')['seconds']))
        del opt, schedule
        self.reset('seen')
        if estimate > 6.5*3600:
            raise RuntimeError('RESOURCE_BUDGET_STOP: projected %.2f hours' % (estimate/3600))

    def train(self, arm):
        import torch
        from experiment.phase20.lime_clean_gram_core import atomic_save, cpu_tree
        self.reset(arm)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        folder = OUT/arm
        folder.mkdir()
        opt, schedule = self.optimizer()
        started = time.monotonic()
        with (folder/'updates.jsonl').open('x') as stream:
            for update, window in enumerate(self.windows, 1):
                before = time.monotonic()
                values = self.update(window, opt, schedule)
                values.update(update=update, seconds=time.monotonic()-before, lr=opt.param_groups[0]['lr'])
                stream.write(json.dumps(values)+'\n'); stream.flush()
                if update % 100 == 0 or update == 995:
                    atomic_save(folder/'latest.pt', dict(branch=cpu_tree(self.branch.state_dict()),
                        optimizer=cpu_tree(opt.state_dict()), scheduler=schedule.state_dict(), completed_updates=update,
                        parent_sha256=self.cfg['parent_sha256']))
                if update == 1 or update % 10 == 0 or update == 995:
                    self.phase('train', arm=arm, completed=update, total=995, loss=values['loss'])
        assert schedule.last_epoch == 995
        atomic_save(folder/'final.pt', dict(branch=cpu_tree(self.branch.state_dict()), completed_updates=995,
            parent_sha256=self.cfg['parent_sha256'], memory_sha256=sha(OUT/'frozen_memories.pt')))
        write(folder/'training_summary.json', dict(state='COMPLETED', updates=995, training_rows=127328,
            seconds=time.monotonic()-started, checkpoint_sha256=sha(folder/'final.pt')))
        del opt, schedule
        gc.collect(); torch.cuda.empty_cache()

    def evaluate(self, arm):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.phase('generate', arm=arm, completed=0, total=1024)
        with (OUT/('predictions_'+arm+'.jsonl')).open('x') as stream:
            for i in range(1024):
                record = self.generate(i)
                stream.write(json.dumps(record)+'\n'); stream.flush()
                if i % 25 == 0 or i == 1023:
                    self.phase('generate', arm=arm, completed=i+1, total=1024)

    def run(self):
        self.profile()
        for arm in ARMS:
            self.train(arm)
            self.evaluate(arm)
        assert self.digest() == self.initial_digest
        assert sha(ROOT/self.cfg['parent_checkpoint']) == self.cfg['parent_sha256']
        write(OUT/'worker_status.json', dict(state='COMPLETED', updated_at=stamp(),
            backbone_unchanged=True, elapsed_seconds=time.monotonic()-self.started, test_read=False))


def finish():
    samples = read(OUT/'samples.json')
    ranks = {}
    for arm in ('parent',)+ARMS:
        prediction = rows(OUT/('predictions_'+arm+'.jsonl'))
        assert len(prediction) == len(samples) == 1024
        rebuilt = []
        for record, sample in zip(prediction, samples):
            assert (record['user_id'], record['cohort'], record['target']) == (sample['user_id'], sample['cohort'], sample['target'])
            legal = [(score, item) for item, score in zip(record['ids'], record['scores']) if item not in sample['history']]
            order = [item for score, item in sorted(legal, key=lambda v: (-v[0], v[1]))]
            rank = next((i+1 for i, item in enumerate(order) if item == sample['target']), 2**30)
            assert rank == record['rank']
            rebuilt.append(rank)
        ranks[arm] = np.array(rebuilt)
    groups = {}
    for cohort in ('combined1024', 'existing512', 'additional512'):
        ix = np.array([cohort == 'combined1024' or s['cohort'] == cohort for s in samples])
        groups[cohort] = dict(users=int(ix.sum()), metrics={a: metrics(r[ix]) for a, r in ranks.items()},
            contrasts={a: contrast(ranks['attribute'][ix], ranks[a][ix]) for a in ('parent', 'seen', 'shuffled')})
    m = groups['combined1024']['metrics']
    gate = all(m['attribute']['N10'] >= 1.01*m[a]['N10'] and m['attribute']['H10'] >= m[a]['H10'] for a in ('parent', 'seen'))
    gate = gate and all(groups[c]['metrics']['attribute']['N10'] >= groups[c]['metrics']['seen']['N10'] for c in ('existing512', 'additional512'))
    support = gate and m['attribute']['N10'] > m['shuffled']['N10'] and m['attribute']['H10'] >= m['shuffled']['H10']
    reference = read(SOURCE/'summary.json')['domains']['Beauty']['development']['metrics']['fixed_decode']
    result = dict(state='COMPLETED', updated_at=stamp(), groups=groups, incremental_pass=bool(gate),
        description_support=bool(support), independently_recomputed_rankings=4096,
        fixed_decode_existing512_reference=reference, test_read=False, single_domain=True,
        resource_profile=read(OUT/'resource_profile.json'))
    write(OUT/'summary.json', result)
    lines = ['# 原输入保留的描述记忆：Beauty实验结果', '', 'Origin Skill：academic-research-suite。',
        '状态：COMPLETED；'+stamp()+'。原骨干冻结，三支路各完整一轮995次更新，真实beam50评价。', '',
        '| 集合 | 方法 | NDCG@10 | H10 | H50 |', '|---|---|---:|---:|---:|']
    for cohort, group in groups.items():
        for arm, values in group['metrics'].items():
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' % (cohort, arm, values['N10'], values['H10'], values['H50']))
    lines += ['', 'Beauty增量投入门槛：%s；描述对应支持：%s。' % (gate, support),
        '原512人已有固定引导参照：N10=%.6f、H10=%.6f。此值不与合并1024人混排。' % (reference['N10'], reference['H10']), '',
        '四组共4096条排名由另一种排序实现复算一致；逐组配对区间、新增/丢失和资源实测见summary.json。',
        '描述用冻结编码器压缩为每件商品一个向量；新增离线编码、记忆存储和推理支路均是额外成本。',
        '仅一个骨干和适配seed、已研究开发集；未读取test。即使通过也仅支持后续确认，不证明跨域成功或胜过当前完整专家系统。',
        '未通过不自动增加训练轮数、改变学习率或选择另一个臂作为主方案。', '']
    REPORT.write_text('\n'.join(lines))


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    resources = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in resources.splitlines()}
    if free[GPU] < 16384:
        raise RuntimeError('RESOURCE_BUDGET_STOP: less than 16GiB free on declared GPU')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    started = time.monotonic()
    with (OUT/'worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker'], cwd=str(ROOT), env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        write(OUT/'launch.json', dict(started_at=stamp(), supervisor_pid=os.getpid(), worker_pid=process.pid, resources=resources, gpu=GPU))
        while process.poll() is None:
            worker = read(OUT/'worker_status.json') if (OUT/'worker_status.json').exists() else {}
            write(OUT/'status.json', dict(state='RUNNING', updated_at=stamp(), pid=os.getpid(), worker_pid=process.pid,
                worker=worker, elapsed_seconds=time.monotonic()-started, test_read=False))
            if time.monotonic()-started > 7*3600:
                print('Seven-hour hard timeout: stopping only the attribute-memory worker.', flush=True)
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise TimeoutError('Seven-hour attribute-memory hard timeout')
            time.sleep(15)
        assert process.returncode == 0, ('worker_exit', process.returncode, read(OUT/'worker_status.json'))
    finish()
    write(OUT/'status.json', dict(state='COMPLETED', updated_at=stamp(), seconds=time.monotonic()-started,
        incremental_pass=read(OUT/'summary.json')['incremental_pass'], report=str(REPORT.relative_to(ROOT)), test_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'supervise', 'worker'))
    args = parser.parse_args()
    try:
        if args.action == 'prepare': prepare()
        elif args.action == 'worker': Worker().run()
        else: supervise()
    except BaseException as error:
        if OUT.exists():
            (OUT/(args.action+'_failure.txt')).write_text(traceback.format_exc())
            state = 'RESOURCE_BUDGET_STOP' if 'RESOURCE_BUDGET_STOP' in str(error) else 'FAILED'
            write(OUT/('worker_status.json' if args.action == 'worker' else 'status.json'),
                dict(state=state, updated_at=stamp(), error=repr(error), test_read=False))
            if args.action == 'supervise':
                REPORT.write_text('# 原输入保留的描述记忆：执行状态\n\n状态：'+state+'；'+stamp()+
                    '。本批次未完整完成，不能据此判定方法有效或无效。\n\n错误：'+repr(error)+'\n')
        raise
