"""Pinned LIME training and out-of-fold development comparison; never reads test."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now
from experiment.phase20.catalog_scope_core import ranks, topk, metrics, contrast
from experiment.phase20.lime_comparison_core import gate_folds, histories_from_bank, left_pad, semantic_users

sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
from lime_rec.models import SASRec
from lime_rec.itemcf import build_itemcf_model
from lime_rec.experts import ItemCFExpert, LLMSemanticExpert
from lime_rec.evaluation import normalize_score_rows
from lime_rec.data import _read_metadata
from scripts.data.download_gram_data import convert_meta
from scripts.training.train_sasrec import train_sasrec, _seed_everything
from scripts.evaluation.run_repeat_aware_gate import features, RepeatAwareGate

CONFIG = ROOT/'experiment/phase20/lime_comparison_20260920_v1.json'


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


class ProgressOutput:
    def __init__(self, worker, seed, stream):
        self.worker, self.seed, self.stream = worker, seed, stream
        self.pending = ''
        self.start = time.monotonic()

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            found = re.search(r'\[SASRec\] epoch=(\d+) loss=([\d.]+)', line)
            if found:
                epoch = int(found[1])
                elapsed = time.monotonic()-self.start
                self.worker.status('训练SASRec', seed=self.seed, epoch=epoch, max_epochs=200,
                    loss=float(found[2]), seconds_per_epoch=round(elapsed/epoch, 2),
                    current_seed_max_remaining_seconds=round((200-epoch)*elapsed/epoch),
                    completed_seeds=self.worker.completed_seeds, early_stopping=True)
        return len(text)

    def flush(self):
        self.stream.flush()


class Worker:
    def __init__(self, domain, smoke=False, resume_from=None, recovery_attempt='recovery_v1'):
        self.c = read(CONFIG)
        self.domain, self.smoke = domain, smoke
        self.resume_from = Path(resume_from) if resume_from else None
        self.root = ROOT/self.c['output']
        if self.resume_from:
            assert self.resume_from == self.root/domain
            assert recovery_attempt in ('recovery_v1', 'recovery_v2')
            self.root = self.root/recovery_attempt
        self.out = self.root/(domain+'_smoke' if smoke else domain)
        self.out.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.inputs = {}
        self.completed_seeds = []
        self.timings = {}
        torch.set_num_threads(self.c['cpu_threads'])
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(self.c['cuda_memory_fraction'])
        self.device = torch.device('cuda')
        self.status('核对用户、ID与训练边界')

    def status(self, stage, state='RUNNING', **extra):
        write(self.out/'status.json', dict(state=state, stage=stage, updated_at=now(),
            domain=self.domain, pid=os.getpid(), elapsed_seconds=round(time.monotonic()-self.started, 1),
            peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1), test_read=False, **extra))

    def use(self, path):
        path = Path(path)
        key = str(path.relative_to(ROOT))
        assert 'test.json' not in key and '/test/' not in key
        if key not in self.inputs:
            self.inputs[key] = sha(path)
        return path

    def load(self):
        old = ROOT/f'artifacts/phase20/{self.c["calibration_source"]}/{self.domain}'
        self.old = old
        self.q = arrays(self.use(old/'queries.npz'))
        self.users, self.targets = self.q['users'], self.q['targets']
        self.cal, self.eval = self.q['calibration_indices'], self.q['evaluation_indices']
        self.n, self.v = len(self.users), len(self.q['frequencies'])-1
        near = ROOT/'artifacts/phase20/nearest_comparison_20260918'
        bank = arrays(self.use(near/f'{self.domain}.train_sequences.npz'))
        self.histories = histories_from_bank(bank, self.users)
        for i, h in enumerate(self.histories):
            assert h[-20:] == self.q['history'][i][self.q['history'][i] > 0].tolist()
            assert int(self.targets[i]) not in h
        np.testing.assert_array_equal(np.bincount(bank['items'], minlength=self.v+1), self.q['frequencies'])
        self.records = [json.loads(line) for line in self.use(near/f'{self.domain}.items.jsonl').open()]
        assert [r['item_id'] for r in self.records] == list(range(1, self.v+1))
        self.source_ids = [r['source_item_id'] for r in self.records]
        assert len(set(self.source_ids)) == self.v
        raw_text = self.use(ROOT/f'GRAM/rec_datasets/{self.domain}/item_plain_text.txt')
        metadata_path = self.out/'author_metadata.jsonl'
        convert_meta(raw_text, metadata_path)
        self.item_text = _read_metadata(metadata_path)
        assert all(item in self.item_text for item in self.source_ids)
        self.folds = gate_folds(self.users, self.eval, self.c['folds'], self.c['fold_seed'])
        np.testing.assert_array_equal(np.sort(np.concatenate(self.folds)), self.eval)
        assert not set(self.cal) & set(self.eval)
        self.user_order = np.argsort(self.users)
        self.A = arrays(self.use(old/'A_scores.npz'))
        self.old_eval = arrays(self.use(old/'evaluation.npz'))
        np.testing.assert_array_equal(self.old_eval['indices'], self.eval)
        self.weight = read(self.use(old/'selection.json'))['primary_weight']
        # These score files are immutable inputs shared by every candidate source.
        self.directory = np.load(self.use(old/'directory_z.npy'), mmap_mode='r')
        self.collaborative = np.load(self.use(old/'collaborative.npy'), mmap_mode='r')
        assert self.directory.shape == self.collaborative.shape == (self.n, self.v+1)
        write(self.out/'cohorts.json', dict(calibration_indices=self.cal.tolist(),
            evaluation_indices=self.eval.tolist(), gate_evaluation_folds=[f.tolist() for f in self.folds],
            note='SASRec sees no evaluation labels. Each gate excludes its predicted fold.'))
        write(self.out/'input_manifest.json', self.inputs)

    def train(self):
        if self.resume_from:
            reused = {}
            for seed in self.c['seeds']:
                path = self.use(self.resume_from/f'sasrec_seed{seed}.pt')
                saved = torch.load(path, map_location='cpu')
                assert saved['train_complete'] is True and saved['seed'] == seed
                assert saved['source_commit'] == self.c['author_commit']
                assert saved['num_items'] == self.v and saved['maxlen'] == 50
                assert saved['loss_label_smoothing'] == .1 and saved['gradient_clip'] == 1.0
                assert all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values())
                shutil.copy2(path, self.out/path.name)
                assert sha(path) == sha(self.out/path.name)
                reused[str(seed)] = dict(source=str(path.relative_to(ROOT)), sha256=sha(path),
                    selected_epoch=saved['epoch'], original_training_seconds=saved['training_seconds'])
                self.timings[f'sasrec_seed{seed}_reused_training_seconds'] = saved['training_seconds']
                self.completed_seeds.append(seed)
            write(self.out/'reused_training.json', dict(new_sasrec_optimizer_steps=0, checkpoints=reused))
            self.status('已核对并复用三个SASRec检查点', completed_seeds=self.completed_seeds)
            return
        for seed in self.c['seeds']:
            self.status('训练SASRec', seed=seed, epoch=0, max_epochs=200, completed_seeds=self.completed_seeds)
            _seed_everything(seed)
            started = time.monotonic()
            best = self.out/f'sasrec_seed{seed}.best.pt'
            kwargs = dict(self.c['sasrec'])
            old_stdout = sys.stdout
            try:
                sys.stdout = ProgressOutput(self, seed, old_stdout)
                model = train_sasrec(
                    train_seqs=[self.histories[i] for i in self.user_order], num_items=self.v,
                    valid_seqs=[self.histories[i] for i in self.cal], valid_targets=self.targets[self.cal].tolist(),
                    ckpt_path=str(best), ckpt_meta=dict(seed=seed, test_read=False), device=self.device, **kwargs)
            finally:
                sys.stdout = old_stdout
            assert best.exists(), 'No positive calibration Recall checkpoint; preserve as training failure'
            selected = torch.load(best, map_location='cpu')
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), selected['state_dict'][k], rtol=0, atol=0)
            selected.update(train_complete=True, source_commit=self.c['author_commit'],
                environment=dict(python=sys.version, torch=torch.__version__), calibration_users=len(self.cal),
                loss_label_smoothing=0.1, gradient_clip=1.0, training_seconds=time.monotonic()-started)
            torch.save(selected, self.out/f'sasrec_seed{seed}.pt')
            self.timings[f'sasrec_seed{seed}'] = selected['training_seconds']
            self.completed_seeds.append(seed)
            del model, selected
            gc.collect(); torch.cuda.empty_cache()

    def static_experts(self):
        start = time.monotonic()
        histories = {str(u): [self.source_ids[j-1] for j in h] for u, h in zip(self.users, self.histories)}
        self.status('核对已有ItemCF' if self.resume_from else '构建训练集ItemCF邻居')
        if self.resume_from:
            model = read(self.use(self.resume_from/'itemcf.json'))
            assert model['num_training_users'] == self.n
            assert model['num_training_interactions'] == sum(map(len, self.histories))
            assert model['num_catalog_items'] == self.v
            shutil.copy2(self.resume_from/'itemcf.json', self.out/'itemcf.json')
        else:
            model = build_itemcf_model(dataset_name=self.domain, histories=histories,
                item_ids=self.source_ids, top_k=100, window_size=20)
            write(self.out/'itemcf.json', model)
        assert model['training_only']
        cf = ItemCFExpert(str(self.out/'itemcf.json'), self.source_ids)
        if self.resume_from:
            cf_scores = np.load(self.use(self.resume_from/'itemcf_scores.npy'), mmap_mode='r')
            assert cf_scores.shape == (self.n, self.v) and cf_scores.dtype == np.float32
            for i in np.linspace(0, self.n-1, 32, dtype=int):
                np.testing.assert_array_equal(cf_scores[i], cf.score(str(self.users[i]), histories[str(self.users[i])]))
            write(self.out/'reused_itemcf.json', dict(path=str((self.resume_from/'itemcf_scores.npy').relative_to(ROOT)),
                checked_rows=32, sha256=sha(self.resume_from/'itemcf_scores.npy')))
        else:
            cf_scores = np.lib.format.open_memmap(self.out/'itemcf_scores.npy', mode='w+', dtype='float32', shape=(self.n, self.v))
            for i, u in enumerate(self.users):
                cf_scores[i] = cf.score(str(u), histories[str(u)])
                if i % 2048 == 0: self.status('计算ItemCF全目录分数', done=i, total=self.n)
            cf_scores.flush()
        self.timings['itemcf_build_and_score'] = time.monotonic()-start
        asset = self.root/'assets'
        while True:
            status = read(asset/'status.json') if (asset/'status.json').exists() else {}
            if status.get('state') == 'COMPLETED': break
            assert status.get('state') != 'FAILED', status
            assert time.monotonic()-self.started < self.c['domain_timeout_seconds']
            self.status('等待独立BGE模型下载', asset_status=status)
            time.sleep(10)
        for path in (asset/'bge-base-en-v1.5').rglob('*'):
            if path.is_file(): self.use(path)
        self.use(asset/'manifest.json')
        from transformers import AutoModel, AutoTokenizer
        local = str(asset/'bge-base-en-v1.5')
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        encoder = AutoModel.from_pretrained(local, local_files_only=True).to(self.device).eval()
        # Use the author's converter AND loader. Its converted description also
        # includes repeated fields and price/salesrank; do not silently simplify it.
        texts = [self.item_text[item] for item in self.source_ids]
        embeddings = np.zeros((self.v, 768), dtype=np.float32)
        start = time.monotonic()
        with torch.no_grad():
            for i in range(0, self.v, self.c['embedding_batch']):
                self.status('离线编码BGE商品文本', done=i, total=self.v)
                batch = tokenizer(texts[i:i+self.c['embedding_batch']], padding=True, truncation=True,
                    max_length=512, return_tensors='pt').to(self.device)
                # Pinned SentenceTransformer modules: Transformer -> CLS pooling -> Normalize.
                output = F.normalize(encoder(**batch).last_hidden_state[:, 0], dim=-1)
                embeddings[i:i+len(output)] = output.cpu().numpy()
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1., atol=2e-6)
        np.savez_compressed(self.out/'embeddings.npz', item_ids=np.asarray(self.source_ids), embeddings=embeddings)
        self.timings['bge_offline_encoding'] = time.monotonic()-start
        del encoder, tokenizer, output, batch
        gc.collect(); torch.cuda.empty_cache()
        start = time.monotonic()
        user_emb = semantic_users(self.histories, embeddings)
        semantic = LLMSemanticExpert(self.source_ids, {}, embeddings=embeddings)
        sem_scores = np.lib.format.open_memmap(self.out/'semantic_scores.npy', mode='w+', dtype='float32', shape=(self.n, self.v))
        item_tensor = torch.as_tensor(semantic.item_embeddings_norm, device=self.device)
        with torch.no_grad():
            for i in range(0, self.n, self.c['score_batch']):
                sem_scores[i:i+self.c['score_batch']] = (torch.as_tensor(user_emb[i:i+self.c['score_batch']],
                    device=self.device) @ item_tensor.T).cpu().numpy()
        sem_scores.flush()
        errors = []
        for i in np.linspace(0, self.n-1, 12, dtype=int):
            expected = semantic.score(str(self.users[i]), histories[str(self.users[i])])
            np.testing.assert_allclose(sem_scores[i], expected, atol=1e-6, rtol=1e-5)
            errors.append(float(np.max(np.abs(sem_scores[i]-expected))))
        self.timings['semantic_online_scores'] = time.monotonic()-start
        write(self.out/'semantic_parity.json', dict(author_score_max_abs_error=max(errors),
            offline_model=self.c['encoder'], pooling='CLS', normalization='L2', max_seq_length=512))
        del item_tensor, semantic, model, cf
        return cf_scores, sem_scores

    def score_sasrec(self, seed):
        state = torch.load(self.out/f'sasrec_seed{seed}.pt', map_location='cpu')
        model = SASRec(self.v, 64, 50, 2, 2, 0., architecture='legacy').to(self.device)
        model.load_state_dict(state['state_dict']); model.eval()
        scores = np.lib.format.open_memmap(self.out/f'sasrec_scores_seed{seed}.npy', mode='w+', dtype='float32', shape=(self.n, self.v))
        started = time.monotonic()
        with torch.no_grad():
            for i in range(0, self.n, self.c['score_batch']):
                batch = left_pad(self.histories[i:i+self.c['score_batch']])
                values = model.predict(torch.as_tensor(batch, device=self.device))[:, 1:]
                assert bool(torch.isfinite(values).all())
                scores[i:i+len(batch)] = values.cpu().numpy()
                if i % 2048 == 0: self.status('计算SASRec全目录分数', seed=seed, done=i, total=self.n)
        scores.flush()
        self.timings[f'sasrec_online_scores_seed{seed}'] = time.monotonic()-started
        del model, state
        gc.collect(); torch.cuda.empty_cache()
        return scores

    def tensor_batch(self, indices, experts):
        raw = [np.asarray(e[indices]) for e in experts]
        scores = np.stack([normalize_score_rows(s) for s in raw], axis=1)
        assert np.isfinite(scores).all()
        hist = np.zeros((len(indices), self.v), dtype=bool)
        for row, index in enumerate(indices): hist[row, np.asarray(self.histories[index])-1] = True
        return (torch.as_tensor(scores, device=self.device),
            torch.as_tensor([len(self.histories[i]) for i in indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.targets[indices]-1, device=self.device), torch.as_tensor(hist, device=self.device))

    def fit_gate(self, seed, name, fit, predict, experts):
        assert not set(map(int, fit)) & set(map(int, predict))
        torch.manual_seed(seed)
        g = self.c['gate']
        gate = RepeatAwareGate(14, g['initial_weights'], g['max_penalty']).to(self.device)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=g['lr'], weight_decay=g['weight_decay'])
        fit = np.asarray(sorted(map(int, fit), key=lambda i: str(self.users[i])))
        for epoch in range(g['epochs']):
            losses = []
            for start in range(0, len(fit), g['batch_size']):
                ix = fit[start:start+g['batch_size']]
                s, lengths, targets, h = self.tensor_batch(ix, experts)
                weights, penalty = gate(features(s, lengths, h))
                fused = (weights.unsqueeze(2)*s).sum(1)-penalty.unsqueeze(1)*h
                loss = F.cross_entropy(g['score_scale']*fused, targets)
                assert bool(torch.isfinite(loss))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            self.status('拟合LIME融合器', seed=seed, gate=name, epoch=epoch+1,
                max_epochs=g['epochs'], fit_users=len(fit), loss=float(np.mean(losses)))
        torch.save(dict(state_dict=gate.state_dict(), fit_indices=fit, predicted_indices=predict,
            seed=seed, config=g, test_read=False), self.out/f'gate_seed{seed}_{name}.pt')
        gate.eval()
        return gate

    def evaluate_gate(self, seed, gate, ix_all, experts, store, rows):
        ids = np.arange(1, self.v+1)
        started = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(ix_all), self.c['score_batch']):
                ix = np.asarray(ix_all[start:start+self.c['score_batch']])
                scores, lengths, targets, hist = self.tensor_batch(ix, experts)
                weights, penalty = gate(features(scores, lengths, hist))
                fused = ((weights.unsqueeze(2)*scores).sum(1)-penalty.unsqueeze(1)*hist).cpu().numpy()
                repeat = fused.copy()
                for j, i in enumerate(ix): fused[j, self.q['history'][i][self.q['history'][i] > 0]-1] = -np.inf
                common = ((1-self.weight)*np.asarray(self.directory[ix], dtype=np.float64)
                    +self.weight*np.asarray(self.collaborative[ix], dtype=np.float64))[:, 1:]
                for j, i in enumerate(ix): common[j, self.q['history'][i][self.q['history'][i] > 0]-1] = -np.inf
                lime_ids = topk(fused, ids)
                gen_ids = self.q['candidate_ids'][ix]
                arms = {'LIME': (fused, ids), 'LIME_repeat_allowed': (repeat, ids),
                    'LIME_scores_G50': (np.take_along_axis(fused, gen_ids-1, 1), gen_ids),
                    'common_G50': (np.take_along_axis(common, gen_ids-1, 1), gen_ids),
                    'common_LIME50': (np.take_along_axis(common, lime_ids-1, 1), lime_ids)}
                for e, label in enumerate(('SASRec', 'ItemCF', 'BGE')):
                    raw = np.asarray(experts[e][ix]).copy()
                    for j, i in enumerate(ix): raw[j, self.q['history'][i][self.q['history'][i] > 0]-1] = -np.inf
                    selected = topk(raw, ids)
                    arms[label] = raw, ids
                    if label in ('SASRec', 'BGE'):
                        arms['common_'+label+'50'] = np.take_along_axis(common, selected-1, 1), selected
                output_rows = np.asarray([rows[int(i)] for i in ix])
                for label, (values, item_ids) in arms.items():
                    rr = ranks(values, item_ids, self.targets[ix])
                    tt = topk(values, item_ids)
                    # Independent rank check via saved ordered top50, with misses clipped at >50.
                    match = tt == self.targets[ix, None]
                    expected = np.where(match.any(1), match.argmax(1)+1, 2**30)
                    np.testing.assert_array_equal(np.minimum(rr, 51), np.minimum(expected, 51))
                    store['ranks'][label][output_rows] = rr
                    store['top50'][label][output_rows] = tt
                store['weights'][output_rows] = weights.cpu().numpy()
                store['penalty'][output_rows] = penalty.cpu().numpy()
        return time.monotonic()-started

    def new_results(self):
        names = ['LIME', 'LIME_repeat_allowed', 'LIME_scores_G50', 'common_G50',
            'common_LIME50', 'SASRec', 'ItemCF', 'BGE', 'common_SASRec50', 'common_BGE50']
        n = len(self.eval)
        return dict(ranks={k: np.full(n, -1, dtype=np.int64) for k in names},
            top50={k: np.zeros((n, 50), dtype=np.int64) for k in names},
            weights=np.zeros((n, 3)), penalty=np.zeros(n))

    def summarize_seed(self, seed, name, result):
        for rr in result['ranks'].values(): assert (rr > 0).all()
        np.testing.assert_array_equal(result['ranks']['common_G50'], self.old_eval['ranks'][2])
        # No refitting or reinterpretation of the three historical A heads.
        a_ranks = self.old_eval['A_ranks']
        arms = list(result['ranks'])
        np.savez_compressed(self.out/f'evaluation_seed{seed}_{name}.npz',
            users=self.users[self.eval], indices=self.eval, targets=self.targets[self.eval],
            arm_names=np.asarray(arms), ranks=np.stack([result['ranks'][k] for k in arms]),
            top50=np.stack([result['top50'][k] for k in arms]),
            weights=result['weights'], penalty=result['penalty'], A_seeds=self.A['seeds'], A_ranks=a_ranks)
        tail = self.q['tail_mask'][self.eval]
        summary = dict(seed=seed, gate_protocol=name, evaluation_users=len(self.eval),
            metrics={k: dict(all=metrics(rr), low_frequency=metrics(rr[tail]),
                other=metrics(rr[~tail])) for k, rr in result['ranks'].items()},
            A_head_seeds={str(s): metrics(rr) for s, rr in zip(self.A['seeds'], a_ranks)},
            A_repeat_allowed={str(s): metrics(rr[self.eval]) for s, rr in zip(self.A['seeds'], self.A['original_ranks'])},
            comparisons={}, mean_weights=result['weights'].mean(0).tolist(),
            mean_penalty=float(result['penalty'].mean()), test_read=False)
        for label, first, second in [('A_primary_minus_LIME', a_ranks[0], result['ranks']['LIME']),
            ('G50_minus_LIME50_common', result['ranks']['common_G50'], result['ranks']['common_LIME50']),
            ('G50_minus_SASRec50_common', result['ranks']['common_G50'], result['ranks']['common_SASRec50']),
            ('G50_minus_BGE50_common', result['ranks']['common_G50'], result['ranks']['common_BGE50']),
            ('LIME_scores_G50_minus_full', result['ranks']['LIME_scores_G50'], result['ranks']['LIME'])]:
            summary['comparisons'][label] = contrast(first, second, self.c['bootstrap_repeats'], 20260920)
        write(self.out/f'summary_seed{seed}_{name}.json', summary)
        return summary

    def run(self):
        self.load()
        if self.smoke:
            _seed_everything(987)
            longest = sorted(range(self.n), key=lambda i: len(self.histories[i]), reverse=True)[:128]
            model = train_sasrec([self.histories[i] for i in longest], self.v, hidden=64, maxlen=50,
                num_layers=2, num_heads=2, dropout=.2, batch_size=32, epochs=1, loss_name='ce', device=self.device)
            with torch.no_grad():
                scores = model.predict(torch.as_tensor(left_pad([self.histories[i] for i in self.cal[:8]]), device=self.device))
                assert bool(torch.isfinite(scores).all())
            self.status('长序列训练与完整目录前向通过', state='PRECHECK_PASSED')
            return
        self.train()
        shared = self.static_experts()
        summaries = {}
        rows = {int(index): row for row, index in enumerate(self.eval)}
        for seed in self.c['seeds']:
            sequential = self.score_sasrec(seed)
            experts = [sequential, *shared]
            primary = self.new_results()
            start = time.monotonic()
            for f, predicted in enumerate(self.folds):
                fit = np.setdiff1d(np.arange(self.n), predicted)
                gate = self.fit_gate(seed, f'fold{f}', fit, predicted, experts)
                self.evaluate_gate(seed, gate, predicted, experts, primary, rows)
            self.timings[f'crossfit_gates_and_evaluation_seed{seed}'] = time.monotonic()-start
            summaries[f'{seed}_crossfit'] = self.summarize_seed(seed, 'crossfit', primary)
            del primary, gate
            cal_only = self.new_results()
            gate = self.fit_gate(seed, 'cal2048', self.cal, self.eval, experts)
            self.evaluate_gate(seed, gate, self.eval, experts, cal_only, rows)
            summaries[f'{seed}_cal2048'] = self.summarize_seed(seed, 'cal2048', cal_only)
            del cal_only, gate, sequential
            gc.collect(); torch.cuda.empty_cache()
        self.status('核验输入未修改，汇总全部种子')
        for path, digest in self.inputs.items(): assert sha(ROOT/path) == digest, path
        write(self.out/'input_manifest.json', self.inputs)
        write(self.out/'summary.json', dict(state='COMPLETED', domain=self.domain, results=summaries,
            timings=self.timings, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            seed_level_note='LIME seeds are independent SASRec trainings; A seeds vary only its small head.',
            limitation='Historical development validation with cross-fitted gate, not an untouched test.'))
        write(self.out/'verification.json', dict(state='VERIFIED', common_G50_matches_previous_C=True,
            every_saved_top50_matches_reported_rank=True, gate_fit_predict_disjoint=True,
            input_sha256_unchanged=True, test_read=False, expected_evaluation_users=len(self.eval)))
        self.status('本域全部种子与候选对照完成', state='COMPLETED', completed_seeds=self.completed_seeds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', required=True, choices=['Toys', 'Beauty'])
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume-from')
    p.add_argument('--recovery-attempt', choices=['recovery_v1', 'recovery_v2'], default='recovery_v1')
    args = p.parse_args()
    w = Worker(args.domain, args.smoke, args.resume_from, args.recovery_attempt)
    try:
        w.run()
    except BaseException as e:
        (w.out/'failure.txt').write_text(traceback.format_exc())
        w.status('本域异常，保留全部记录', state='FAILED', error=repr(e))
        raise


if __name__ == '__main__':
    main()
