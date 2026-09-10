"""Full-data RaSeRec training followed by its GRAM memory adaptation.

No test split is loaded. Each stage has explicit outputs, full validation, and a
shared hard deadline. Failures are recorded and are not automatically retried.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.raserec_author import DuoRec
from experiment.phase18.core.raserec_data import (
    MemoryCollator, MemoryDataset, PositiveCases, memory_batch,
    sequence_tensors, training_frequencies,
)
from experiment.phase18.core.raserec_gram import RaSeRecGram, RaSeRecReader, ReaderConfig, exact_case_neighbors
from experiment.phase18.protocol.s18_diff_gram import CatalogDecoder, load_inputs, seed_everything

ROOT = Path(__file__).resolve().parents[3]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp)
    tmp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def metrics(ranks):
    ranks = np.asarray(ranks)
    return {'users': len(ranks), **{
        f'{name}@{k}': float(np.mean((ranks <= k) * (1 if name == 'hit' else 1/np.log2(ranks+1))))
        for k in (5, 10, 20, 50) for name in ('hit', 'ndcg')}}


def ranks_from_top(top, targets):
    matched = top.eq(targets[:, None])
    return torch.where(matched.any(1), matched.long().argmax(1) + 1,
                       torch.full_like(targets, top.size(1) + 1)).cpu().tolist()


class Run:
    def __init__(self, config, config_path, output, device):
        self.config, self.config_path, self.output, self.device = config, config_path, output, device
        self.stage = 'initializing'
        self.started = time.time()
        self.data_config = json.loads((ROOT/config['data_config']).read_text())
        self.catalog, self.train, self.validation, self.data_manifest, self.tokenizer, self.collator = load_inputs(self.data_config)
        self.num_items = len(self.catalog['items'])
        self.seq = {s: sequence_tensors(d.records) for s, d in [('train', self.train), ('validation', self.validation)]}
        all_users = sorted({r['user_id'] for r in self.train.records})
        user_ids = {u: i for i, u in enumerate(all_users)}
        self.users = {s: torch.tensor([user_ids[r['user_id']] for r in d.records])
                      for s, d in [('train', self.train), ('validation', self.validation)]}
        self.frequencies = training_frequencies(self.train.records, self.num_items)
        with (ROOT/config['pcrf']['reference']).open() as f:
            self.reference = {r['user_id']: r for r in csv.DictReader(f, delimiter='\t')}
        for row in self.validation.records:
            ref = self.reference[row['user_id']]
            if self.frequencies[row['target']] != int(ref['target_frequency']):
                raise AssertionError('Training-frequency identity failed')
        self.rng = np.random.default_rng(config['seed'])

    def event(self, event, **values):
        row = {'time': time.time(), 'stage': self.stage, 'event': event, **values}
        with (self.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        write_json(self.output/'status.json', {'pid': os.getpid(), 'state': 'running',
                   'elapsed_seconds': round(time.time()-self.started, 2), **row})

    def snapshot(self):
        files = [self.config_path, ROOT/self.config['data_config'],
                 ROOT/'experiment/phase18/run_stage18_raserec_gram.sh',
                 ROOT/'third_party/RaSeRec/SOURCE.json']
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file and file.endswith('.py') and ROOT in Path(file).resolve().parents:
                files.append(Path(file).resolve())
        hashes = {}
        for file in sorted(set(files)):
            rel = file.relative_to(ROOT)
            dst = self.output/'sources'/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst)
            hashes[str(rel)] = sha256(file)
        write_json(self.output/'manifest.json', {
            'config': self.config, 'data_config': self.data_config,
            'prepared_manifest': self.data_manifest, 'source_sha256': hashes,
            'torch': torch.__version__, 'device': str(self.device),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'test_read': False, 'started_at': self.started,
            'deadline': self.started+self.config['max_wall_seconds'],
            'pcrf_checkpoint_sha256': sha256(ROOT/self.config['pcrf']['checkpoint'])})

    def encoder(self, load=False):
        model = DuoRec(self.config['pretrain'], SimpleNamespace(num_items=self.num_items)).to(self.device)
        if load:
            model.load_state_dict(torch.load(self.output/'pretrain/best.pt', map_location='cpu')['model'], strict=True)
        return model

    @torch.no_grad()
    def evaluate_encoder(self, model):
        model.eval()
        histories, lengths, targets = self.seq['validation']
        ranks = []
        for start in range(0, len(targets), 1024):
            stop = start+1024
            q = model(histories[start:stop].to(self.device), lengths[start:stop].to(self.device))
            scores = q @ model.item_embedding.weight.T
            scores[:, 0] = -torch.inf
            ranks.extend(ranks_from_top(scores.topk(50, dim=1).indices, targets[start:stop].to(self.device)))
        return metrics(ranks)

    def pretrain(self):
        self.stage = 'pretrain'
        cfg = self.config['pretrain']
        destination = self.output/'pretrain'
        destination.mkdir(exist_ok=True)
        if (destination/'best.pt').exists():
            raise FileExistsError('Pretraining already has results; choose another stage or a new output')
        model = self.encoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        histories, lengths, targets = self.seq['train']
        positives = PositiveCases(self.train.records)
        best, bad = -math.inf, 0
        for epoch in range(1, cfg['epochs']+1):
            model.train()
            permutation = torch.randperm(len(targets))
            total = 0.0
            for start in range(0, len(targets), cfg['train_batch_size']):
                ix = permutation[start:start+cfg['train_batch_size']]
                sem = positives.sample(ix, self.rng)
                batch = {'history': histories[ix], 'length': lengths[ix], 'target': targets[ix],
                         'sem_aug': histories[sem], 'sem_aug_lengths': lengths[sem]}
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model.calculate_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite pretraining loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(ix)
                if start % (cfg['train_batch_size']*25) == 0:
                    self.event('training', epoch=epoch, seen=start+len(ix), total=len(targets), loss=float(loss.detach()))
            result = self.evaluate_encoder(model)
            improved = result['ndcg@10'] > best
            bad = 0 if improved else bad+1
            if improved:
                best = result['ndcg@10']
                save(destination/'best.pt', {'model': cpu_state(model), 'epoch': epoch, 'metrics': result})
            save(destination/'latest.pt', {'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                                          'epoch': epoch, 'torch_rng': torch.get_rng_state(),
                                          'numpy_rng': self.rng.bit_generator.state})
            self.event('validation', epoch=epoch, loss=total/len(targets), metrics=result, best=best, bad_epochs=bad)
            if epoch >= 10 and bad >= cfg['patience']:
                break
        write_json(destination/'summary.json', {'best_ndcg10': best, 'last_epoch': epoch, 'test_read': False})
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def bank(self):
        self.stage = 'bank'
        destination = self.output/'bank'
        destination.mkdir(exist_ok=True)
        if (destination/'manifest.json').exists():
            raise FileExistsError('Memory bank already exists')
        model = self.encoder(load=True).eval()
        queries = {}
        for split in ('train', 'validation'):
            histories, lengths, _ = self.seq[split]
            parts = []
            for start in range(0, len(histories), 1024):
                parts.append(model(histories[start:start+1024].to(self.device),
                                   lengths[start:start+1024].to(self.device)).cpu())
            queries[split] = torch.cat(parts)
        item_embeddings = model.item_embedding.weight.detach().cpu()
        keys = queries['train'].to(self.device)
        key_users = self.users['train'].to(self.device)
        for split in ('train', 'validation'):
            chunks = []
            for start in range(0, len(queries[split]), 4096):
                chunks.append(exact_case_neighbors(
                    queries[split][start:start+4096].to(self.device), keys,
                    self.users[split][start:start+4096].to(self.device), key_users,
                    self.config['reader']['topk'], chunk_size=256))
                self.event('retrieval', split=split, seen=min(start+4096, len(queries[split])))
            indices = torch.cat(chunks)
            if self.users['train'][indices].eq(self.users[split][:, None]).any():
                raise AssertionError('Self-user memory leak')
            save(destination/f'{split}.pt', {'queries': queries[split], 'indices': indices,
                 'users': [r['user_id'] for r in getattr(self, split).records]})
        save(destination/'values.pt', {'keys': queries['train'],
             'values': item_embeddings[self.seq['train'][2]], 'item_embeddings': item_embeddings})
        write_json(destination/'manifest.json', {
            'pretrain_sha256': sha256(self.output/'pretrain/best.pt'),
            'prepared_files': self.data_manifest['prepared_files'],
            'num_cases': len(self.train), 'exclusion': 'all cases of query user',
            'retrieval': 'exact cosine top20; no query-target input; no IVF approximation',
            'files': {p.name: sha256(p) for p in destination.glob('*.pt')}, 'test_read': False})
        del model, keys
        gc.collect()
        torch.cuda.empty_cache()

    def load_bank(self):
        path = self.output/'bank'
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest['prepared_files'] != self.data_manifest['prepared_files']:
            raise AssertionError('Bank/data mismatch')
        for file, expected in manifest['files'].items():
            if sha256(path/file) != expected:
                raise AssertionError('Bank changed after construction')
        values = torch.load(path/'values.pt', map_location='cpu')
        splits = {s: torch.load(path/f'{s}.pt', map_location='cpu') for s in ('train', 'validation')}
        for split, content in splits.items():
            if content['users'] != [r['user_id'] for r in getattr(self, split).records]:
                raise AssertionError('Memory lookup row-order mismatch')
        return values, splits

    @torch.no_grad()
    def evaluate_reader(self, reader, values, split):
        reader.eval()
        ranks = []
        targets = self.seq['validation'][2]
        for start in range(0, len(targets), 1024):
            ix = split['indices'][start:start+1024]
            output = reader(split['queries'][start:start+1024].to(self.device),
                            values['keys'][ix].to(self.device), values['values'][ix].to(self.device))
            scores = output @ values['item_embeddings'].to(self.device).T
            scores[:, 0] = -torch.inf
            ranks.extend(ranks_from_top(scores.topk(50, dim=1).indices, targets[start:start+1024].to(self.device)))
        return metrics(ranks)

    def reader(self):
        self.stage = 'reader'
        cfg = self.config['reader_training']
        destination = self.output/'reader'
        destination.mkdir(exist_ok=True)
        if (destination/'best.pt').exists():
            raise FileExistsError('Reader training already has results')
        values, splits = self.load_bank()
        reader = RaSeRecReader(ReaderConfig(**self.config['reader'])).to(self.device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=cfg['learning_rate'])
        item_embeddings = values['item_embeddings'].to(self.device)
        targets = self.seq['train'][2]
        best, bad = -math.inf, 0
        for epoch in range(1, cfg['epochs']+1):
            reader.train()
            order = torch.randperm(len(targets))
            total = 0.0
            for start in range(0, len(targets), cfg['batch_size']):
                ix = order[start:start+cfg['batch_size']]
                cases = splits['train']['indices'][ix]
                output = reader(splits['train']['queries'][ix].to(self.device),
                                values['keys'][cases].to(self.device), values['values'][cases].to(self.device))
                loss = F.cross_entropy(output @ item_embeddings.T, targets[ix].to(self.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite reader loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(ix)
            result = self.evaluate_reader(reader, values, splits['validation'])
            improved = result['ndcg@10'] > best
            bad = 0 if improved else bad+1
            if improved:
                best = result['ndcg@10']
                save(destination/'best.pt', {'model': cpu_state(reader), 'epoch': epoch, 'metrics': result})
            self.event('validation', epoch=epoch, loss=total/len(targets), metrics=result, best=best, bad_epochs=bad)
            if epoch >= 10 and bad >= cfg['patience']:
                break
        write_json(destination/'summary.json', {'best_ndcg10': best, 'last_epoch': epoch, 'test_read': False})
        del reader, optimizer, item_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    def gram(self):
        from experiment.phase17.core.full_latte_gram_backend import create_fresh_gram_model
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        values, splits = self.load_bank()
        cfg = self.data_config
        if sha256(ROOT/cfg['parent_checkpoint']) != cfg['parent_sha256']:
            raise AssertionError('GRAM parent changed')
        gram = create_fresh_gram_model(ROOT, 'G0_GRAM_B0_FRESH', self.tokenizer, seed=self.config['seed'])
        parent = torch.load(ROOT/cfg['parent_checkpoint'], map_location='cpu')
        gram.resize_token_embeddings(parent['shared.weight'].size(0))
        gram.load_state_dict(parent, strict=True)
        reader = RaSeRecReader(ReaderConfig(**self.config['reader']))
        reader.load_state_dict(torch.load(self.output/'reader/best.pt', map_location='cpu')['model'], strict=True)
        model = RaSeRecGram(gram, reader, values['item_embeddings']).to(self.device)
        install_cross_cache_reuse(model.gram, self.config['gram_training']['beam_size'])
        collators = {s: MemoryCollator(self.collator, splits[s]['queries'], splits[s]['indices'],
                                      values['keys'], values['values']) for s in splits}
        datasets = {s: MemoryDataset(getattr(self, s).records, self.catalog) for s in splits}
        decoder = CatalogDecoder(self.catalog, {'training': self.config['gram_training']})
        return model, datasets, collators, decoder

    def smoke(self):
        self.stage = 'smoke'
        model, datasets, collators, decoder = self.gram()
        rows = [datasets['train'][i] for i in range(self.config['gram_training']['microbatch'])]
        batch = collators['train'](rows)
        model.set_backbone_frozen(True)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        original = model.gram.shared.weight.detach().cpu().clone()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**memory_batch(batch, self.device, True)).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite GRAM smoke loss')
            loss.backward()
            if not model.memory_projection.weight.grad.abs().sum() > 0:
                raise AssertionError('No gradient reached evidence bridge')
            optimizer.step()
        torch.testing.assert_close(original, model.gram.shared.weight.detach().cpu(), atol=0, rtol=0)
        model.set_backbone_frozen(False)
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model(**memory_batch(batch, self.device, True)).loss
        loss.backward()
        if not model.gram.shared.weight.grad.abs().sum() > 0:
            raise AssertionError('No gradient reached joint GRAM backbone')
        model.zero_grad(set_to_none=True)
        model.eval()
        single = collators['validation']([datasets['validation'][0]])
        predictions = decoder.generate(model, memory_batch(single, self.device, False))
        if len(predictions[0][0]) != 50:
            raise AssertionError('Invalid beam width')
        result = {'status': 'passed', 'loss': float(loss.detach()),
                  'beam_size': len(predictions[0][0]),
                  'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                  'parameters': sum(p.numel() for p in model.parameters()), 'test_read': False}
        write_json(self.output/'smoke.json', result)
        self.event('complete', **result)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def pcrf(self, predictions):
        from experiment.phase9.eval_cf0_b3_beamfusion import score_item_head, standardize
        from experiment.phase9.eval_cf0_b4_reliability import rank_matrix
        lookup = {r['user_id']: r for r in self.validation.records}
        records = []
        for r in predictions:
            source = lookup[r['user_id']]
            items = r['ranked_item_ids']
            records.append({'user': r['user_id'], 'history': source['history'][-20:],
                'candidate_ids': items, 'seq': np.asarray(r['scores'], dtype=np.float64),
                'target_position': items.index(source['target']) if source['target'] in items else -1})
        score_item_head(records, ROOT/self.config['pcrf']['checkpoint'], 256)
        pop = np.asarray([self.frequencies[r['candidate_ids']] for r in records])
        ranks, _ = rank_matrix(
            np.stack([standardize(r['seq']) for r in records]),
            np.stack([standardize(r['cf']) for r in records]),
            np.stack([standardize(np.log1p(p)) for p in pop]),
            (pop[:, :10] <= self.config['pcrf']['q1']).mean(1),
            np.asarray([r['target_position'] for r in records]),
            tuple(self.config['pcrf'][k] for k in ('lambda', 'beta', 'gamma')))
        return ranks.tolist()

    @torch.no_grad()
    def evaluate_gram(self, model, dataset, collator, decoder, tag, subset=None):
        model.eval()
        cfg = self.config['gram_training']
        rows = dataset if subset is None else Subset(dataset, subset)
        loader = DataLoader(rows, batch_size=cfg['validation_microbatch'], collate_fn=collator,
                            shuffle=False, num_workers=cfg['num_workers'])
        predictions, ranks = [], []
        destination = self.output/'gram'/f'{tag}.jsonl'
        destination.parent.mkdir(exist_ok=True)
        with destination.open('w') as f:
            for batch in loader:
                generated = decoder.generate(model, memory_batch(batch, self.device, False))
                for uid, target, (items, scores) in zip(batch['user_ids'], batch['target_item_ids'].tolist(), generated):
                    row = {'user_id': uid, 'gold_item_id': target, 'ranked_item_ids': items,
                           'scores': scores, 'split': 'validation', 'scope': 'full' if subset is None else 'trend'}
                    f.write(json.dumps(row)+'\n')
                    predictions.append(row)
                    ranks.append(items.index(target)+1 if target in items else 51)
                if len(predictions) % 100 == 0:
                    self.event('evaluating', tag=tag, seen=len(predictions), total=len(rows))
        reranked = self.pcrf(predictions)
        baseline = [int(self.reference[r['user_id']]['baseline_rank']) for r in predictions]
        baseline_pcrf = [int(self.reference[r['user_id']]['pcrf_rank']) for r in predictions]
        result = {'gram_memory': metrics(ranks), 'gram_memory_pcrf': metrics(reranked),
                  'original_gram': metrics(baseline), 'original_gram_pcrf': metrics(baseline_pcrf),
                  'new_top10_outside_original50': sum(a <= 10 and b > 50 for a, b in zip(ranks, baseline)),
                  'scope': 'full' if subset is None else 'trend', 'test_read': False}
        write_json(destination.with_suffix('.metrics.json'), result)
        self.event('validation', tag=tag, **result)
        return result

    def train_gram(self):
        self.stage = 'gram'
        if json.loads((self.output/'smoke.json').read_text())['status'] != 'passed':
            raise RuntimeError('A successful GRAM smoke is required')
        destination = self.output/'gram'
        destination.mkdir(exist_ok=True)
        if (destination/'latest.pt').exists():
            raise FileExistsError('GRAM training already has a checkpoint; do not silently restart')
        model, datasets, collators, decoder = self.gram()
        cfg = self.config['gram_training']
        epochs = cfg['warmup_epochs'] + cfg['joint_epochs']
        updates_per_epoch = math.ceil(len(datasets['train'])/cfg['effective_batch_size'])
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if not n.startswith('gram.')], 'lr': cfg['new_learning_rate']},
            {'params': list(model.gram.parameters()), 'lr': cfg['backbone_learning_rate']}], weight_decay=cfg['weight_decay'], eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=int(updates_per_epoch*epochs*cfg['warmup_fraction']),
            num_training_steps=updates_per_epoch*epochs)
        cohort = sorted(range(len(datasets['validation'])), key=lambda i: hashlib.sha256(
            ('s18_raserec_trend:'+datasets['validation'].records[i]['user_id']).encode()).hexdigest())[:cfg['trend_users']]
        write_json(destination/'trend_cohort.json', [datasets['validation'].records[i]['user_id'] for i in cohort])
        best, best_pcrf = -math.inf, -math.inf
        for epoch in range(1, epochs+1):
            model.set_backbone_frozen(epoch <= cfg['warmup_epochs'])
            model.train()
            loader = DataLoader(datasets['train'], batch_size=cfg['microbatch'], shuffle=True,
                                collate_fn=collators['train'], num_workers=cfg['num_workers'])
            optimizer.zero_grad(set_to_none=True)
            seen, in_update, total_loss = 0, 0, 0.0
            update_size = min(cfg['effective_batch_size'], len(datasets['train']))
            for batch in loader:
                size = len(batch['user_ids'])
                loss = model(**memory_batch(batch, self.device, True)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite GRAM loss')
                (loss * size/update_size).backward()
                seen += size
                in_update += size
                total_loss += float(loss.detach())*size
                if in_update == update_size:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite GRAM gradients')
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    in_update = 0
                    update_size = min(cfg['effective_batch_size'], len(datasets['train'])-seen)
                if seen % 1024 == 0 or seen == len(datasets['train']):
                    self.event('training', epoch=epoch, seen=seen, total=len(datasets['train']),
                               loss=total_loss/seen, learning_rates=scheduler.get_last_lr(),
                               frozen=model.backbone_frozen)
            if in_update:
                raise AssertionError('Incomplete final gradient accumulation')
            save(destination/'latest.pt', {'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                 'numpy_rng': np.random.get_state()})
            result = self.evaluate_gram(model, datasets['validation'], collators['validation'], decoder,
                                        f'epoch_{epoch:02d}_trend', cohort)
            # Warm-up is observed but does not replace a jointly trained checkpoint.
            if epoch > cfg['warmup_epochs']:
                score = result['gram_memory']['ndcg@10']
                score_pcrf = result['gram_memory_pcrf']['ndcg@10']
                if score > best:
                    best = score
                    save(destination/'best.pt', {'model': cpu_state(model), 'epoch': epoch, 'trend': result})
                if score_pcrf > best_pcrf:
                    best_pcrf = score_pcrf
                    save(destination/'best_pcrf.pt', {'model': cpu_state(model), 'epoch': epoch, 'trend': result})
        for label in ('best', 'best_pcrf'):
            selected = torch.load(destination/f'{label}.pt', map_location='cpu')
            model.load_state_dict(selected['model'], strict=True)
            self.evaluate_gram(model, datasets['validation'], collators['validation'], decoder, f'{label}_full')
        self.event('complete', epochs=epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase18/config/s18_raserec_gram_toys.json')
    parser.add_argument('--stage', choices=('all', 'pretrain', 'bank', 'reader', 'smoke', 'gram'), default='all')
    args = parser.parse_args()
    config_path = (ROOT/args.config).resolve()
    config = json.loads(config_path.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all' and (output/'manifest.json').exists():
        raise FileExistsError('Existing run; use an explicit uncompleted stage or a new output')
    seed_everything(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    if free/2**20 < config['minimum_free_mib']:
        raise RuntimeError(f'GPU has only {free/2**20:.0f} MiB free; no training was started')
    torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'], 0)
    def deadline(signum, frame):
        raise TimeoutError('Fixed experiment wall-clock budget reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config['max_wall_seconds'])
    run = Run(config, config_path, output, torch.device('cuda:0'))
    if not (output/'manifest.json').exists():
        run.snapshot()
    try:
        stages = ['pretrain', 'bank', 'reader', 'smoke', 'gram'] if args.stage == 'all' else [args.stage]
        for stage in stages:
            getattr(run, {'gram': 'train_gram'}.get(stage, stage))()
        write_json(output/'status.json', {'state': 'complete', 'stages': stages, 'pid': os.getpid(),
                   'updated_at': time.time(), 'elapsed_seconds': time.time()-run.started, 'test_read': False})
    except BaseException as error:
        write_json(output/'status.json', {'state': 'failed', 'stage': run.stage, 'error': repr(error),
                   'pid': os.getpid(), 'updated_at': time.time(), 'test_read': False})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
