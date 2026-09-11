"""Parallel S-DPO hard-negative adaptation; does not modify the live SPRec run."""
import argparse
import csv
import gc
import json
import math
import os
import random
import signal
import time
import traceback

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.protocol.s18_diff_gram import CatalogDecoder, seed_everything
from experiment.phase20.run_preference import ROOT, Run, Stopped
from experiment.phase20.sdpo_core import multi_logps, softmax_preference_loss, choose_hard_negatives


class SoftmaxRun(Run):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mining = dict(self.config['decoding'], beam_size=self.config['mining_beams'])
        self.mining_decoder = CatalogDecoder(self.catalog, {'training': mining})
        self.parent_reference_loaded = False

    @torch.no_grad()
    def make_pairs(self, batch):
        generated = self.mining_decoder.generate(self.model, {
            'input_ids': batch['item_text_ids'].to(self.device),
            'attention_mask': batch['item_text_masks'].to(self.device)})
        targets = batch['target_item_ids'].tolist()
        return [choose_hard_negatives(items, target, self.config['negative_count'])
                for target, (items, _) in zip(targets, generated)]

    @torch.no_grad()
    def mine_bank(self):
        self.stage, self.step = 'hard_negative_generation', 0
        self.model.eval()
        bank, previous = {}, time.time()
        destination = self.output/'train_preference_bank.jsonl'
        self.event('mining_started', total=len(self.train), negative_count=self.config['negative_count'])
        with destination.open('x') as f:
            for indices, batch in self.loader(self.train, list(range(len(self.train))), self.config['mining_microbatch']):
                self.check_stop()
                negatives = self.make_pairs(batch)
                rp, rn, _ = multi_logps(self.model, batch, negatives, self.decoder.paths, self.device)
                if not torch.isfinite(rp).all() or not torch.isfinite(rn).all():
                    raise FloatingPointError('Nonfinite frozen reference scores')
                for i, neg, cp, cn in zip(indices, negatives, rp.tolist(), rn.tolist()):
                    record = {'train_index': i, 'user_id': self.train.records[i]['user_id'],
                        'chosen': self.train.records[i]['target'], 'rejected': neg,
                        'reference_chosen': cp, 'reference_rejected': cn, 'split': 'train'}
                    bank[i] = record
                    f.write(json.dumps(record)+'\n')
                self.step += len(indices)
                if time.time()-previous >= 30:
                    self.event('mining_progress', examples=self.step, total=len(self.train))
                    f.flush()
                    previous = time.time()
        write_json(self.output/'preference_bank_manifest.json', {
            'rows': len(bank), 'negatives_per_row': self.config['negative_count'],
            'bank_sha256': sha256(destination), 'parent_sha256': self.data_config['parent_sha256'],
            'train_file_sha256': self.data_manifest['prepared_files']['train.jsonl'],
            'reference_policy': 'original_parent_fixed_for_all_epochs', 'dropout': False,
            'tf32': False, 'test_read': False})
        self.event('mining_complete', examples=len(bank), negatives=len(bank)*self.config['negative_count'])
        return bank

    def load_matched_parent(self):
        if self.parent_reference_loaded:
            return
        shared = ROOT/self.config['shared_parent_output']
        path = shared/'matched_parent_reference.json'
        self.stage = 'waiting_for_matched_parent_reference'
        while not path.exists():
            self.check_stop()
            status_path = shared/'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('state') in ('FAILED', 'STOPPED'):
                raise RuntimeError('Shared parent inference stopped before producing its reference')
            self.event('waiting_reference', dependency=str(path.relative_to(ROOT)))
            time.sleep(30)
        manifest = json.loads((shared/'manifest.json').read_text())
        if (manifest['data_manifest']['prepared_files'] != self.data_manifest['prepared_files']
                or manifest['data_manifest']['parent_sha256'] != self.data_config['parent_sha256']
                or manifest['config']['decoding'] != self.config['decoding']
                or manifest['precision'] != 'float32' or manifest['tf32'] is not False):
            raise AssertionError('Shared parent protocol differs')
        reference = json.loads(path.read_text())
        if set(reference) != set(self.reference):
            raise AssertionError('Shared parent users differ')
        self.reference = reference
        self.parent_reference_loaded = True
        write_json(self.output/'shared_parent_reference_manifest.json', {
            'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
            'producer_manifest_sha256': sha256(shared/'manifest.json'),
            'data_parent_and_decoding_matched': True})
        self.event('matched_parent_loaded', users=len(reference))

    def train_epoch(self, bank, indices=None):
        self.stage = 'sdpo'
        self.model.eval()  # deterministic policy log ratios; gradients remain enabled
        indices = list(range(len(self.train))) if indices is None else indices
        micro, effective = self.config['microbatch'], self.config['effective_batch_size']
        loader = self.loader(self.train, indices, micro, shuffle=True)
        accumulation = effective//micro
        previous, seen, epoch_step = time.time(), 0, 0
        totals = {'loss': 0., 'margin': 0., 'positive_nll': 0.}
        self.optimizer.zero_grad(set_to_none=True)
        self.event('training_started', examples=len(indices), optimizer_steps=math.ceil(len(indices)/effective))
        for batch_index, (indices_batch, batch) in enumerate(loader):
            records = [bank[i] for i in indices_batch]
            if any(r['chosen'] != self.train.records[i]['target'] or r['train_index'] != i
                   for i, r in zip(indices_batch, records)):
                raise AssertionError('Preference cache row mismatch')
            negatives = [r['rejected'] for r in records]
            rp = torch.tensor([r['reference_chosen'] for r in records], device=self.device)
            rn = torch.tensor([r['reference_rejected'] for r in records], device=self.device)
            cp, cn, nll = multi_logps(self.model, batch, negatives, self.decoder.paths, self.device)
            losses, margins = softmax_preference_loss(cp, cn, rp, rn, self.config['beta'])
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite S-DPO loss')
            group_start = (batch_index//accumulation)*effective
            group_size = min(effective, len(indices)-group_start)
            (loss*(len(indices_batch)/group_size)).backward()
            seen += len(indices_batch)
            for name, value in [('loss', loss), ('margin', margins.mean()), ('positive_nll', nll)]:
                totals[name] += float(value.detach())*len(indices_batch)
            if (batch_index+1) % accumulation == 0 or batch_index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                epoch_step += 1
                if self.step % 100 == 0 or self.stop_reason:
                    self.save_checkpoint('last_progress', include_optimizer=True)
                self.check_stop()
                if time.time()-previous >= 30:
                    self.event('training_progress', examples=seen, total=len(indices), epoch_step=epoch_step,
                        optimizer_steps=math.ceil(len(indices)/effective),
                        **{k: v/seen for k, v in totals.items()},
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                    previous = time.time()
        path = self.save_checkpoint(f'epoch{self.round}_sdpo', include_optimizer=True)
        self.event('training_complete', examples=seen, epoch_step=epoch_step,
                   checkpoint=str(path.relative_to(ROOT)), **{k: v/seen for k, v in totals.items()})

    def smoke(self):
        self.stage = 'sdpo_smoke'
        self.model.eval()
        longest = sorted(range(len(self.train)), key=lambda i: len(self.train.records[i]['history']), reverse=True)
        indices = longest[:self.config['microbatch']]
        batch = self.collator([self.train[i] for i in indices])
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        negatives = self.make_pairs(batch)
        with torch.no_grad():
            rp, rn, _ = multi_logps(self.model, batch, negatives, self.decoder.paths, self.device)
        cp, cn, nll = multi_logps(self.model, batch, negatives, self.decoder.paths, self.device)
        losses, margins = softmax_preference_loss(cp, cn, rp, rn, self.config['beta'])
        torch.testing.assert_close(losses, torch.full_like(losses, math.log(1+self.config['negative_count'])), atol=1e-5, rtol=0)
        losses.mean().backward()
        gradients = {}
        for prefix in ('encoder.', 'decoder.', 'shared.'):
            values = [p.grad for n, p in self.model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            if not values or any(not torch.isfinite(v).all() for v in values):
                raise AssertionError(f'Invalid gradients: {prefix}')
            gradients[prefix] = sum(float(v.abs().sum()) for v in values)
            if gradients[prefix] <= 0:
                raise AssertionError(f'Zero gradients: {prefix}')
        gradients['position_embedding'] = float(self.model.position_embedding.weight.grad.abs().sum())
        before = self.model.shared.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        changed = float((self.model.shared.weight.detach()-before).abs().max())
        if changed <= 0:
            raise AssertionError('Optimizer did not change policy weights')
        self.model.zero_grad(set_to_none=True)
        result = {'state': 'PASSED', 'negative_count': self.config['negative_count'],
            'initial_loss': float(losses.mean()), 'initial_margin_max_abs': float(margins.abs().max()),
            'gradient_abs_sum': gradients, 'max_parameter_change': changed,
            'longest_history': max(len(self.train.records[i]['history']) for i in indices),
            'microbatch': len(indices), 'seconds': time.monotonic()-started,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
            'test_read': False, 'scientific_efficacy_result': False}
        del optimizer, before, cp, cn, losses
        gc.collect()
        torch.cuda.empty_cache()
        # Restore untouched parent and run the actual cached-bank/epoch code on
        # eight train rows, with two optimizer updates and no warmup in smoke.
        del self.model
        self.model = self.new_model()
        complete = self.train
        self.train = type(complete)(complete.records[:8], self.catalog)
        bank = self.mine_bank()
        formal_effective = self.config['effective_batch_size']
        self.config['effective_batch_size'] = 4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 0, 2)
        self.round, self.step = 1, 0
        self.train_epoch(bank)
        self.config['effective_batch_size'] = formal_effective
        self.train = complete
        result['cached_reference_pipeline'] = {'train_rows': 8, 'optimizer_updates': 2}
        write_json(self.output/'smoke.json', result)
        self.event('smoke_passed', **result)

    def run(self):
        seed_everything(self.config['seed'])
        self.model = self.new_model()
        self.snapshot()
        write_json(self.output/'sdpo_source.json', {
            'source_manifest': 'experiment/phase20/sdpo_source_manifest.json',
            'sha256': sha256(ROOT/'experiment/phase20/sdpo_source_manifest.json'),
            'method': 'S-DPO objective with four fixed beam-mined hard negatives; GRAM full-catalog adaptation'})
        if self.mode == 'smoke':
            self.smoke()
            return
        smoke = json.loads((ROOT/self.config['smoke_output']/'smoke.json').read_text())
        if smoke['state'] != 'PASSED':
            raise AssertionError('S-DPO smoke has not passed')
        bank = self.mine_bank()
        self.step = 0
        steps = math.ceil(len(self.train)/self.config['effective_batch_size'])*self.config['epochs']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=.01)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, max(1, int(.05*steps)), steps)
        selection = sorted(random.Random(20260910).sample(range(len(self.validation)), self.config['selection_users']))
        write_json(self.output/'selection_indices.json', selection)
        for epoch in range(1, self.config['epochs']+1):
            self.round = epoch
            self.train_epoch(bank)
            self.load_matched_parent()
            result = self.evaluate(f'epoch{epoch}_sdpo', selection)
            self.outcomes.append(result)
            for name in self.best:
                if self.best[name] is None or result[name]['ndcg@10'] > self.best[name]['score']:
                    self.best[name] = {'tag': result['tag'], 'score': result[name]['ndcg@10']}
            write_json(self.output/'selection.json', {'best': self.best, 'outcomes': self.outcomes})
        self.optimizer, self.scheduler = None, None
        gc.collect()
        torch.cuda.empty_cache()
        full = {}
        for tag in sorted({r['tag'] for r in self.best.values()}):
            checkpoint = torch.load(self.output/f'{tag}.pt', map_location='cpu')
            self.model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            full[tag] = self.evaluate(f'{tag}_full', list(range(len(self.validation))))
        raw, combo = full[self.best['raw']['tag']], full[self.best['pcrf']['tag']]
        a = raw['raw']['ndcg@10']-raw['original_gram']['ndcg@10']
        b = combo['pcrf']['ndcg@10']-combo['original_gram_pcrf']['ndcg@10']
        write_json(self.output/'summary.json', {'state': 'COMPLETED', 'best': self.best,
            'full_validation': full, 'raw_ndcg_delta_vs_gram': a,
            'combined_ndcg_delta_vs_original_pcrf': b, 'positive_signal': a > 0 or b > 0,
            'beauty_automatically_started': False, 'test_read': False,
            'scope': 'single-seed Toys exploration; not an independent confirmation or isolated SPRec ablation'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase20/sdpo_toys_v1.json')
    parser.add_argument('--mode', required=True, choices=['smoke', 'run'])
    args = parser.parse_args()
    config_path = ROOT/args.config
    config = json.loads(config_path.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    output = ROOT/config['smoke_output' if args.mode == 'smoke' else 'output']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Existing attempt; refusing overwrite or automatic retry')
    run = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(0)
        free, _ = torch.cuda.mem_get_info()
        if free/2**20 < config['minimum_free_mib']:
            raise RuntimeError(f'Insufficient available memory: {free/2**20:.0f}MiB')
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
        if config['effective_batch_size'] % config['microbatch']:
            raise ValueError('Invalid batch accumulation')
        run = SoftmaxRun(config, config_path, output, args.mode)
        def stop(signum, frame):
            run.stop_reason = f'SIGNAL_{signum}'
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, stop)
        signal.alarm(900 if args.mode == 'smoke' else config['max_wall_seconds'])
        run.run()
        signal.alarm(0)
        write_json(output/'status.json', {'state': 'COMPLETED', 'method': 'S-DPO-hard4',
            'mode': args.mode, 'dataset': config['dataset'], 'pid': os.getpid(),
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'elapsed_seconds': time.time()-run.started, 'test_read': False})
    except BaseException as error:
        write_json(output/'status.json', {'state': 'STOPPED' if isinstance(error, Stopped) else 'FAILED',
            'error': repr(error), 'traceback': traceback.format_exc(), 'pid': os.getpid(),
            'stage': run.stage if run else 'initializing', 'mode': args.mode,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'test_read': False})
        raise


if __name__ == '__main__':
    main()
