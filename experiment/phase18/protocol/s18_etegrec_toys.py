"""Single-domain ETEGRec screen: profile, RQ pretrain, alternating train, finetune."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiment.phase18.core import etegrec_toys_adapter as a
from experiment.phase18.analysis.screen_saved_reference import metrics as rank_metrics


class BudgetEnded(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, directory, limit, prior_seconds=0):
        self.directory = Path(directory)
        self.started = time.monotonic()
        self.limit = limit
        self.prior_seconds = prior_seconds
        self.last_event = 0
        self.state = {'state': 'STARTING', 'pid': os.getpid(), 'started_at': now()}

    @property
    def elapsed(self):
        return self.prior_seconds + time.monotonic() - self.started

    def update(self, **kwargs):
        self.state.update(kwargs, updated_at=now(), elapsed_seconds=self.elapsed)
        a.write_json(self.directory / 'status.json', self.state)

    def event(self, **kwargs):
        row = dict(time=now(), elapsed_seconds=self.elapsed, **kwargs)
        with (self.directory / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        self.last_event = time.monotonic()

    def check(self):
        if self.elapsed >= self.limit:
            raise BudgetEnded('12 GPU-hour budget exhausted; no automatic extension')


def loader(rows, batch, shuffle=False):
    return DataLoader(rows, batch_size=batch, shuffle=shuffle, collate_fn=a.collate,
                      num_workers=0, pin_memory=True)


def setup_device(config):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid']:
        raise ValueError('CUDA visibility does not match selected GPU UUID')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    free, total = torch.cuda.mem_get_info()
    cap = min(22 * 2 ** 30, free - 2 * 2 ** 30)
    if cap < 8 * 2 ** 30:
        raise RuntimeError('Insufficient free memory for the profiled batch')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return {'gpu_uuid': config['gpu_uuid'], 'device_name': torch.cuda.get_device_name(),
            'free_bytes_before': free, 'allocator_cap_bytes': cap, 'torch': torch.__version__}


def snapshot(config, output):
    output.mkdir(parents=True, exist_ok=False)
    a.write_json(output / 'config.json', config)
    paths = ['experiment/phase18/core/etegrec_adapter.py',
             'experiment/phase18/core/etegrec_toys_adapter.py',
             'experiment/phase18/protocol/s18_etegrec_toys.py',
             'experiment/phase18/config/s18_etegrec_toys.json',
             'experiment/phase18/run_stage18_etegrec_toys.sh']
    hashes = {}
    for path in paths:
        dest = output / 'sources' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.ROOT / path, dest)
        hashes[path] = a.digest(dest)
    a.write_json(output / 'local_sources.json', hashes)
    shutil.copy2(a.ROOT / config['source_manifest'], output / 'author_source_manifest.json')


def configure_optimizers(config, rec, rq, steps):
    opts = {'rec': torch.optim.AdamW(rec.parameters(), lr=config['lr_rec'], weight_decay=config['weight_decay']),
            'id': torch.optim.AdamW(rq.parameters(), lr=config['lr_id'], weight_decay=config['weight_decay'])}
    schedules = {key: a.scheduler(opts[key], steps * (config['joint_epochs'] // 2), config[f'warmup_{key}'])
                 for key in opts}
    return opts, schedules


def train_pass(config, rec, rq, codes, batches, opt, sched, phase, epoch, log, counters):
    a.set_phase(rec, rq, phase)
    sums, n, started = {}, 0, time.monotonic()
    for index, batch in enumerate(batches):
        log.check()
        opt.zero_grad(set_to_none=True)
        loss, parts = a.losses(rec, rq, codes, batch, phase,
                               alignment=epoch >= config['alignment_start_epoch'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {phase} loss')
        loss.backward()
        params = rq.parameters() if phase == 'id' else rec.parameters()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()
        counters[phase] = counters.get(phase, 0) + 1
        count = len(batch['targets'])
        n += count
        for k, v in dict(total=loss, **parts).items():
            sums[k] = sums.get(k, 0.0) + float(v.detach()) * count
        if time.monotonic() - log.last_event > 20:
            log.update(state='TRAINING', phase=phase, epoch=epoch, batches=index + 1,
                       total_batches=len(batches), optimizer_steps=dict(counters), lr=opt.param_groups[0]['lr'])
            log.event(phase=phase, epoch=epoch, samples=n, loss=sums['total'] / n,
                      optimizer_steps=dict(counters), lr=opt.param_groups[0]['lr'])
    return dict(losses={k: v / n for k, v in sums.items()}, samples=n,
                seconds=time.monotonic() - started, lr=opt.param_groups[0]['lr'])


@torch.no_grad()
def evaluate(config, rec, codes, rows, references, output, scope, label, log):
    started = time.monotonic()
    trie = a.CatalogTrie(codes, rec.code_number)
    path = output / f'{scope}_{label}.jsonl'
    partial = path.with_suffix('.partial.jsonl')
    ranks = []
    with a.isolated_rng(), partial.open('w') as f:
        for batch in loader(rows, config['eval_batch_size']):
            log.check()
            items, scores = a.generate_items(rec, codes, trie, batch, config['beam'])
            for user, target, ids, values in zip(batch['users'], batch['targets'].tolist(), items.tolist(), scores.tolist()):
                if len(set(ids)) != config['beam']:
                    raise ValueError('Duplicate generated items')
                rank = ids.index(target) + 1 if target in ids else 51
                ranks.append(rank)
                f.write(json.dumps({'user_id': user, 'gold_item_id': target, 'ranked_item_ids': ids,
                                    'scores': values, 'rank': rank, 'scope': scope, 'split': 'validation'}) + '\n')
            if time.monotonic() - log.last_event > 20:
                elapsed = time.monotonic() - started
                log.update(state='VALIDATING', validation_scope=scope, label=label, evaluated=len(ranks),
                           validation_total=len(rows), validation_eta_seconds=elapsed / len(ranks) * (len(rows) - len(ranks)))
                log.event(phase='validation', scope=scope, evaluated=len(ranks), total=len(rows))
    partial.replace(path)
    metrics = rank_metrics(ranks)
    reference = references['matched_subset' if scope == 'trend' else 'historical_full']
    value = {'scope': scope, 'label': label, 'examples': len(ranks), 'metrics': metrics,
             'historical_gram': reference['gram'], 'historical_gram_pcrf': reference['gram_pcrf'],
             'delta': {k: v - reference['gram'][k] for k, v in metrics.items()},
             'delta_vs_gram_pcrf': {k: v - reference['gram_pcrf'][k] for k, v in metrics.items()},
             'predictions': str(path.relative_to(a.ROOT)), 'predictions_sha256': a.digest(path),
             'seconds': time.monotonic() - started, 'test_read': False}
    a.write_json(path.with_suffix('.json'), value)
    log.event(phase='validation_complete', **value)
    return value


def plateau(score, state, enabled, min_delta):
    """Best checkpoint selection is separate from meaningful-improvement patience."""
    if not enabled:
        state.update(anchor=max(score, state.get('anchor', -math.inf)), stale=0, active=False)
    elif not state.get('active', False):
        state.update(anchor=max(score, state.get('anchor', -math.inf)), stale=0, active=True)
    elif score > state['anchor'] + min_delta:
        state.update(anchor=score, stale=0)
    else:
        state['stale'] += 1
    return state['stale']


def pretrain(config, rec, rq, train_items, output, log, resume=None):
    vectors = rec.semantic_embedding.weight[torch.tensor(train_items, device=rec.device)].detach()
    steps = math.ceil(len(vectors) / config['rq_batch_size'])
    opt = torch.optim.AdamW(rq.parameters(), lr=config['rq_lr'], weight_decay=config['rq_weight_decay'])
    sched = a.scheduler(opt, steps * config['rq_epochs'], kind='linear')
    progress = {'phase': 'rq_pretrain', 'epoch': 0, 'steps': 0, 'best': math.inf,
                'anchor': math.inf, 'stale': 0, 'active': False}
    if resume:
        _, progress = a.load_bundle(resume, rec, rq, {'rq': opt}, {'rq': sched})
    stop = progress['stale'] >= config['rq_patience']
    for epoch in range(progress['epoch'] + 1, config['rq_epochs'] + 1):
        a.set_phase(rec, rq, 'id')
        order = torch.randperm(len(vectors), device=rec.device)
        started, total_loss = time.monotonic(), 0.0
        for indices in order.split(config['rq_batch_size']):
            log.check()
            x = vectors[indices]
            opt.zero_grad(set_to_none=True)
            recon, quant, *_ = rq(x)
            loss = F.mse_loss(recon, x) + rq.quant_loss_weight * quant
            if not torch.isfinite(loss):
                raise FloatingPointError('RQ pretraining nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rq.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            total_loss += float(loss.detach()) * len(x)
            progress['steps'] += 1
        progress['epoch'] = epoch
        stop = False
        if epoch == 1 or epoch % config['rq_eval_interval'] == 0:
            with torch.no_grad(), a.isolated_rng():
                rq.eval()
                mse = sum(float(F.mse_loss(rq(x)[0], x)) * len(x)
                          for x in vectors.split(config['rq_batch_size'])) / len(vectors)
            if mse < progress['best']:
                progress['best'] = mse
                a.save_bundle(output / 'rq_best.pt', rec, rq, None, {'rq': opt}, {'rq': sched}, progress)
            if epoch >= config['rq_min_epochs']:
                if not progress['active'] or mse < progress['anchor'] * (1 - config['rq_min_relative_delta']):
                    progress.update(anchor=mse, stale=0, active=True)
                else:
                    progress['stale'] += 1
                stop = progress['stale'] >= config['rq_patience']
            log.event(phase='rq_pretrain', epoch=epoch, mse=mse, best=progress['best'],
                      stale=progress['stale'], steps=progress['steps'], lr=opt.param_groups[0]['lr'])
        a.save_bundle(output / 'last.pt', rec, rq, None, {'rq': opt}, {'rq': sched}, progress)
        if time.monotonic() - log.last_event > 20:
            log.update(state='TRAINING', phase='rq_pretrain', epoch=epoch, rq_steps=progress['steps'],
                       epoch_seconds=time.monotonic() - started, lr=opt.param_groups[0]['lr'])
            log.event(phase='rq_pretrain', epoch=epoch, loss=total_loss / len(vectors), steps=progress['steps'])
        if stop:
            break
    a.write_json(output / 'rq_result.json', dict(progress, stopping_reason='plateau' if stop else 'epoch_limit'))
    # Selected pretrained weights, but retain the current RNG for joint training.
    a.load_bundle(output / 'rq_best.pt', rec, rq, restore_random=False)


def paired_report(predictions, config):
    import csv
    ref = json.loads((a.ROOT / config['input_review']).read_text())
    reference_path = next(p for p in ref['files'] if p.endswith('/per_user.tsv'))
    with (a.ROOT / reference_path).open() as f:
        baseline = {r['user_id']: r for r in csv.DictReader(f, delimiter='\t')}
    rows = [json.loads(l) for l in predictions.read_text().splitlines()]
    if set(r['user_id'] for r in rows) != set(baseline):
        raise ValueError('Paired user population mismatch')
    ranks = np.array([r['rank'] for r in rows])
    old = np.array([int(baseline[r['user_id']]['baseline_rank']) for r in rows])
    ndcg = lambda x: np.where(x <= 10, 1 / np.log2(x + 1), 0)
    differences = np.stack([ndcg(ranks) - ndcg(old), (ranks <= 10).astype(float) - (old <= 10)])
    rng = np.random.default_rng(2023)
    bootstrap = np.empty((2000, 2))
    for i in range(2000):
        indices = rng.integers(0, len(rows), len(rows))
        bootstrap[i] = differences[:, indices].mean(axis=1)
    return {'ndcg10_delta': float(differences[0].mean()), 'hit10_net_users': int(differences[1].sum()),
            'bootstrap_replicates': 2000, 'ndcg10_ci95': np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
            'hit10_ci95': np.quantile(bootstrap[:, 1], [.025, .975]).tolist(),
            'interpretation': 'Exploratory paired-user uncertainty; not seed robustness or independent confirmation.'}


def run(config, config_path, resume=False):
    output = a.ROOT / config['output']
    profile_path = a.ROOT / config['profile_output'] / 'profile.json'
    profile = json.loads(profile_path.read_text())
    if profile['config_sha256'] != a.digest(config_path) or not profile['budget_admitted']:
        raise ValueError('A passing profile for this exact config is required')
    for path, sha in profile['local_source_sha256'].items():
        if a.digest(a.ROOT / path) != sha:
            raise ValueError(f'Profiled code changed: {path}')
    if resume:
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration drift')
    else:
        snapshot(config, output)
    prior = profile['seconds']
    if resume:
        prior += json.loads((output / 'status.json').read_text()).get('training_elapsed_seconds', 0)
    log = RunLog(output, config['total_gpu_budget_seconds'], prior)
    try:
        hardware = setup_device(config)
        a.seed_all(config['seed'])
        embeddings, rows, train_items, names = a.load_inputs(config)
        rec, rq = a.build_models(config, embeddings, 'cuda:0')
        references = json.loads((a.ROOT / config['references']).read_text())
        log.update(state='TRAINING', gpu_uuid=config['gpu_uuid'], hardware=hardware,
                   result_path=str((output / 'result.json').relative_to(a.ROOT)), test_read=False)
        log.event(phase='start', config_sha256=a.digest(config_path), profile=profile['estimated_max_seconds'],
                  hardware=hardware, train_examples=len(rows['train']))
        resume_path = output / 'last.pt' if resume else None
        saved_progress = None
        if resume:
            saved_progress = torch.load(resume_path, map_location='cpu', weights_only=False)['progress']
        if not resume or saved_progress['phase'] == 'rq_pretrain':
            pretrain(config, rec, rq, train_items, output, log, resume_path)
            saved_progress = None
        batches = loader(rows['train'], config['batch_size'], True)
        opts, schedules = configure_optimizers(config, rec, rq, len(batches))
        progress = {'phase': 'joint', 'epoch': 0, 'counters': {'id': 0, 'rec': 0, 'finetune': 0},
                    'best_score': -math.inf, 'best_label': None, 'plateau': {}}
        codes = None
        if saved_progress is not None and saved_progress['phase'] == 'joint':
            codes, progress = a.load_bundle(resume_path, rec, rq, opts, schedules)
        if saved_progress is None or saved_progress['phase'] == 'joint':
            if codes is None:
                codes, code_info = a.make_codes(rec, rq)
                log.event(phase='initial_codes', **code_info)
            else:
                code_info = {'restored': True, 'catalog_items': len(codes) - 1}
            stop = progress['plateau'].get('stale', 0) >= config['joint_patience']
            for epoch in ([] if stop else range(progress['epoch'] + 1, config['joint_epochs'] + 1)):
                phase = 'id' if epoch % 2 else 'rec'
                result = train_pass(config, rec, rq, codes, batches, opts[phase], schedules[phase],
                                    phase, epoch, log, progress['counters'])
                if phase == 'id':
                    codes, code_info = a.make_codes(rec, rq)
                progress['epoch'] = epoch
                stop = False
                if epoch % config['joint_eval_interval'] == 0:
                    evaluation = evaluate(config, rec, codes, rows['trend'], references, output, 'trend', f'joint_{epoch:03d}', log)
                    score = evaluation['metrics']['ndcg@10']
                    if score > progress['best_score']:
                        progress.update(best_score=score, best_label=f'joint_{epoch:03d}')
                        a.save_bundle(output / 'joint_best.pt', rec, rq, codes, opts, schedules, progress)
                    stale = plateau(score, progress['plateau'], epoch >= config['joint_min_epochs'], config['min_delta'])
                    stop = stale >= config['joint_patience']
                a.save_bundle(output / 'last.pt', rec, rq, codes, opts, schedules, progress)
                log.update(state='TRAINING', phase='joint', epoch=epoch, optimizer_steps=progress['counters'],
                           best_score=progress['best_score'], best_label=progress['best_label'])
                log.event(phase='joint_epoch_complete', epoch=epoch, updated_component=phase, **result,
                          optimizer_steps=progress['counters'], code_info=code_info)
                if stop:
                    break
            a.write_json(output / 'joint_result.json', dict(progress, stopping_reason='plateau' if stop else 'epoch_limit'))
            codes, best = a.load_bundle(output / 'joint_best.pt', rec, rq, restore_random=False)
            # Keep the joint best eligible even if all finetuning checkpoints degrade.
            progress = {'phase': 'finetune', 'epoch': 0, 'counters': progress['counters'],
                        'best_score': best['best_score'], 'best_label': best['best_label'],
                        'selected_file': 'joint_best.pt', 'plateau': {}}
        else:
            progress = saved_progress
        finopt = torch.optim.AdamW(rec.parameters(), lr=config['lr_finetune'], weight_decay=config['weight_decay'])
        finsched = a.scheduler(finopt, len(batches) * config['finetune_epochs'])
        if saved_progress is not None and saved_progress['phase'] == 'finetune':
            codes, progress = a.load_bundle(resume_path, rec, rq, {'finetune': finopt}, {'finetune': finsched})
        stop = progress['plateau'].get('stale', 0) >= config['finetune_patience']
        for epoch in ([] if stop else range(progress['epoch'] + 1, config['finetune_epochs'] + 1)):
            result = train_pass(config, rec, rq, codes, batches, finopt, finsched, 'finetune', epoch, log, progress['counters'])
            progress['epoch'] = epoch
            stop = False
            if epoch % config['finetune_eval_interval'] == 0:
                evaluation = evaluate(config, rec, codes, rows['trend'], references, output, 'trend', f'finetune_{epoch:03d}', log)
                score = evaluation['metrics']['ndcg@10']
                if score > progress['best_score']:
                    progress.update(best_score=score, best_label=f'finetune_{epoch:03d}', selected_file='finetune_best.pt')
                    a.save_bundle(output / 'finetune_best.pt', rec, rq, codes, {'finetune': finopt}, {'finetune': finsched}, progress)
                stale = plateau(score, progress['plateau'], epoch >= config['finetune_min_epochs'], config['min_delta'])
                stop = stale >= config['finetune_patience']
            a.save_bundle(output / 'last.pt', rec, rq, codes, {'finetune': finopt}, {'finetune': finsched}, progress)
            log.update(state='TRAINING', phase='finetune', epoch=epoch, optimizer_steps=progress['counters'],
                       best_score=progress['best_score'], best_label=progress['best_label'])
            log.event(phase='finetune_epoch_complete', epoch=epoch, **result)
            if stop:
                break
        a.write_json(output / 'finetune_result.json', dict(progress, stopping_reason='plateau' if stop else 'epoch_limit'))
        codes, _ = a.load_bundle(output / progress['selected_file'], rec, rq, restore_random=False)
        full = evaluate(config, rec, codes, rows['validation'], references, output, 'full_validation', progress['best_label'], log)
        paired = paired_report(a.ROOT / full['predictions'], config)
        result = {'state': 'COMPLETED', 'selected': progress, 'full_validation': full, 'paired': paired,
                  'seconds_including_profile': log.elapsed, 'test_read': False,
                  'interpretation': config['scope'], 'automatic_additional_training': False}
        a.write_json(output / 'result.json', result)
        log.update(state='COMPLETED', phase='complete', full_validation=full,
                   training_elapsed_seconds=time.monotonic() - log.started)
    except BaseException as exc:
        state = 'BUDGET_EXHAUSTED' if isinstance(exc, BudgetEnded) else 'FAILED'
        log.update(state=state, error=str(exc), training_elapsed_seconds=time.monotonic() - log.started,
                   last_complete_checkpoint=str(output / 'last.pt'))
        log.event(phase=state, error=str(exc), traceback=traceback.format_exc())
        raise


def profile(config, config_path):
    output = a.ROOT / config['profile_output']
    snapshot(config, output)
    log = RunLog(output, min(1800, config['total_gpu_budget_seconds']), config.get('previous_profile_seconds', 0))
    try:
        hardware = setup_device(config)
        a.seed_all(config['seed'])
        embeddings, rows, train_items, _ = a.load_inputs(config)
        rec, rq = a.build_models(config, embeddings, 'cuda:0')
        train_ids = torch.tensor(train_items, device=rec.device)
        vectors = rec.semantic_embedding.weight[train_ids].detach()
        torch.cuda.reset_peak_memory_stats()
        a.set_phase(rec, rq, 'id')
        rqopt = torch.optim.AdamW(rq.parameters(), lr=config['rq_lr'], weight_decay=config['rq_weight_decay'])
        timings = {}
        rq_times = []
        for step in range(25):
            x = vectors[torch.randperm(len(vectors), device=rec.device)[:config['rq_batch_size']]]
            torch.cuda.synchronize()
            started = time.monotonic()
            rqopt.zero_grad(set_to_none=True)
            recon, quant, *_ = rq(x)
            loss = F.mse_loss(recon, x) + quant
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rq.parameters(), 1, error_if_nonfinite=True)
            rqopt.step()
            torch.cuda.synchronize()
            if step >= 5:
                rq_times.append(time.monotonic() - started)
        timings['rq_update'] = float(np.quantile(rq_times, .9))
        # A few timing steps are not a pretrained tokenizer. Allow bounded RQ-only
        # initialization before judging the real catalog contract in this smoke.
        init_passes = 0
        while True:
            try:
                codes, code_info = a.make_codes(rec, rq)
                break
            except ValueError as exc:
                if 'Code collision overflow' not in str(exc) or init_passes >= 2000:
                    raise
            a.set_phase(rec, rq, 'id')
            for _ in range(50):
                for indices in torch.randperm(len(vectors), device=rec.device).split(config['rq_batch_size']):
                    log.check()
                    x = vectors[indices]
                    rqopt.zero_grad(set_to_none=True)
                    recon, quant, *_ = rq(x)
                    loss = F.mse_loss(recon, x) + quant
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite RQ initialization')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rq.parameters(), 1, error_if_nonfinite=True)
                    rqopt.step()
                init_passes += 1
            log.update(state='PROFILING', phase='rq_initialization', initialization_passes=init_passes)
            log.event(phase='rq_initialization', passes=init_passes, loss=float(loss.detach()))
        batches = []
        for n, batch in enumerate(loader(rows['train'], config['batch_size'], True)):
            batches.append(batch)
            if n >= 7:
                break
        opts, schedules = configure_optimizers(config, rec, rq, math.ceil(len(rows['train']) / config['batch_size']))
        gradient_checks = {}
        a.set_phase(rec, rq, 'id')
        for name in ('kl', 'contrastive'):
            rq.zero_grad(set_to_none=True)
            _, parts = a.losses(rec, rq, codes, batches[0], 'id')
            parts[name].backward()
            norm = sum(float(p.grad.abs().sum()) for p in rq.encoder.parameters() if p.grad is not None)
            if not math.isfinite(norm) or norm <= 0:
                raise AssertionError(f'Alignment gradient absent: {name}')
            gradient_checks[name] = norm
        parameter_updates = {}
        for phase in ('id', 'rec', 'finetune'):
            a.set_phase(rec, rq, phase)
            optimizer = opts['id' if phase == 'id' else 'rec']
            schedule = schedules['id' if phase == 'id' else 'rec']
            probe_parameter = next(rq.encoder.parameters()) if phase == 'id' else rec.token_embeddings[0].weight
            parameter_before = probe_parameter.detach().clone()
            durations = []
            for i, batch in enumerate(batches):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.monotonic()
                loss, _ = a.losses(rec, rq, codes, batch, phase)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rq.parameters() if phase == 'id' else rec.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                torch.cuda.synchronize()
                if i >= 2:
                    durations.append(time.monotonic() - started)
                log.update(state='PROFILING', phase=phase, step=i + 1)
            timings[phase + '_update'] = float(np.quantile(durations, .9))
            parameter_updates[phase] = float((probe_parameter.detach() - parameter_before).abs().sum())
            if parameter_updates[phase] <= 0:
                raise AssertionError(f'No actual parameter update in {phase}')
            if phase == 'id':
                codes, code_info = a.make_codes(rec, rq)
        # GPU checkpoint/RNG/optimizer restoration and fixed-input prediction equality.
        a.set_phase(rec, rq, 'rec')
        rec.eval()
        with torch.no_grad():
            before = a.losses(rec, rq, codes, batches[0], 'rec')[1]['ce'].clone()
        a.save_bundle(output / 'restore_probe.pt', rec, rq, codes, opts, schedules, {'steps': 8})
        rng_before = a.rng_state()
        restored_codes, restored = a.load_bundle(output / 'restore_probe.pt', rec, rq, opts, schedules)
        with torch.no_grad():
            after = a.losses(rec, rq, restored_codes, batches[0], 'rec')[1]['ce']
        if not torch.equal(before, after) or not torch.equal(rng_before['torch'], a.rng_state()['torch']):
            raise AssertionError('GPU checkpoint replay mismatch')
        # Profile generation against all catalog codes using real validation histories.
        trie = a.CatalogTrie(codes, rec.code_number)
        durations = []
        for index, batch in enumerate(loader(rows['trend'][:8 * config['eval_batch_size']], config['eval_batch_size'])):
            torch.cuda.synchronize()
            started = time.monotonic()
            ids, scores = a.generate_items(rec, codes, trie, batch, config['beam'])
            torch.cuda.synchronize()
            if index >= 2:
                durations.append((time.monotonic() - started) / len(ids))
            if any(len(set(v)) != config['beam'] for v in ids.tolist()) or not torch.isfinite(scores).all():
                raise AssertionError('Profile generation contract failed')
        timings['validation_user'] = float(np.quantile(durations, .9))
        # Include full checkpoint writes, code refreshes and RQ monitoring overhead.
        torch.cuda.synchronize()
        started = time.monotonic()
        a.make_codes(rec, rq)
        torch.cuda.synchronize()
        timings['code_refresh'] = time.monotonic() - started
        started = time.monotonic()
        a.save_bundle(output / 'restore_probe.pt', rec, rq, codes, opts, schedules, {'steps': 8})
        timings['checkpoint'] = time.monotonic() - started
        steps = math.ceil(len(rows['train']) / config['batch_size'])
        rqsteps = math.ceil(len(train_items) / config['rq_batch_size'])
        j = config['joint_epochs'] // 2
        f = config['finetune_epochs']
        val_users = len(rows['trend']) * (config['joint_epochs'] // config['joint_eval_interval'] + f // config['finetune_eval_interval']) + len(rows['validation'])
        components = {'rq': config['rq_epochs'] * (rqsteps * timings['rq_update'] + timings['checkpoint']),
                      'joint': j * steps * (timings['id_update'] + timings['rec_update']) + j * timings['code_refresh'],
                      'finetune': f * steps * timings['finetune_update'],
                      'evaluation': val_users * timings['validation_user'],
                      'checkpoints': (config['joint_epochs'] + f) * timings['checkpoint'],
                      'rq_monitoring': (config['rq_epochs'] // config['rq_eval_interval']) * rqsteps * timings['rq_update']}
        estimated = config['budget_safety_factor'] * sum(components.values()) + log.elapsed
        result = {'state': 'PASSED', 'hardware': hardware, 'timings': timings, 'code_info': code_info,
                  'alignment_gradient_l1': gradient_checks, 'actual_parameter_update_l1': parameter_updates,
                  'gpu_checkpoint_replay': 'PASSED',
                  'config_sha256': a.digest(config_path),
                  'local_source_sha256': {p: a.digest(a.ROOT / p) for p in ('experiment/phase18/core/etegrec_adapter.py', 'experiment/phase18/core/etegrec_toys_adapter.py', 'experiment/phase18/protocol/s18_etegrec_toys.py', 'experiment/phase18/run_stage18_etegrec_toys.sh')},
                  'max_memory_allocated': torch.cuda.max_memory_allocated(), 'estimate_components': components,
                  'estimated_max_seconds': estimated, 'budget_admitted': estimated <= config['total_gpu_budget_seconds'],
                  'seconds': log.elapsed, 'test_read': False, 'formal_training_started': False,
                  'rq_initialization_passes': init_passes,
                  'torch_versions': {'torch': torch.__version__}, 'training_batch': config['batch_size']}
        a.write_json(output / 'profile.json', result)
        log.update(state='PASSED', budget_admitted=result['budget_admitted'], estimated_max_seconds=estimated)
        log.event(phase='profile_complete', **result)
    except BaseException as exc:
        log.update(state='FAILED', error=str(exc))
        log.event(phase='FAILED', error=str(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('profile', 'run'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    def timeout_handler(signum, frame):
        raise BudgetEnded(f'External timeout signal {signum}; last complete checkpoint retained')
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800 if args.mode == 'profile' else config['total_gpu_budget_seconds'])
    if args.mode == 'profile':
        profile(config, args.config)
    else:
        run(config, args.config, args.resume)


if __name__ == '__main__':
    main()
