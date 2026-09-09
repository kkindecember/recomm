"""One Beauty ReaRec PRL candidate: profile admission, full-prefix train, validation."""
import argparse
import csv
from datetime import datetime, timezone
from functools import partial
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
from torch.utils.data import DataLoader

from experiment.phase18.core import rearec_adapter as a
from experiment.phase18.analysis.screen_saved_reference import metrics as rank_metrics

LOCAL_SOURCES = ('experiment/phase18/core/rearec_adapter.py',
                 'experiment/phase18/protocol/s18_rearec.py',
                 'experiment/phase18/run_stage18_rearec_beauty.sh')


class BudgetEnded(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, output, limit, prior=0):
        self.output, self.limit, self.prior = output, limit, prior
        self.started = time.monotonic()
        self.last = self.started
        self.state = dict(state='STARTING', pid=os.getpid(), started_at=now())

    @property
    def elapsed(self):
        return self.prior + time.monotonic() - self.started

    def update(self, **values):
        self.state.update(values, updated_at=now(), elapsed_seconds=self.elapsed)
        a.write_json(self.output / 'status.json', self.state)
        self.last = time.monotonic()

    def event(self, **values):
        row = dict(time=now(), elapsed_seconds=self.elapsed, **values)
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def check(self):
        if self.elapsed >= self.limit:
            raise BudgetEnded('Fixed GPU-time budget exhausted; last complete checkpoint retained')


def loader(config, rows, training=False):
    return DataLoader(rows, batch_size=config['batch_size' if training else 'eval_batch_size'],
                      shuffle=training, collate_fn=partial(a.collate, max_history=config['max_history']),
                      num_workers=0, pin_memory=True)


def setup_device(config):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid']:
        raise ValueError('GPU UUID visibility mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    free, total = torch.cuda.mem_get_info()
    cap = min(12 * 2 ** 30, free - 3 * 2 ** 30)
    if cap < 6 * 2 ** 30:
        raise RuntimeError('Insufficient GPU headroom for this profile')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return dict(gpu_uuid=config['gpu_uuid'], free_bytes_before=free, allocator_cap_bytes=cap,
                device_name=torch.cuda.get_device_name(), torch=torch.__version__)


def snapshot(config, config_path, output):
    output.mkdir(parents=True, exist_ok=False)
    a.write_json(output / 'config.json', config)
    paths = list(LOCAL_SOURCES) + [str(config_path), config['source_manifest'], config['input_review'],
                                'experiment/phase18/analysis/screen_saved_reference.py']
    hashes = {}
    for path in paths:
        source = a.ROOT / path
        destination = output / 'sources' / source.relative_to(a.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(source.relative_to(a.ROOT))] = a.digest(destination)
    a.write_json(output / 'local_sources.json', hashes)
    return hashes


def optimizer(config, model):
    return torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


def update(config, model, opt, batch, epoch):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, parts = model.loss(model(batch, epoch=epoch))
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite PRL loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
    opt.step()
    return {k: float(v.detach()) for k, v in dict(total=loss, **parts).items()}, float(norm)


@torch.no_grad()
def predictions(config, model, batch, reason_steps=None):
    model.eval()
    out = model(batch, reason_steps=reason_steps)
    scores = out['prediction'].float().log_softmax(-1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite catalog scores')
    order = scores.argsort(dim=-1, descending=True, stable=True)[:, :config['topk']]
    return (order + 1).tolist(), scores.gather(1, order).tolist()


def evaluate(config, model, rows, references, output, scope, epoch, log, reason_steps=None):
    started = time.monotonic()
    path = output / f'{scope}_epoch_{epoch:03d}.jsonl'
    temporary = path.with_suffix('.partial.jsonl')
    ranks = []
    log.update(state='VALIDATING', phase=scope, epoch=epoch, evaluated=0, validation_total=len(rows))
    with a.isolated_rng(), temporary.open('w') as f:
        for batch in loader(config, rows):
            log.check()
            ids, scores = predictions(config, model, batch, reason_steps)
            for user, target, items, values in zip(batch['users'], batch['targets'].tolist(), ids, scores):
                if len(set(items)) != config['topk'] or min(items) < 1:
                    raise ValueError('Invalid output catalog contract')
                rank = items.index(target) + 1 if target in items else 51
                ranks.append(rank)
                f.write(json.dumps(dict(user_id=user, gold_item_id=target, ranked_item_ids=items, scores=values,
                                        rank=rank, split='validation', scope=scope)) + '\n')
            if time.monotonic() - log.last > 15:
                log.update(evaluated=len(ranks))
                log.event(phase=scope, epoch=epoch, evaluated=len(ranks), total=len(rows))
    temporary.replace(path)
    metrics = rank_metrics(ranks)
    reference = references['matched_subset' if scope == 'trend' else 'historical_full']
    result = dict(scope=scope, epoch=epoch, examples=len(ranks), metrics=metrics,
                  reason_steps=config['model']['reason_steps'] if reason_steps is None else reason_steps,
                  predictions=str(path.relative_to(a.ROOT)), predictions_sha256=a.digest(path),
                  historical_gram=reference['gram'], historical_gram_pcrf=reference['gram_pcrf'],
                  delta_vs_gram={k: v - reference['gram'][k] for k, v in metrics.items()},
                  delta_vs_gram_pcrf={k: v - reference['gram_pcrf'][k] for k, v in metrics.items()},
                  seconds=time.monotonic() - started, test_read=False,
                  score_type='log_softmax over all legitimate catalog items; PAD excluded')
    a.write_json(path.with_suffix('.json'), result)
    log.event(phase='validation_complete', **result)
    return result


def paired_report(config, prediction_path):
    with (a.ROOT / config['reference_predictions']).open() as f:
        reference = {r['user_id']: r for r in csv.DictReader(f, delimiter='\t')}
    rows = [json.loads(l) for l in prediction_path.read_text().splitlines()]
    if len(rows) != len(reference) or {r['user_id'] for r in rows} != reference.keys():
        raise ValueError('Paired population mismatch')
    ranks = np.array([r['rank'] for r in rows])
    ndcg = lambda x: np.where(x <= 10, 1 / np.log2(x + 1), 0)
    result = {}
    for name, column in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
        other = np.array([int(reference[r['user_id']][column]) for r in rows])
        differences = np.stack([ndcg(ranks) - ndcg(other), (ranks <= 10).astype(float) - (other <= 10)])
        rng = np.random.default_rng(2023)
        bootstrap = np.empty((2000, 2))
        for i in range(2000):
            bootstrap[i] = differences[:, rng.integers(0, len(rows), len(rows))].mean(1)
        result[name] = dict(ndcg10_delta=float(differences[0].mean()),
                            hit10_net_users=int(differences[1].sum()),
                            ndcg10_ci95=np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
                            hit10_ci95=np.quantile(bootstrap[:, 1], [.025, .975]).tolist())
    return dict(comparisons=result, bootstrap_replicates=2000,
                interpretation='Exploratory paired-user uncertainty; checkpoint selected on validation subset; not seed robustness or independent confirmation.')


def plateau(score, state, enabled, min_delta):
    if not enabled:
        state.update(anchor=max(score, state.get('anchor', -math.inf)), stale=0, active=False)
    elif not state.get('active', False):
        state.update(anchor=max(score, state.get('anchor', -math.inf)), stale=0, active=True)
    elif score > state['anchor'] + min_delta:
        state.update(anchor=score, stale=0)
    else:
        state['stale'] += 1
    return state['stale']


def run(config, config_path, resume):
    output = a.ROOT / config['output']
    prof = json.loads((a.ROOT / config['profile_output'] / 'profile.json').read_text())
    if prof['state'] != 'PASSED' or not prof['budget_admitted'] or a.digest(config_path) != prof['config_sha256']:
        raise ValueError('An admitted profile of this exact configuration is required')
    for path, sha in prof['source_sha256'].items():
        if a.digest(a.ROOT / path) != sha:
            raise ValueError(f'Profiled code/input manifest changed: {path}')
    prior = prof['seconds']
    if resume:
        old = json.loads((output / 'status.json').read_text())
        if old['state'] == 'COMPLETED':
            raise ValueError('Completed run cannot be resumed')
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Resume config drift')
        prior = max(prior, old['elapsed_seconds'])
    else:
        snapshot(config, config_path, output)
    log = RunLog(output, config['total_gpu_budget_seconds'], prior)
    try:
        hardware = setup_device(config)
        a.seed_all(config['seed'])
        rows, names = a.load_inputs(config)
        model = a.ReaRec(config, len(names)).to('cuda:0')
        opt = optimizer(config, model)
        references = json.loads((a.ROOT / config['references']).read_text())
        batches = loader(config, rows['train'], True)
        progress = dict(epoch=0, optimizer_steps=0, best_score=-1.0, best_epoch=None, plateau={})
        if resume:
            progress = a.load_checkpoint(output / 'last.pt', model, opt)
        log.update(state='TRAINING', phase='prl', gpu_uuid=config['gpu_uuid'], hardware=hardware,
                   train_examples=len(rows['train']), total_epochs=config['epochs'],
                   result_path=config['output'] + '/result.json', test_read=False, **progress)
        log.event(phase='start', resume=resume, config_sha256=a.digest(config_path), **progress)
        stopped = progress['plateau'].get('stale', 0) >= config['patience']
        for epoch in ([] if stopped else range(progress['epoch'] + 1, config['epochs'] + 1)):
            started = time.monotonic()
            sums, samples = {}, 0
            for index, batch in enumerate(batches):
                log.check()
                losses, norm = update(config, model, opt, batch, epoch)
                progress['optimizer_steps'] += 1
                count = len(batch['targets'])
                samples += count
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + value * count
                if time.monotonic() - log.last > 15:
                    log.update(state='TRAINING', phase='prl', epoch=epoch, batches=index + 1,
                               total_batches=len(batches), samples=samples,
                               optimizer_steps=progress['optimizer_steps'], loss=sums['total'] / samples)
                    log.event(phase='prl', epoch=epoch, samples=samples, optimizer_steps=progress['optimizer_steps'],
                              loss=sums['total'] / samples, last_gradient_norm=norm)
            progress['epoch'] = epoch
            if epoch % config['eval_interval'] == 0:
                result = evaluate(config, model, rows['trend'], references, output, 'trend', epoch, log)
                score = result['metrics']['ndcg@10']
                if score > progress['best_score']:
                    progress.update(best_score=score, best_epoch=epoch)
                    a.save_checkpoint(output / 'best.pt', model, opt, progress)
                stale = plateau(score, progress['plateau'], epoch >= config['min_epochs'], config['min_delta'])
                stopped = stale >= config['patience']
            a.save_checkpoint(output / 'last.pt', model, opt, progress)
            log.update(state='TRAINING', phase='prl', **progress)
            log.event(phase='epoch_complete', epoch=epoch, samples=samples,
                      losses={k: v / samples for k, v in sums.items()}, seconds=time.monotonic() - started,
                      optimizer_steps=progress['optimizer_steps'], best_score=progress['best_score'], best_epoch=progress['best_epoch'])
            if stopped:
                break
        a.write_json(output / 'training_result.json', dict(progress, stopping_reason='plateau' if stopped else 'epoch_limit'))
        if progress['best_epoch'] is None:
            raise ValueError('No selected validation checkpoint')
        a.load_checkpoint(output / 'best.pt', model, restore_random=False)
        full = evaluate(config, model, rows['validation'], references, output, 'full_validation', progress['best_epoch'], log)
        zero = evaluate(config, model, rows['validation'], references, output, 'zero_step_full_validation', progress['best_epoch'], log, 0)
        log.update(state='ANALYZING', phase='paired_validation')
        paired = paired_report(config, a.ROOT / full['predictions'])
        result = dict(state='COMPLETED', selected_epoch=progress['best_epoch'], training=progress,
                      full_validation=full, zero_step_diagnostic=zero, paired=paired,
                      two_step_minus_zero_step={k: v - zero['metrics'][k] for k, v in full['metrics'].items()},
                      zero_step_interpretation='Same trained two-step checkpoint; inference diagnostic only, not a separately trained baseline.',
                      seconds_including_profile=log.elapsed, test_read=False, interpretation=config['scope'])
        a.write_json(output / 'result.json', result)
        log.update(state='COMPLETED', phase='complete', full_validation=full, selected_epoch=progress['best_epoch'])
        log.event(phase='complete', selected_epoch=progress['best_epoch'], metrics=full['metrics'])
    except BaseException as exc:
        log.update(state='BUDGET_EXHAUSTED' if isinstance(exc, BudgetEnded) else 'FAILED', error=str(exc))
        log.event(phase=log.state['state'], error=str(exc), traceback=traceback.format_exc())
        raise


def profile(config, config_path):
    output = a.ROOT / config['profile_output']
    hashes = snapshot(config, config_path, output)
    log = RunLog(output, 1800, config.get('previous_profile_seconds', 0))
    try:
        hardware = setup_device(config)
        a.seed_all(config['seed'])
        rows, names = a.load_inputs(config)
        model = a.ReaRec(config, len(names)).to('cuda:0')
        opt = optimizer(config, model)
        batches = iter(loader(config, rows['train'], True))
        torch.cuda.reset_peak_memory_stats()
        timings = []
        probe = model.model.reason_pos_emb.weight.detach().clone()
        gradients = {}
        for index in range(16):
            log.check()
            torch.cuda.synchronize()
            start = time.monotonic()
            batch = next(batches)
            losses, norm = update(config, model, opt, batch, 1 if index < 2 else 3)
            torch.cuda.synchronize()
            if index >= 4:
                timings.append(time.monotonic() - start)
            log.update(state='PROFILING', phase='prl', step=index + 1, losses=losses)
        actual_update = float((model.model.reason_pos_emb.weight.detach() - probe).abs().sum())
        if actual_update <= 0:
            raise AssertionError('No reasoning-position update')
        model.train()
        for part in ('ce', 'progressive', 'contrastive'):
            opt.zero_grad(set_to_none=True)
            _, parts = model.loss(model(batch, epoch=3))
            parts[part].backward()
            gradient = model.model.reason_pos_emb.weight.grad
            gradients[part] = 0.0 if gradient is None else float(gradient.abs().sum())
            if not math.isfinite(gradients[part]):
                raise AssertionError(f'Nonfinite {part} gradient')
        if gradients['ce'] <= 0:
            raise AssertionError('Final prediction has no reasoning gradient')
        # Exact Adam/RNG replay across a real noisy update on this GPU.
        a.save_checkpoint(output / 'restore_probe.pt', model, opt, dict(epoch=3, optimizer_steps=16))
        update(config, model, opt, batch, 3)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        a.load_checkpoint(output / 'restore_probe.pt', model, opt)
        update(config, model, opt, batch, 3)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in expected.items()):
            raise AssertionError('GPU Adam/RNG replay differs')
        del expected
        evaluation_times = []
        val_batches = iter(loader(config, rows['trend']))
        for index in range(8):
            log.check()
            torch.cuda.synchronize()
            start = time.monotonic()
            val = next(val_batches)
            ids, scores = predictions(config, model, val)
            torch.cuda.synchronize()
            if index >= 2:
                evaluation_times.append((time.monotonic() - start) / len(ids))
            if any(len(set(x)) != config['topk'] or min(x) < 1 or max(x) >= len(names) for x in ids):
                raise AssertionError('Invalid dense catalog ranking')
        # Zero-step is a saved-checkpoint diagnostic, never a training branch.
        predictions(config, model, val, reason_steps=0)
        start = time.monotonic()
        a.save_checkpoint(output / 'restore_probe.pt', model, opt, dict(epoch=3, optimizer_steps=17))
        checkpoint_seconds = time.monotonic() - start
        train_p90, eval_p90 = float(np.quantile(timings, .9)), float(np.quantile(evaluation_times, .9))
        steps = math.ceil(len(rows['train']) / config['batch_size'])
        validations = config['epochs'] // config['eval_interval']
        components = dict(training=config['epochs'] * steps * train_p90,
                          validation=(validations * len(rows['trend']) + 2 * len(rows['validation'])) * eval_p90,
                          checkpoints=(config['epochs'] + validations) * checkpoint_seconds)
        estimate = config['budget_safety_factor'] * sum(components.values()) + log.elapsed
        result = dict(state='PASSED', budget_admitted=estimate <= config['total_gpu_budget_seconds'],
                      config_sha256=a.digest(config_path), source_sha256=hashes, hardware=hardware,
                      max_memory_allocated=torch.cuda.max_memory_allocated(), train_update_p90=train_p90,
                      validation_user_p90=eval_p90, checkpoint_seconds=checkpoint_seconds,
                      steps_per_epoch=steps, estimated_max_seconds=estimate, estimate_components=components,
                      actual_reason_position_update_l1=actual_update, reasoning_gradient_l1=gradients,
                      gpu_checkpoint_replay='PASSED', warmup_no_noise='PASSED', dense_top50='PASSED',
                      seconds=log.elapsed, test_read=False, formal_training_started=False)
        a.write_json(output / 'profile.json', result)
        log.update(state='PASSED', budget_admitted=result['budget_admitted'], estimated_max_seconds=estimate)
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
        raise BudgetEnded(f'External timeout signal {signum}')
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800 if args.mode == 'profile' else config['total_gpu_budget_seconds'])
    if args.mode == 'profile':
        profile(config, args.config)
    else:
        run(config, args.config, args.resume)


if __name__ == '__main__':
    main()
