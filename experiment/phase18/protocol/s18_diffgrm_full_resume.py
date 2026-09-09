"""Resume native DiffGRM's original optimizer schedule with sparse validation."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import torch
from torch.utils.data import Subset
from transformers import get_cosine_schedule_with_warmup

from experiment.phase18.analysis.screen_saved_reference import dataset_report
from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.screen_budget import cohort_indices, optimizer_steps
from experiment.phase18.protocol.s18_screen_resume import (
    ROOT, Backend, Run, alive, evaluate, load_checkpoint, now, restore_probe, save_checkpoint,
)


def restore_schedule(optimizer, source, config, per_epoch):
    """LambdaLR construction changes LR: restore Adam again after constructing it."""
    steps = optimizer_steps(source['optimizer'])
    if steps != int(source['epoch']) * per_epoch or 'scheduler' not in source:
        raise ValueError('Expected an original complete-epoch optimizer/scheduler checkpoint')
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config['overrides']['warmup_steps'], per_epoch * config['epochs'])
    scheduler.load_state_dict(source['scheduler'])
    optimizer.load_state_dict(source['optimizer'])
    expected = [base * function(steps) for base, function in
                zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    actual = [group['lr'] for group in optimizer.param_groups]
    if scheduler.last_epoch != steps or len(actual) != len(expected) or any(
            not math.isclose(a, b, rel_tol=0, abs_tol=1e-14)
            for a, b in zip(actual, expected)):
        raise ValueError('Original optimizer LR/scheduler position mismatch')
    if actual != scheduler.get_last_lr():
        raise ValueError('Saved scheduler and Adam learning rates disagree')
    return scheduler


def stopping_due(epoch, minimum_epoch, bad_checks, patience):
    return epoch >= minimum_epoch and bad_checks >= patience


def matched_evaluate(backend, dataset, run, stage_epoch, full):
    # DataLoader and generation may consume RNG. Cheap monitoring must not consume
    # the training RNG stream; all checkpoint RNG states remain training states.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        result = evaluate(backend, dataset, run, stage_epoch, full)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    references = run.references['historical_full' if full else 'matched_subset']
    result.update(epoch=backend.inherited_epoch + stage_epoch,
                  historical_gram=references['gram'], historical_gram_pcrf=references['gram_pcrf'],
                  delta={k: value - references['gram'][k] for k, value in result['metrics'].items()},
                  delta_vs_gram_pcrf={k: value - references['gram_pcrf'][k]
                                     for k, value in result['metrics'].items()},
                  interpretation='Exploratory comparison with historical ranks on identical validation users')
    write_json(run.directory / f'{result["scope"]}_stage_epoch_{stage_epoch:02d}.json', result)
    run.context['latest_full_validation' if full else 'latest_trend_validation'] = result
    run.event('matched_validation_completed', **result)
    return result


def save_training(path, backend, run, scheduler, epoch, steps, **extra):
    save_checkpoint(path, backend, epoch=epoch, stage_epoch=epoch-backend.inherited_epoch,
                    optimizer_steps=steps, stage_optimizer_steps=steps-backend.initial_steps,
                    scheduler=scheduler.state_dict(), config=backend.config,
                    revision_config=run.spec, **extra)


def train_original(backend, run, scheduler, prior_best=None):
    indices = cohort_indices(backend.records, run.spec['dataset'], backend.config['seed'], run.spec['trend_users'])
    previous = json.loads((run.root / 'screen_r1/trend_cohort.json').read_text())
    users = [str(backend.records[i]['user_id']) for i in indices]
    if users != previous['user_ids']:
        raise ValueError('Original-schedule continuation must keep the existing trend cohort')
    write_json(run.directory / 'trend_cohort.json', previous)
    cohort = Subset(backend.validation, indices)
    started = time.monotonic()
    # Score the restored starting model on this cohort, so a deteriorating
    # continuation cannot win merely because it is the first evaluated epoch.
    initial = matched_evaluate(backend, cohort, run, 0, False)
    best, best_epoch, bad = initial['metrics']['ndcg@10'], backend.inherited_epoch, 0
    steps = backend.initial_steps
    save_training(run.directory / 'best_trend.pt', backend, run, scheduler,
                  best_epoch, steps, selection=initial, bad_checks=bad)
    if prior_best is not None and prior_best['score'] >= best:
        best, best_epoch = prior_best['score'], prior_best['epoch']
        shutil.copy2(prior_best['checkpoint'], run.directory / 'best_trend.pt')
        run.event('historical_best_retained', epoch=best_epoch, score=best)
    for epoch in range(backend.inherited_epoch + 1, run.spec['total_epochs'] + 1):
        stage_epoch = epoch - backend.inherited_epoch
        backend.model.train()
        generator = torch.Generator().manual_seed(backend.config['seed'] + epoch)
        loader = backend.loader(backend.training, backend.microbatch, generator)
        backend.optimizer.zero_grad(set_to_none=True)
        seen = group = 0
        total_loss = 0.0
        epoch_start = last_update = time.monotonic()
        run.status('TRAINING', phase=run.spec.get('training_phase', 'native_original_schedule'), stage_epoch=stage_epoch, epoch=epoch,
                   seen=0, train_total=len(backend.training), optimizer_steps=steps,
                   stage_optimizer_steps=steps-backend.initial_steps,
                   learning_rates=scheduler.get_last_lr(), best_epoch=best_epoch, bad_checks=bad)
        for index, batch in enumerate(loader):
            loss = backend.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite original-schedule training loss')
            count = backend.size(batch)
            (loss * count).backward()
            seen += count
            group += count
            total_loss += float(loss.detach()) * count
            if group == backend.effective or index + 1 == len(loader):
                for parameter in backend.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group)
                torch.nn.utils.clip_grad_norm_(backend.model.parameters(), backend.clip, error_if_nonfinite=True)
                backend.optimizer.step()
                scheduler.step()
                backend.optimizer.zero_grad(set_to_none=True)
                steps += 1
                group = 0
            current = time.monotonic()
            if current - last_update >= 30:
                progress = dict(stage_epoch=stage_epoch, epoch=epoch, seen=seen,
                    train_total=len(backend.training), optimizer_steps=steps,
                    stage_optimizer_steps=steps-backend.initial_steps, loss=total_loss/seen,
                    epoch_elapsed_seconds=current-epoch_start,
                    epoch_eta_seconds=(len(backend.training)-seen)*(current-epoch_start)/seen,
                    learning_rates=scheduler.get_last_lr(), best_epoch=best_epoch, bad_checks=bad)
                run.status('TRAINING', phase=run.spec.get('training_phase', 'native_original_schedule'), **progress)
                run.event('training_progress', **progress)
                last_update = current
        if group or steps != epoch * backend.per_epoch:
            raise ValueError('Optimizer update count / accumulation boundary mismatch')
        save_training(run.directory / 'last.pt', backend, run, scheduler, epoch, steps,
                      best_epoch=best_epoch, best_score=best, bad_checks=bad)
        run.event('epoch_completed', stage_epoch=stage_epoch, epoch=epoch, optimizer_steps=steps,
                  loss=total_loss/seen, seconds=time.monotonic()-epoch_start,
                  learning_rates=scheduler.get_last_lr())
        if ((epoch - run.spec.get('evaluation_epoch_offset', 0)) % run.spec['trend_interval'] == 0
                or epoch == run.spec['total_epochs']):
            result = matched_evaluate(backend, cohort, run, stage_epoch, False)
            score = result['metrics']['ndcg@10']
            if score > best:
                best, best_epoch, bad = score, epoch, 0
                save_training(run.directory / 'best_trend.pt', backend, run, scheduler, epoch, steps,
                              selection=result, bad_checks=bad)
            else:
                bad += 1
            write_json(run.directory / 'selection_state.json', dict(
                epoch=epoch, best_epoch=best_epoch, best_score=best, bad_checks=bad))
            if (not run.spec.get('disable_early_stopping', False)
                    and stopping_due(epoch, run.spec['minimum_epochs'], bad, run.spec['patience'])):
                break
    selected = load_checkpoint(run.directory / 'best_trend.pt')
    backend.model.load_state_dict(selected['model'], strict=True)
    del selected
    reused = prior_best is not None and best_epoch == prior_best['epoch']
    if reused:
        full = dict(prior_best['full_validation'], reused=True,
                    source_summary=prior_best['full_summary_path'],
                    source_predictions=prior_best['full_predictions_path'],
                    source_revision=prior_best['revision'])
        write_json(run.directory / 'reused_full_validation.json', full)
        run.context['latest_full_validation'] = full
        run.event('full_validation_reused', **full)
    else:
        full = matched_evaluate(backend, backend.validation, run, best_epoch-backend.inherited_epoch, True)
    result = dict(state='COMPLETED', revision=run.spec['revision'],
        completion_scope=run.spec.get('completion_scope', 'original_schedule_continuation'),
        source_epoch=backend.inherited_epoch, selected_absolute_epoch=best_epoch,
        selected_stage_epoch=best_epoch-backend.inherited_epoch, epochs_completed=epoch,
        additional_epochs=epoch-backend.inherited_epoch, optimizer_steps=steps,
        stopping_reason='patience' if epoch < run.spec['total_epochs'] else 'epoch_limit',
        early_stopping_enabled=not run.spec.get('disable_early_stopping', False),
        inherited_best_epoch=prior_best['epoch'] if prior_best else None,
        full_validation_reused=reused,
        automatic_additional_training=False, direction_rejected=False, full_validation=full,
        seconds=time.monotonic()-started,
        interpretation=run.spec.get('interpretation',
            'Original training LR schedule restored; selection uses a fixed validation subset. '
            'Exploratory efficacy comparison, not a matched-budget structural attribution.'))
    write_json(run.directory / 'result.json', result)
    write_json(run.root / 'result.json', result)
    run.status('COMPLETED', phase='original_schedule_complete', result=result)
    run.event('original_schedule_completed', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    if spec['family'] != 'diffgrm' or spec['dataset'] != 'Beauty':
        raise ValueError('This authorized continuation is native DiffGRM Beauty only')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_uuid']:
        raise ValueError('GPU does not match the continuation config')
    run = Run(spec, args.config)
    lock = (run.root / '.native_full_resume.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((run.root / 'status.json').read_text())
    if previous['state'] != 'COMPLETED' or previous.get('revision') != 'screen_r1':
        raise ValueError('Expected the completed screen_r1 run; refusing to replace an active status')
    if alive(int(previous['pid'])):
        raise ValueError('Previous process is still alive; no duplicate run started')
    checkpoint = ROOT / spec['source_checkpoint']
    if sha256(checkpoint) != spec['source_checkpoint_sha256']:
        raise ValueError('Original epoch-33 checkpoint hash changed')
    source = load_checkpoint(checkpoint)
    config = json.loads((run.root / 'config.json').read_text())
    if source['config'] != config or source['epoch'] != spec['source_epoch']:
        raise ValueError('Original checkpoint/config identity mismatch')
    for name in ('epochs', 'minimum_epochs', 'patience'):
        if config[name] != spec['total_epochs' if name == 'epochs' else name]:
            raise ValueError(f'Original budget mismatch: {name}')
    if (spec['warmup_steps'] != config['overrides']['warmup_steps'] or
            spec['additional_epochs'] != spec['total_epochs'] - source['epoch']):
        raise ValueError('Original warmup or remaining epoch count mismatch')
    run.directory.mkdir()  # Existing revisions are never overwritten.
    write_json(run.directory / 'config.json', spec)
    write_json(run.directory / 'status_before_resume.json', previous)
    shutil.copy2(run.root / 'result.json', run.directory / 'result_before_resume.json')
    sources = [Path(__file__).resolve(), ROOT / args.config,
        ROOT / 'experiment/phase18/protocol/s18_screen_resume.py',
        ROOT / 'experiment/phase18/protocol/s18_diffgrm_native.py',
        ROOT / 'experiment/phase18/core/screen_budget.py', ROOT / 'experiment/phase18/core/diff_data.py',
        ROOT / 'experiment/phase18/analysis/screen_saved_reference.py',
        ROOT / 'experiment/phase18/run_stage18_diffgrm_full_resume.sh']
    for path in sources:
        destination = run.directory / 'sources' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(run.directory / 'source_manifest.json', {str(p.relative_to(ROOT)): sha256(p) for p in sources})
    run.context.update(inherited_epoch=source['epoch'], inherited_optimizer_steps=optimizer_steps(source['optimizer']),
        total_epochs=spec['total_epochs'], warmup_steps=spec['warmup_steps'], gpu_uuid=spec['gpu_uuid'],
        result_path=str((run.directory / 'result.json').relative_to(ROOT)),
        hard_timeout_seconds=spec['max_wall_seconds'])
    def timeout_handler(signum, frame):
        raise TimeoutError('Continuation exceeded the recorded hard wall-time limit')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, spec['max_wall_seconds'])
    try:
        run.status('RESUMING', phase='restore_original_schedule', source_checkpoint=spec['source_checkpoint'])
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable for authorized training')
        run.references = dataset_report('Beauty')
        write_json(run.directory / 'historical_references.json', run.references)
        backend = Backend(run, source)
        restore_probe(backend, run, source)
        scheduler = restore_schedule(backend.optimizer, source, config, backend.per_epoch)
        report = dict(state='PASSED', source_checkpoint=spec['source_checkpoint'],
            source_checkpoint_sha256=sha256(checkpoint), source_epoch=source['epoch'],
            optimizer_steps=backend.initial_steps, scheduler_last_epoch=scheduler.last_epoch,
            first_update_lrs=[g['lr'] for g in backend.optimizer.param_groups],
            lrs_after_next_update=[base * fn(backend.initial_steps+1)
                                   for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)],
            warmup_steps=spec['warmup_steps'], total_steps=backend.per_epoch*spec['total_epochs'],
            cpu_cuda_rng_restored=True, optimizer_updates_during_probe=0)
        write_json(run.directory / 'schedule_restore_check.json', report)
        run.event('original_schedule_restored', **report)
        save_training(run.directory / 'restored_start.pt', backend, run, scheduler,
                      backend.inherited_epoch, backend.initial_steps)
        del source
        train_original(backend, run, scheduler)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_json(run.directory / 'failure.json', failure)
        run.status('TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', **failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
