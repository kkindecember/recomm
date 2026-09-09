"""Fresh DIFF→GRAM rerun: fixed full-data phases, sparse selection, one full validation."""

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import get_linear_schedule_with_warmup

from experiment.phase18.analysis.screen_saved_reference import close, keyed, metrics
from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.screen_budget import cohort_indices
from experiment.phase18.protocol import s18_diff_gram as original
from experiment.phase18.protocol.s18_screen_resume import Backend, Run, evaluate, load_checkpoint
from experiment.phase18.protocol.s18_diffgrm_full_resume import save_training

ROOT = original.ROOT


class FixedRun(Run):
    def __init__(self, spec, config_path, directory=None):
        super().__init__(spec, config_path)
        self.root = ROOT / (directory or spec['run_directory'])
        self.directory = self.root
        self.context.update(total_epochs=spec['total_epochs'], seed=spec['seed'],
            training_policy='fresh_parent_fixed_phases_no_early_stopping',
            schedule_provenance='fresh Adam; planned frozen and joint linear schedules; no handoff/bridge',
            result_path=str((self.root / 'result.json').relative_to(ROOT)),
            hard_timeout_seconds=spec['max_wall_seconds'])


class FreshBackend(Backend):
    """Reuse the established data/loss/generation interfaces without restoring trained DIFF weights."""
    def __init__(self, config, microbatch, device):
        self.old, self.family, self.config, self.device = original, 'diff_gram', config, device
        (self.catalog, self.training, self.validation, self.manifest,
         self.tokenizer, self.collator) = original.load_inputs(config)
        self.model = original.build_model(config, self.catalog, self.tokenizer, device)
        self.optimizer = original.make_optimizer(self.model, config)
        self.decoder = original.CatalogDecoder(self.catalog, config)
        self.microbatch = microbatch
        self.effective = config['training']['effective_batch_size']
        self.eval_batch, self.workers = 1, config['training']['num_workers']
        self.clip = config['training']['gradient_clip']
        self.baseline, self.records = self.manifest['baseline_validation'], self.validation.records
        self.per_epoch = math.ceil(len(self.training) / self.effective)
        self.initial_steps = self.inherited_epoch = 0


@contextmanager
def preserve_training_rng():
    python_state, numpy_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def matched_validation(backend, dataset, run, epoch, full):
    with preserve_training_rng():
        result = evaluate(backend, dataset, run, epoch, full)
    references = run.references['historical_full' if full else 'matched_subset']
    result.update(epoch=epoch, historical_gram=references['gram'],
        historical_gram_pcrf=references['gram_pcrf'],
        delta={k: v - references['gram'][k] for k, v in result['metrics'].items()},
        delta_vs_gram_pcrf={k: v - references['gram_pcrf'][k] for k, v in result['metrics'].items()},
        interpretation='Same-user exploratory validation; fixed seed and selected checkpoint, not independent test.')
    write_json(run.directory / f"{result['scope']}_stage_epoch_{epoch:02d}.json", result)
    run.context['latest_full_validation' if full else 'latest_trend_validation'] = result
    run.event('matched_validation_completed', **result)
    return result


def frozen_references(backend, run):
    spec = run.spec
    for path_key, hash_key in [('cohort_source', 'cohort_sha256'),
                               ('reference_cache', 'reference_cache_sha256')]:
        if sha256(ROOT / spec[path_key]) != spec[hash_key]:
            raise ValueError(f'Frozen reference changed: {path_key}')
    cohort = json.loads((ROOT / spec['cohort_source']).read_text())
    references = json.loads((ROOT / spec['reference_cache']).read_text())
    indices = cohort_indices(backend.records, spec['dataset'], spec['seed'], spec['trend_users'])
    if [str(backend.records[i]['user_id']) for i in indices] != cohort['user_ids']:
        raise ValueError('Selection cohort changed')
    reference_path = ROOT / references['reference']
    if sha256(reference_path) != references['reference_sha256']:
        raise ValueError('Historical per-user cache changed')
    with reference_path.open() as handle:
        cached = keyed(list(csv.DictReader(handle, delimiter='\t')))
    if set(cached) != {str(r['user_id']) for r in backend.records}:
        raise ValueError('Validation users differ from frozen reference')
    for scope, users in [('historical_full', list(cached)), ('matched_subset', cohort['user_ids'])]:
        for model, field in [('gram', 'baseline_rank'), ('gram_pcrf', 'pcrf_rank')]:
            close(metrics([int(cached[u][field]) for u in users]), references[scope][model])
    close(references['historical_full']['gram'], backend.baseline)
    run.references = {k: references[k] for k in ('dataset', 'historical_full', 'matched_subset',
        'reference', 'reference_sha256', 'cohort_sha256', 'provenance_note')}
    write_json(run.directory / 'historical_references.json', run.references)
    write_json(run.directory / 'trend_cohort.json', cohort)
    return Subset(backend.validation, indices)


def train_fixed(backend, run, cohort):
    settings = backend.config['training']
    if backend.optimizer.state or backend.initial_steps or backend.inherited_epoch:
        raise ValueError('Fixed rerun must start with fresh Adam and no inherited DIFF training')
    if backend.effective % backend.microbatch:
        raise ValueError('Microbatch must divide effective batch')
    phases = [('warmup', settings['warmup_epochs']), ('joint', settings['joint_epochs'])]
    if sum(length for _, length in phases) != run.spec['total_epochs']:
        raise ValueError('Declared budget differs from fixed phase lengths')
    started = time.monotonic()
    best, best_epoch, steps, epoch = -math.inf, None, 0, 0
    plan = {phase: dict(epochs=length, optimizer_steps=length * backend.per_epoch,
            warmup_steps=max(1, int(length * backend.per_epoch * settings['warmup_fraction'])))
            for phase, length in phases}
    write_json(run.directory / 'schedule_plan.json', dict(phases=plan,
        total_epochs=run.spec['total_epochs'], optimizer_steps=run.spec['total_epochs'] * backend.per_epoch,
        early_stopping_enabled=False, bridge_steps=0, fresh_optimizer=True))
    for phase, phase_epochs in phases:
        backend.model.set_backbone_frozen(phase == 'warmup')
        for group, peak in zip(backend.optimizer.param_groups,
                               [settings['new_learning_rate'], settings['backbone_learning_rate']]):
            group['lr'] = group['initial_lr'] = peak
        scheduler = get_linear_schedule_with_warmup(backend.optimizer,
            plan[phase]['warmup_steps'], plan[phase]['optimizer_steps'])
        run.event('phase_started', phase=phase, **plan[phase], learning_rates=scheduler.get_last_lr(),
                  accumulated_optimizer_steps=steps, optimizer_moments_preserved=(phase == 'joint'))
        if epoch == 0:
            save_training(run.directory / 'initial.pt', backend, run, scheduler, epoch, steps, phase=phase)
        for phase_epoch in range(1, phase_epochs + 1):
            epoch += 1
            backend.model.train()
            generator = torch.Generator().manual_seed(backend.config['seed'] + epoch)
            loader = backend.loader(backend.training, backend.microbatch, generator)
            backend.optimizer.zero_grad(set_to_none=True)
            seen = group_size = 0
            total_loss = 0.0
            epoch_start = last_update = time.monotonic()
            run.status('TRAINING', phase=phase, epoch=epoch, phase_epoch=phase_epoch,
                seen=0, train_total=len(backend.training), optimizer_steps=steps, best_epoch=best_epoch,
                learning_rates=scheduler.get_last_lr(), effective_batch=backend.effective,
                microbatch=backend.microbatch)
            for index, batch in enumerate(loader):
                loss = backend.loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite fixed-schedule loss')
                count = backend.size(batch)
                (loss * count).backward()
                seen += count
                group_size += count
                total_loss += float(loss.detach()) * count
                if group_size == backend.effective or index + 1 == len(loader):
                    for parameter in backend.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_size)
                    torch.nn.utils.clip_grad_norm_(backend.model.parameters(), backend.clip, error_if_nonfinite=True)
                    backend.optimizer.step()
                    scheduler.step()
                    backend.optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    group_size = 0
                current = time.monotonic()
                if current - last_update >= 30:
                    progress = dict(phase=phase, epoch=epoch, phase_epoch=phase_epoch, seen=seen,
                        train_total=len(backend.training), optimizer_steps=steps, loss=total_loss / seen,
                        epoch_elapsed_seconds=current - epoch_start,
                        epoch_eta_seconds=(len(backend.training)-seen)*(current-epoch_start)/seen,
                        learning_rates=scheduler.get_last_lr(), best_epoch=best_epoch)
                    run.status('TRAINING', **progress)
                    run.event('training_progress', **progress)
                    last_update = current
            if (group_size or seen != len(backend.training) or steps != epoch * backend.per_epoch
                    or scheduler.last_epoch != phase_epoch * backend.per_epoch):
                raise ValueError('Data coverage, optimizer count or fixed scheduler position mismatch')
            if phase == 'warmup' or phase_epoch % settings['evaluation_interval'] == 0 or epoch == run.spec['total_epochs']:
                result = matched_validation(backend, cohort, run, epoch, False)
                if result['metrics']['ndcg@10'] > best:
                    best, best_epoch = result['metrics']['ndcg@10'], epoch
                    save_training(run.directory / 'best_trend.pt', backend, run, scheduler, epoch, steps,
                                  phase=phase, selection=result)
                write_json(run.directory / 'selection_state.json', dict(epoch=epoch,
                    best_epoch=best_epoch, best_score=best, early_stopping_enabled=False))
            save_training(run.directory / 'last.pt', backend, run, scheduler, epoch, steps,
                          phase=phase, phase_epoch=phase_epoch, best_epoch=best_epoch, best_score=best)
            run.event('epoch_completed', phase=phase, epoch=epoch, optimizer_steps=steps,
                train_examples_seen=seen, loss=total_loss / seen, seconds_including_selection=time.monotonic()-epoch_start,
                learning_rates=scheduler.get_last_lr(), best_epoch=best_epoch)
    selected = load_checkpoint(run.directory / 'best_trend.pt')
    backend.model.load_state_dict(selected['model'], strict=True)
    del selected
    full = matched_validation(backend, backend.validation, run, best_epoch, True)
    result = dict(state='COMPLETED', revision=run.spec['revision'],
        completion_scope='fresh_parent_fixed_full_data_schedule', source_epoch=0,
        selected_absolute_epoch=best_epoch, selected_stage_epoch=best_epoch, epochs_completed=epoch,
        optimizer_steps=steps, stopping_reason='epoch_limit', early_stopping_enabled=False,
        automatic_additional_training=False, full_validation=full, seconds=time.monotonic()-started,
        interpretation='Same-seed clean-schedule rerun from historical GRAM parent; not a multi-seed '
                       'confirmation, matched pure-GRAM continuation control, or independent test.')
    write_json(run.directory / 'result.json', result)
    run.status('COMPLETED', phase='fixed_schedule_complete', result=result)
    run.event('training_completed', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('smoke', 'run'), required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    if (spec['family'], spec['dataset'], spec['seed'], spec['total_epochs']) != ('diff_gram', 'Beauty', 2023, 11):
        raise ValueError('This run is the authorized Beauty fixed-schedule rerun only')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_uuid']:
        raise ValueError('GPU UUID differs from the run specification')
    if sha256(ROOT / spec['model_config']) != spec['model_config_sha256']:
        raise ValueError('Frozen model/training configuration changed')
    config = json.loads((ROOT / spec['model_config']).read_text())
    if config['seed'] != spec['seed'] or config['training']['warmup_epochs'] != 1 or config['training']['joint_epochs'] != 10:
        raise ValueError('Original seed or 1+10 phase budget changed')
    directory = spec['smoke_directory'] if args.mode == 'smoke' else spec['run_directory']
    run = FixedRun(spec, args.config, directory)
    run.directory.mkdir(parents=True, exist_ok=False)
    def timeout_handler(signum, frame):
        raise TimeoutError('Fixed-schedule run exceeded its recorded wall-time limit')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 1200 if args.mode == 'smoke' else spec['max_wall_seconds'])
    try:
        run.status('INITIALIZING', phase='fresh_gram_parent', mode=args.mode)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.mode == 'run':
            smoke_path = ROOT / spec['smoke_directory']
            summary = json.loads((smoke_path / 'smoke_summary.json').read_text())
            prior = json.loads((smoke_path / 'manifest.json').read_text())
            if summary['state'] != 'SMOKE_PASSED' or summary['microbatch'] != spec['microbatch'] or prior['config'] != config:
                raise ValueError('GPU smoke or frozen configuration mismatch')
            for name, expected in prior['source_sha256'].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError(f'Source changed since GPU smoke: {name}')
            if json.loads((smoke_path / 'execution_spec.json').read_text()) != spec:
                raise ValueError('Execution specification changed since GPU smoke')
        original.seed_everything(spec['seed'])
        backend = FreshBackend(config, spec['microbatch'], torch.device('cuda:0'))
        if (len(backend.training), len(backend.validation), backend.per_epoch) != (131413, 22363, 1027):
            raise ValueError('Expected full Beauty data and 1027 optimizer updates per epoch')
        original.snapshot(run.directory, config, args, backend.manifest, backend.model)
        write_json(run.directory / 'execution_spec.json', spec)
        cohort = frozen_references(backend, run)
        if args.mode == 'smoke':
            original.smoke(backend.model, backend.training, backend.collator, backend.decoder, config,
                           run.directory, backend.device, spec['microbatch'])
            run.status('SMOKE_PASSED', phase='gpu_checks_complete', optimizer_updates_used_for_training=0)
        else:
            write_json(run.directory / 'initialization_check.json', dict(state='PASSED',
                parent_checkpoint=config['parent_checkpoint'], parent_sha256=config['parent_sha256'],
                strict_parent_load=True, optimizer_steps=0, fresh_new_branch=True,
                smoke_weights_used=False, train_examples=len(backend.training),
                validation_examples=len(backend.validation), steps_per_epoch=backend.per_epoch,
                seed=spec['seed'], test_read=False))
            train_fixed(backend, run, cohort)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_json(run.directory / 'failure.json', failure)
        run.status('TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', **failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
