"""Complete Beauty's remaining original cosine schedule after the earlier stop."""

import argparse
import fcntl
import json
from pathlib import Path
import os
import shutil
import signal
import traceback

import torch

from experiment.phase18.analysis.screen_saved_reference import dataset_report
from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.screen_budget import optimizer_steps
from experiment.phase18.protocol.s18_diffgrm_full_resume import (
    restore_schedule, save_training, train_original,
)
from experiment.phase18.protocol.s18_screen_resume import (
    ROOT, Backend, Run, alive, load_checkpoint, restore_probe,
)


def load_sources(spec, config):
    if (spec['dataset'] != 'Beauty' or spec['family'] != 'diffgrm'
            or spec['total_epochs'] != config['epochs'] or not spec['disable_early_stopping']
            or spec['warmup_steps'] != config['overrides']['warmup_steps']
            or spec['microbatch'] != config['microbatch']
            or spec['additional_epochs'] != spec['total_epochs'] - spec['source_epoch']):
        raise ValueError('Finish-only budget/config mismatch')
    for name in ('source_checkpoint', 'prior_best_checkpoint'):
        if sha256(ROOT / spec[name]) != spec[name + '_sha256']:
            raise ValueError(f'Checkpoint hash changed: {name}')
    source = load_checkpoint(ROOT / spec['source_checkpoint'])
    best = load_checkpoint(ROOT / spec['prior_best_checkpoint'])
    for checkpoint in (source, best):
        if (checkpoint['config'] != config
                or checkpoint['revision_config']['revision'] != spec['previous_revision']
                or optimizer_steps(checkpoint['optimizer']) != checkpoint['epoch'] * spec['expected_steps_per_epoch']
                or 'cpu_rng' not in checkpoint or len(checkpoint['cuda_rng']) != 1):
            raise ValueError('Checkpoint provenance or complete-epoch state mismatch')
    if source['epoch'] != spec['source_epoch'] or not 0 < best['epoch'] < source['epoch']:
        raise ValueError('Wrong continuation/best epoch')
    previous = ROOT / spec['run_directory'] / spec['previous_revision']
    result = json.loads((previous / 'result.json').read_text())
    selection = json.loads((previous / 'selection_state.json').read_text())
    full = result['full_validation']
    if (result['epochs_completed'] != source['epoch']
            or result['selected_absolute_epoch'] != best['epoch']
            or selection['best_epoch'] != best['epoch']
            or selection['best_score'] != best['selection']['metrics']['ndcg@10']
            or source['best_score'] != selection['best_score']
            or best['selection']['examples'] != spec['trend_users']
            or full['epoch'] != best['epoch'] or full['scope'] != 'full_validation'):
        raise ValueError('Historical best/selection mismatch')
    full_summary = previous / f'full_validation_stage_epoch_{full["stage_epoch"]:02d}.json'
    if json.loads(full_summary.read_text()) != full:
        raise ValueError('Historical full validation summary changed')
    full_predictions = full_summary.with_suffix('.jsonl')
    if sha256(full_predictions) != full['predictions_sha256']:
        raise ValueError('Historical full validation predictions changed')
    trend_predictions = previous / f'trend_subset_stage_epoch_{best["stage_epoch"]:02d}.jsonl'
    if sha256(trend_predictions) != best['selection']['predictions_sha256']:
        raise ValueError('Historical selection predictions changed')
    return source, dict(epoch=best['epoch'], score=selection['best_score'],
        checkpoint=str(ROOT / spec['prior_best_checkpoint']), full_validation=full,
        full_summary_path=str(full_summary.relative_to(ROOT)),
        full_predictions_path=str(full_predictions.relative_to(ROOT)), revision=spec['previous_revision'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_uuid']:
        raise ValueError('GPU does not match the continuation config')
    run = Run(spec, args.config)
    lock = (run.root / '.native_full_resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((run.root / 'status.json').read_text())
    if (previous['state'] != 'COMPLETED' or previous['revision'] != spec['previous_revision']
            or alive(int(previous['pid']))):
        raise ValueError('Expected an inactive completed full_r2 run')
    config = json.loads((run.root / 'config.json').read_text())
    source, prior_best = load_sources(spec, config)
    run.directory.mkdir()  # Never overwrite a previous attempt.
    write_json(run.directory / 'config.json', spec)
    write_json(run.directory / 'status_before_resume.json', previous)
    write_json(run.directory / 'prior_best.json', prior_best)
    shutil.copy2(run.root / 'result.json', run.directory / 'result_before_resume.json')
    paths = [Path(__file__).resolve(), ROOT / args.config,
        ROOT / 'experiment/phase18/protocol/s18_diffgrm_full_resume.py',
        ROOT / 'experiment/phase18/protocol/s18_screen_resume.py',
        ROOT / 'experiment/phase18/protocol/s18_diffgrm_native.py',
        ROOT / 'experiment/phase18/core/diff_data.py', ROOT / 'experiment/phase18/core/screen_budget.py',
        ROOT / 'experiment/phase18/analysis/screen_saved_reference.py',
        ROOT / 'experiment/phase18/run_stage18_diffgrm_finish.sh']
    for path in paths:
        dest = run.directory / 'sources' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(run.directory / 'source_manifest.json', {str(p.relative_to(ROOT)): sha256(p) for p in paths})
    run.context.update(inherited_epoch=source['epoch'], inherited_optimizer_steps=optimizer_steps(source['optimizer']),
        inherited_best_epoch=prior_best['epoch'], inherited_best_score=prior_best['score'],
        total_epochs=spec['total_epochs'], warmup_steps=spec['warmup_steps'], gpu_uuid=spec['gpu_uuid'],
        early_stopping_enabled=False, result_path=str((run.directory / 'result.json').relative_to(ROOT)),
        hard_timeout_seconds=spec['max_wall_seconds'])
    def timeout_handler(signum, frame):
        raise TimeoutError('Cosine-tail continuation exceeded its wall-time limit')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, spec['max_wall_seconds'])
    try:
        run.status('RESUMING', phase='restore_original_cosine_tail')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        run.references = dataset_report(spec['dataset'])
        write_json(run.directory / 'historical_references.json', run.references)
        backend = Backend(run, source)
        restore_probe(backend, run, source)
        scheduler = restore_schedule(backend.optimizer, source, config, backend.per_epoch)
        report = dict(state='PASSED', source_checkpoint=spec['source_checkpoint'],
            source_checkpoint_sha256=spec['source_checkpoint_sha256'], source_epoch=source['epoch'],
            optimizer_steps=backend.initial_steps, scheduler_last_epoch=scheduler.last_epoch,
            first_update_lrs=scheduler.get_last_lr(),
            lrs_after_next_update=[b*f(backend.initial_steps+1) for b,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)],
            lrs_at_final_update=[b*f(spec['total_epochs']*backend.per_epoch) for b,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)],
            total_steps=spec['total_epochs']*backend.per_epoch, cpu_cuda_rng_restored=True,
            optimizer_updates_during_probe=0, early_stopping_enabled=False)
        write_json(run.directory / 'schedule_restore_check.json', report)
        run.event('original_schedule_restored', **report)
        save_training(run.directory / 'restored_start.pt', backend, run, scheduler,
                      backend.inherited_epoch, backend.initial_steps)
        del source
        train_original(backend, run, scheduler, prior_best=prior_best)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_json(run.directory / 'failure.json', failure)
        run.status('TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', **failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
