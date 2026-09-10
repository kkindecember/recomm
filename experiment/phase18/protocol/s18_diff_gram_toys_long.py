"""Fresh Toys GRAM parent + new DIFF/Adam, with a fixed 1+20 epoch schedule."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import torch
from torch.utils.data import Subset

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.protocol import s18_diff_gram as original
from experiment.phase18.protocol.s18_diff_gram_fixed import (
    FixedRun, FreshBackend, frozen_references, train_fixed,
)

ROOT = original.ROOT


def validate_spec(spec):
    if (spec['family'], spec['dataset'], spec['seed'], spec['total_epochs'],
            spec['additional_epochs']) != ('diff_gram', 'Toys', 2023, 21, 21):
        raise ValueError('This contract is a fresh Toys 1+20 run only')
    for name, key in [('model_config', 'model_config_sha256'),
                      ('original_model_config', 'original_model_config_sha256')]:
        if sha256(ROOT / spec[name]) != spec[key]:
            raise ValueError(f'Frozen config changed: {name}')
    original_config = json.loads((ROOT / spec['original_model_config']).read_text())
    config = json.loads((ROOT / spec['model_config']).read_text())
    expected = copy.deepcopy(original_config)
    expected['training']['joint_epochs'] = 20
    if config != expected:
        raise ValueError('Only the declared joint horizon may differ from the original Toys model config')
    if spec['microbatch'] != 4 or spec['trend_users'] != 2000:
        raise ValueError('Retain the old Toys microbatch and selection population')
    if (config['training']['warmup_epochs'] != 1 or config['training']['evaluation_interval'] != 2 or
            config['parent_checkpoint'] != spec['parent_checkpoint'] or
            config['parent_sha256'] != spec['parent_sha256']):
        raise ValueError('Frozen phases, selection cadence or original GRAM parent changed')
    return config


def runtime_profile(backend, run):
    # A deterministic sample independent of targets/scores, used solely for timing.
    indices = torch.randperm(len(backend.validation), generator=torch.Generator().manual_seed(617)).tolist()[:16]
    backend.model.eval()
    timings, lengths = [], []
    for index in indices:
        batch = next(iter(backend.loader(Subset(backend.validation, [index]), 1)))
        torch.cuda.synchronize()
        started = time.monotonic()
        rows = backend.predict(batch)
        torch.cuda.synchronize()
        timings.append(time.monotonic()-started)
        lengths.append(len(backend.records[index]['history']))
        if len(rows) != 1 or len(rows[0][2]) != 50:
            raise ValueError('Timing sample must produce 50 unique catalog candidates')
    mean = sum(timings)/len(timings)
    result = dict(generation_users=len(indices), seconds_per_user=mean,
        per_user_seconds=timings, history_lengths=lengths,
        projected_full_validation_hours=mean*len(backend.validation)/3600,
        projected_all_validation_hours=mean*(len(backend.validation)+11*2000)/3600,
        interpretation='Small-sample timing on disposable smoke weights; shared GPU load may change.',
        efficacy_evidence=False)
    write_json(run.directory/'runtime_profile.json', result)
    run.event('runtime_profile_completed', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('smoke', 'run'), required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT/args.config).read_text())
    config = validate_spec(spec)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_uuid']:
        raise ValueError('GPU UUID differs from the frozen specification')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One visible CUDA GPU is required')
    directory = spec['smoke_directory'] if args.mode == 'smoke' else spec['run_directory']
    run = FixedRun(spec, args.config, directory)
    run.directory.mkdir(parents=True, exist_ok=False)
    def timeout_handler(signum, frame):
        raise TimeoutError('Fresh Toys long schedule exceeded the frozen wall-time limit')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 1200 if args.mode == 'smoke' else spec['max_wall_seconds'])
    try:
        run.context.update(gpu_uuid=spec['gpu_uuid'], early_stopping_enabled=False,
            initialization='original_gram_parent_new_diff_new_adam', cumulative_joint_epochs=20)
        run.status('INITIALIZING', phase='fresh_gram_parent', mode=args.mode)
        if args.mode == 'run':
            smoke = ROOT/spec['smoke_directory']
            summary = json.loads((smoke/'smoke_summary.json').read_text())
            if (summary['state'] != 'SMOKE_PASSED' or summary['microbatch'] != spec['microbatch'] or
                    json.loads((smoke/'status.json').read_text())['state'] != 'SMOKE_PASSED' or
                    json.loads((smoke/'runtime_profile.json').read_text())['generation_users'] != 16 or
                    json.loads((smoke/'execution_spec.json').read_text()) != spec):
                raise ValueError('Passing GPU smoke or execution specification mismatch')
            for path, expected in json.loads((smoke/'manifest.json').read_text())['source_sha256'].items():
                if sha256(ROOT/path) != expected:
                    raise ValueError(f'Source changed since GPU smoke: {path}')
        original.seed_everything(spec['seed'])
        backend = FreshBackend(config, spec['microbatch'], torch.device('cuda:0'))
        if (len(backend.training), len(backend.validation), backend.per_epoch) != (109361,19412,855):
            raise ValueError('Full Toys data or optimizer count mismatch')
        original.snapshot(run.directory, config, args, backend.manifest, backend.model)
        manifest = json.loads((run.directory/'manifest.json').read_text())
        for path in [spec['model_config'], spec['original_model_config'],
                     'experiment/phase18/run_stage18_diff_gram_toys_long.sh']:
            destination = run.directory/'sources'/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/path, destination)
            manifest['source_sha256'][path] = sha256(ROOT/path)
        write_json(run.directory/'manifest.json', manifest)
        write_json(run.directory/'execution_spec.json', spec)
        cohort = frozen_references(backend, run)
        if args.mode == 'smoke':
            original.smoke(backend.model, backend.training, backend.collator, backend.decoder,
                           config, run.directory, backend.device, spec['microbatch'])
            runtime_profile(backend, run)
            run.status('SMOKE_PASSED', phase='gpu_checks_complete', optimizer_updates_used_for_training=0)
        else:
            write_json(run.directory/'initialization_check.json', dict(state='PASSED',
                parent_checkpoint=config['parent_checkpoint'], parent_sha256=config['parent_sha256'],
                strict_parent_load=True, optimizer_steps=0, fresh_new_branch=True,
                old_diff_checkpoint_used=False, smoke_weights_used=False,
                train_examples=109361, validation_examples=19412, steps_per_epoch=855,
                seed=spec['seed'], test_read=False))
            train_fixed(backend, run, cohort)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_json(run.directory/'failure.json', failure)
        run.status('TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', **failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
