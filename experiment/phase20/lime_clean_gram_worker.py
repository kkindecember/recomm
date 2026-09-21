"""Bounded checks and formal GRAM -> C1 training on unexposed train events."""
import argparse
import gc
import math
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.lime_clean_gram_core import (
    ROOT, T5, SPECS, read, write, sha, now, load_inputs, epoch_order, update_windows,
    seed_all, cpu_tree, atomic_save, pack_state, load_state, build_model,
    make_optimizer, optimizer_update, total_updates,
)

CONFIG = ROOT / os.environ.get('RECOMM_CLEAN_GRAM_CONFIG', 'experiment/phase20/lime_clean_gram_20260920_v1.json')


def check_manifest(manifest):
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, f'Frozen input/source changed: {name}'


def prepare(config):
    out = ROOT / config['output']
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'prepared.json').exists(), 'Prepared inputs already frozen'
    source_paths = set((ROOT / 'experiment/phase20').glob('*lime_clean_gram*'))
    source_paths.update((ROOT / 'GRAM/src').rglob('*.py'))
    source_paths.update((ROOT / 'experiment/phase17/core').rglob('*.py'))
    source_paths.update((ROOT / 'experiment/phase17/registry').rglob('*.py'))
    source_paths.update(ROOT / p for p in ('experiment/phase20/lime_clean_upstream_core.py',
        'experiment/phase20/guided_decoding_job.py', 'experiment/phase4/gcdh_p0.py',
        'experiment/phase3/hbtr_b1_smoke.py', config['contract']))
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(source_paths) if p.is_file()}
    for relative in sources:
        target = out / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write(out / 'config.json', config)
    shutil.copy2(ROOT / config['contract'], out / 'protocol.md')
    domains = {}
    for domain in config['domains']:
        data = load_inputs(domain)
        path = out / domain
        path.mkdir(exist_ok=True)
        manifest = dict(data['manifest'], **sources)
        if config.get('execution_mode_evidence'):
            evidence = ROOT / config['execution_mode_evidence']
            result = read(evidence)
            assert result['state'] == 'COMPLETED'
            assert result['reports']['native_tf32']['numerical_check_passed']
            manifest[config['execution_mode_evidence']] = sha(evidence)
        write(path / 'input_manifest.json', manifest)
        domains[domain] = dict(data['summary'], expected_gram_updates=total_updates(data, 'gram', config),
                               expected_c1_updates=total_updates(data, 'c1', config))
        write(path / 'inputs.json', domains[domain])
        print(domains[domain], flush=True)
        del data
    check_manifest(sources)
    write(out / 'prepared.json', dict(state='PREPARED', created_at=now(), domains=domains,
          source_sha256=sources, config_sha256=sha(CONFIG), development_labels_read=False, test_read=False))


class Worker:
    def __init__(self, config, domain, action, gpu_uuid=None, resume=False):
        self.c, self.domain, self.action = config, domain, action
        self.root = ROOT / config['output'] / domain
        self.out = self.root / action
        self.out.mkdir(exist_ok=resume)
        self.resume = resume
        self.start = time.monotonic()
        self.gpu_uuid = gpu_uuid
        self.device = torch.device('cpu' if action == 'cpu_check' else 'cuda')
        self.stop_requested = False
        torch.set_num_threads(config['cpu_threads'])
        torch.set_num_interop_threads(1)
        self.thread_limit = threadpool_limits(config['cpu_threads'])
        torch.backends.cuda.matmul.allow_tf32 = config.get('allow_tf32', False)
        torch.backends.cudnn.allow_tf32 = config.get('allow_tf32', False)
        self.manifest = read(self.root / 'input_manifest.json')
        self.status('核对冻结输入')
        check_manifest(self.manifest)
        self.data = load_inputs(domain)
        assert all(self.manifest.get(k) == v for k, v in self.data['manifest'].items())
        if self.device.type == 'cuda':
            assert gpu_uuid and os.environ.get('CUDA_VISIBLE_DEVICES') == gpu_uuid
            import subprocess
            free = int(subprocess.check_output(['nvidia-smi', '--id='+gpu_uuid,
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            assert free >= config['minimum_free_mib'], f'Free GPU memory fell to {free} MiB before worker launch'
            assert torch.cuda.is_available()
            total = torch.cuda.get_device_properties(0).total_memory / 2**20
            torch.cuda.set_per_process_memory_fraction(config['allocator_limit_mib']/total)
        seed_all(config['seed'])

    def status(self, message, state='RUNNING', **kwargs):
        value = dict(state=state, stage=message, domain=self.domain, action=self.action,
                     pid=os.getpid(), updated_at=now(), elapsed_seconds=time.monotonic()-self.start,
                     gpu_uuid=self.gpu_uuid, development_labels_read=False, test_read=False, **kwargs)
        write(self.out / 'status.json', value)
        print(value, flush=True)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def new_stage(self, stage, parent=None):
        seed_all(self.c['seed'])
        model, collator = build_model(self.data, stage, self.c, self.device, parent)
        optimizer, scheduler = make_optimizer(model, stage, self.c, total_updates(self.data, stage, self.c))
        return model, collator, optimizer, scheduler

    def cpu_check(self):
        """Real dropout/optimizer replay, independently for each domain and stage."""
        reports, parent = {}, None
        for stage in ('gram', 'c1'):
            self.status('CPU真实训练和恢复检查', training_stage=stage)
            model, collator, optimizer, scheduler = self.new_stage(stage, parent)
            dataset = self.data['dataset' if stage == 'gram' else 'c1_dataset']
            # Include longest histories, not target/coverage-selected examples.
            order = np.argsort([len(r['history']) for r in dataset.rows], kind='stable')
            ix = order[np.linspace(0, len(order)-1, 4, dtype=int)].tolist()
            first, second = [[ix[0]], [ix[1]]], [[ix[2]], [ix[3]]]
            a = optimizer_update(model, optimizer, scheduler, collator, dataset, first, stage, self.c, self.device)
            checkpoint = pack_state(model, optimizer, scheduler, dict(epoch=1, window=1, updates=1),
                self.manifest, stage, self.c, probe_only=True, formal_parent=False)
            path = self.out / f'{stage}_resume_probe.pt'
            atomic_save(path, checkpoint)
            b = optimizer_update(model, optimizer, scheduler, collator, dataset, second, stage, self.c, self.device)
            continuous = cpu_tree(model.state_dict())
            continuous_opt = cpu_tree(optimizer.state_dict())
            del checkpoint
            saved = torch.load(path, map_location='cpu')
            position = load_state(saved, model, optimizer, scheduler, self.manifest, stage, self.c)
            assert position == dict(epoch=1, window=1, updates=1)
            del saved
            replay = optimizer_update(model, optimizer, scheduler, collator, dataset, second, stage, self.c, self.device)
            assert b['loss'] == replay['loss']
            for key, value in continuous.items():
                assert torch.isfinite(value).all() and torch.equal(value, model.state_dict()[key].cpu()), key
            for key, value in continuous_opt['state'].items():
                for name, tensor in value.items():
                    other = optimizer.state_dict()['state'][key][name]
                    assert torch.equal(tensor, other.cpu()) if torch.is_tensor(tensor) else tensor == other
            reports[stage] = dict(loss_first=a['loss'], loss_second=b['loss'], exact_resume=True,
                                  model_tensors=len(continuous), real_training_rows=ix,
                                  checkpoint_sha256=sha(path), formal_parent=False)
            if stage == 'gram':
                parent = continuous
            del model, optimizer, scheduler, collator, continuous_opt
            gc.collect()
        check_manifest(self.manifest)
        write(self.out / 'summary.json', dict(state='PASSED', domains=self.domain, stages=reports,
            input_sha256=self.manifest, seconds=time.monotonic()-self.start, test_read=False,
            development_labels_read=False, formal_training_started=False))
        self.status('CPU保存恢复与连续训练完全一致', state='PASSED')

    def profile(self):
        reports, parent = {}, None
        for stage in ('gram', 'c1'):
            model, collator, optimizer, scheduler = self.new_stage(stage, parent)
            dataset = self.data['dataset' if stage == 'gram' else 'c1_dataset']
            width = self.c['batch_size'] * self.c['gradient_accumulation']
            longest = sorted(range(len(dataset)), key=lambda i: len(dataset.rows[i]['history']), reverse=True)[:width]
            stress_batches = list(update_windows(longest, self.c['batch_size'], self.c['gradient_accumulation']))[0]
            groups = list(update_windows(epoch_order(len(dataset), self.c['seed'], 1),
                          self.c['batch_size'], self.c['gradient_accumulation']))[:self.c['profile_random_updates']]
            records = []
            torch.cuda.reset_peak_memory_stats()
            for i, batches in enumerate([stress_batches] + groups):
                self.status('正式批量与最大输入形状检查', training_stage=stage, profile_window=i,
                            batch_size=self.c['batch_size'], accumulation=self.c['gradient_accumulation'])
                self.sync()
                started = time.monotonic()
                record = optimizer_update(model, optimizer, scheduler, collator, dataset, batches,
                                          stage, self.c, self.device, stress=(i == 0))
                self.sync()
                record.update(seconds=time.monotonic()-started, stress=(i == 0))
                records.append(record)
                assert all(torch.isfinite(p).all() for p in model.parameters())
            durations = [r['seconds'] for r in records if not r['stress']]
            reports[stage] = dict(records=records, allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                mean_projected_training_hours=float(np.mean(durations))*total_updates(self.data, stage, self.c)/3600,
                conservative_projected_training_hours=max(durations)*total_updates(self.data, stage, self.c)/3600)
            if stage == 'gram':
                parent = cpu_tree(model.state_dict())
            del model, optimizer, scheduler, collator
            gc.collect()
            torch.cuda.empty_cache()
        check_manifest(self.manifest)
        total = sum(r['conservative_projected_training_hours'] for r in reports.values())
        accepted = total <= self.c['maximum_projected_training_hours']
        write(self.out / 'summary.json', dict(state='PASSED' if accepted else 'COST_GATE_FAILED',
            stages=reports, conservative_projected_training_hours=total, gpu_uuid=self.gpu_uuid,
            seconds=time.monotonic()-self.start, input_sha256=self.manifest,
            formal_training_started=False, profile_weights_discarded=True,
            estimate_excludes='candidate generation, LIME calibration, ranker fitting, evaluation; load may change'))
        self.status('正式批量检查完成' if accepted else '当前吞吐超过已定成本门槛',
                    state='PASSED' if accepted else 'COST_GATE_FAILED', projected_training_hours=total)

    def train(self):
        assert read(self.root / 'cpu_check/summary.json')['state'] == 'PASSED'
        profile = read(self.root / 'profile/summary.json')
        assert profile['state'] == 'PASSED' and profile['input_sha256'] == self.manifest
        previous_seconds = 0.
        stage_summaries = {}
        for stage in ('gram', 'c1'):
            directory = self.out / stage
            directory.mkdir(exist_ok=self.resume)
            if (directory / 'summary.json').exists():
                assert self.resume
                summary = read(directory / 'summary.json')
                assert summary['state'] == 'COMPLETED'
                assert sha(directory / 'final.pt') == summary['checkpoint_sha256']
                stage_summaries[stage] = summary
                previous_seconds += summary['training_seconds']
                continue
            parent = None
            if stage == 'c1':
                gram_checkpoint = torch.load(self.out / 'gram/final.pt', map_location='cpu')
                assert gram_checkpoint['input_sha256'] == self.manifest and gram_checkpoint['formal_parent']
                assert gram_checkpoint['position']['epoch'] == self.c['gram_epochs']+1
                parent = gram_checkpoint['state_dict']
                del gram_checkpoint
            model, collator, optimizer, scheduler = self.new_stage(stage, parent)
            del parent
            dataset = self.data['dataset' if stage == 'gram' else 'c1_dataset']
            position = dict(epoch=1, window=0, updates=0)
            old_stage_seconds = 0.
            if self.resume and (directory / 'latest.pt').exists():
                checkpoint = torch.load(directory / 'latest.pt', map_location='cpu')
                assert checkpoint['formal_parent'] and not checkpoint['probe_only']
                position = load_state(checkpoint, model, optimizer, scheduler, self.manifest, stage, self.c)
                old_stage_seconds = checkpoint['training_seconds']
                del checkpoint
            stage_start = time.monotonic()
            last_report = 0.

            def elapsed():
                return old_stage_seconds + time.monotonic()-stage_start

            def save(name='latest.pt'):
                atomic_save(directory / name, pack_state(model, optimizer, scheduler, position,
                    self.manifest, stage, self.c, training_seconds=elapsed(),
                    probe_only=False, formal_parent=True, domain=self.domain,
                    catalog=self.data['ids'], lexical_type=SPECS[self.domain]['lexical']))

            model.train()
            with (directory / 'updates.jsonl').open('a' if self.resume else 'x', buffering=1) as log:
                import json
                log.write(json.dumps(dict(event='attempt_start', pid=os.getpid(), resumed=self.resume,
                    restored_position=position.copy(), created_at=now()))+'\n')
                while position['epoch'] <= self.c[f'{stage}_epochs']:
                    epoch = position['epoch']
                    windows = list(update_windows(epoch_order(len(dataset), self.c['seed'], epoch),
                                   self.c['batch_size'], self.c['gradient_accumulation']))
                    for index in range(position['window'], len(windows)):
                        if self.stop_requested or previous_seconds+elapsed() > self.c['training_timeout_seconds']:
                            save()
                            raise InterruptedError('Stop requested' if self.stop_requested else '72-hour cumulative training limit')
                        started = time.monotonic()
                        record = optimizer_update(model, optimizer, scheduler, collator, dataset, windows[index],
                                                  stage, self.c, self.device)
                        self.sync()
                        position.update(epoch=epoch, window=index+1, updates=position['updates']+1)
                        if index+1 == len(windows):
                            position.update(epoch=epoch+1, window=0)
                        record.update(epoch=epoch, window=index, update=position['updates'],
                                      seconds=time.monotonic()-started, lr=optimizer.param_groups[0]['lr'],
                                      attempt_pid=os.getpid())
                        log.write(json.dumps(record)+'\n')
                        if position['updates'] == 1 or position['updates'] % self.c['checkpoint_every_updates'] == 0 or position['window'] == 0:
                            save()
                        if time.monotonic()-last_report >= self.c['status_interval_seconds'] or position['window'] == 0:
                            self.status('正式训练', training_stage=stage, epoch=epoch,
                                max_epochs=self.c[f'{stage}_epochs'], next_position=position.copy(),
                                expected_updates=total_updates(self.data, stage, self.c), last_loss=record['loss'],
                                training_seconds=previous_seconds+elapsed(), last_update_seconds=record['seconds'])
                            last_report = time.monotonic()
                    check_manifest(self.manifest)
                    if epoch % 5 == 0:
                        atomic_save(directory / f'epoch_{epoch:02d}.pt', dict(state_dict=cpu_tree(model.state_dict()),
                            epoch=epoch, stage=stage, input_sha256=self.manifest, formal_parent=True, domain=self.domain))
            assert position['updates'] == total_updates(self.data, stage, self.c)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            save('final.pt')
            summary = dict(state='COMPLETED', completed_at=now(), stage=stage, epochs=self.c[f'{stage}_epochs'],
                optimizer_updates=position['updates'], training_seconds=elapsed(), checkpoint_sha256=sha(directory/'final.pt'),
                input_sha256=self.manifest, fixed_final_epoch=True, development_labels_read=False, test_read=False)
            write(directory / 'summary.json', summary)
            stage_summaries[stage] = summary
            previous_seconds += summary['training_seconds']
            del model, optimizer, scheduler, collator
            gc.collect()
            torch.cuda.empty_cache()
        check_manifest(self.manifest)
        write(self.out / 'summary.json', dict(state='COMPLETED', domain=self.domain, completed_at=now(),
            stages=stage_summaries, input_sha256=self.manifest, new_recommendation_metrics=False,
            next_step='verify parents, generate new candidates and expert scores, fit gate/ranker on disjoint train holdouts'))
        self.status('GRAM和C1正式训练完成，待候选与排序验证', state='COMPLETED')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'cpu_check', 'profile', 'train'))
    parser.add_argument('--domain', choices=tuple(SPECS))
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = read(CONFIG)
    if args.action == 'prepare':
        prepare(config)
        return
    assert args.domain
    assert not args.resume or args.action == 'train', 'Only formal training supports explicit resume'
    output = ROOT / config['output'] / args.domain / args.action
    assert args.resume or not output.exists(), 'Existing attempt is preserved; do not overwrite its status'
    job = None
    try:
        job = Worker(config, args.domain, args.action, args.gpu_uuid, args.resume)
        def request_stop(signum, frame):
            if args.action == 'train':
                job.stop_requested = True
            else:
                raise InterruptedError(f'Signal {signum}')
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if args.action != 'train':
            def timeout(signum, frame):
                raise TimeoutError('Bounded check timeout')
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(config['profile_timeout_seconds'] if args.action == 'profile' else 900)
        getattr(job, args.action)()
    except BaseException as error:
        directory = ROOT / config['output'] / args.domain / args.action
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        write(directory / 'status.json', dict(state='FAILED', updated_at=now(), pid=os.getpid(),
            domain=args.domain, stage=args.action, error=repr(error), test_read=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
