"""Checkpoint-preserving switch to the authorized stage-18 screening budget."""

import argparse
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase18.core.screen_budget import BudgetSchedule, cohort_indices, optimizer_steps

ROOT = Path(__file__).resolve().parents[3]


def now():
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(path):
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


class Run:
    def __init__(self, spec, config_path):
        self.spec, self.config_path = spec, config_path
        self.root = ROOT / spec['run_directory']
        self.directory = self.root / spec['revision']
        self.context = dict(revision=spec['revision'], dataset=spec['dataset'], family=spec['family'],
                            active_config=config_path, stage_epochs=spec['additional_epochs'])

    def status(self, state, **values):
        # Write a fresh schema: a new epoch cannot inherit an old epoch's ETA.
        write_json(self.root / 'status.json', dict(state=state, updated_at=now(), pid=os.getpid(),
                                                  **self.context, **values))

    def event(self, event, **values):
        row = dict(time=now(), event=event, **{'revision': self.spec['revision'], **values})
        with (self.directory / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return False
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def process_command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()


def handoff(run):
    old = json.loads((run.root / 'status.json').read_text())
    pid = int(old['pid'])
    expected_module = ('s18_diff_gram' if run.spec['family'] == 'diff_gram' else 's18_diffgrm_native')
    if pid <= 1 or not alive(pid) or expected_module not in process_command(pid):
        raise RuntimeError('Old training process identity could not be verified')
    if str(run.spec['run_directory']) not in process_command(pid):
        raise RuntimeError('Old process output path mismatch')
    if Path(f'/proc/{pid}').stat().st_uid != os.getuid():
        raise RuntimeError('Refusing to signal another user process')
    target_epoch = int(old.get('epoch', 1))
    checkpoint_path = run.root / 'last.pt'
    started, stamp, saved_epoch = time.monotonic(), None, -1
    notice = dict(revision=run.spec['revision'], requested_at=now(),
                  state='WAITING_FOR_EPOCH_CHECKPOINT', target_epoch=target_epoch,
                  coordinator_pid=os.getpid(), config=run.config_path)
    write_json(run.root / 'pending_revision.json', notice)
    run.event('handoff_requested', old_pid=pid, target_epoch=target_epoch)
    while True:
        if not alive(pid):
            raise RuntimeError('Old process exited before the coordinated handoff')
        current = json.loads((run.root / 'status.json').read_text())
        if current.get('pending_revision') != notice:
            current['pending_revision'] = notice
            # The active trainer uses status.json.tmp; never share that temp file.
            temporary = run.root / f'.status.{run.spec["revision"]}.tmp'
            temporary.write_text(json.dumps(current, indent=2) + '\n')
            temporary.replace(run.root / 'status.json')
        if checkpoint_path.exists() and checkpoint_path.stat().st_mtime_ns != stamp:
            stamp = checkpoint_path.stat().st_mtime_ns
            saved = load_checkpoint(checkpoint_path)
            saved_epoch = int(saved['epoch'])
            optimizer_steps(saved['optimizer'])
            if not saved['model']:
                raise ValueError('Empty source checkpoint')
            del saved
        if saved_epoch >= target_epoch:
            break
        if time.monotonic() - started > 7200:
            raise TimeoutError('Checkpoint handoff exceeded two hours; old process left running')
        time.sleep(2)
    write_json(run.directory / 'status_before_switch.json', current)
    # SIGINT lets the old Python runner close its workers and release CUDA.
    os.kill(pid, signal.SIGINT)
    for _ in range(60):
        if not alive(pid):
            break
        time.sleep(1)
    if alive(pid):
        raise RuntimeError('Old process did not exit after SIGINT; no duplicate run started')
    stopped = json.loads((run.root / 'status.json').read_text())
    write_json(run.directory / 'status_after_interrupt.json', stopped)
    source = load_checkpoint(checkpoint_path)
    report = dict(old_pid=pid, checkpoint=str(checkpoint_path.relative_to(ROOT)),
                  checkpoint_sha256=sha256(checkpoint_path), checkpoint_epoch=source['epoch'],
                  optimizer_steps=optimizer_steps(source['optimizer']),
                  interruption_reason='User-authorized budget revision, not a scientific failure',
                  last_observed_status=current,
                  note='Resume at last complete epoch; any boundary batches after that checkpoint are replayed.')
    write_json(run.directory / 'handoff.json', report)
    run.event('handoff_completed', **{k: v for k, v in report.items() if k != 'last_observed_status'})
    run.context['inherited_epoch'] = int(source['epoch'])
    run.context['inherited_optimizer_steps'] = report['optimizer_steps']
    run.status('RESUMING', phase='restore_checkpoint', checkpoint=report['checkpoint'])
    return source


class FrozenSidTokenizer:
    vocab_size, sid_offset, mask_token = 1027, 3, -1

    def __init__(self, mapping):
        self.mapping = mapping

    def codebooks_to_item_id(self, code):
        items = self.mapping.get(tuple(code))
        return items[0] if items else None


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Backend:
    def __init__(self, run, source):
        self.family = run.spec['family']
        self.device = torch.device('cuda:0')
        if self.family == 'diff_gram':
            from experiment.phase18.protocol import s18_diff_gram as old
            self.old = old
            self.config = json.loads((run.root / 'manifest.json').read_text())['config']
            if source['config'] != self.config:
                raise AssertionError('GRAM checkpoint/config mismatch')
            old.seed_everything(self.config['seed'])
            catalog, self.training, self.validation, manifest, tokenizer, self.collator = old.load_inputs(self.config)
            self.model = old.build_model(self.config, catalog, tokenizer, self.device)
            self.model.set_backbone_frozen(False)
            self.optimizer = old.make_optimizer(self.model, self.config)
            self.decoder = old.CatalogDecoder(catalog, self.config)
            self.microbatch = run.spec['microbatch']
            self.effective = self.config['training']['effective_batch_size']
            self.eval_batch, self.workers = 1, self.config['training']['num_workers']
            self.clip = self.config['training']['gradient_clip']
            self.peak_lrs = [self.config['training']['new_learning_rate'],
                             self.config['training']['backbone_learning_rate']]
        else:
            from experiment.phase18.protocol import s18_diffgrm_native as old
            self.old = old
            self.config = json.loads((run.root / 'config.json').read_text())
            if source['config'] != self.config:
                raise AssertionError('Native checkpoint/config mismatch')
            old.seed_all(self.config['seed'])
            values = old.official_config(self.config, run.directory, 'cuda:0')
            values.update(warmup_steps=run.spec['warmup_steps'],
                          epochs=run.spec.get('total_epochs', run.spec['additional_epochs']))
            prepared = run.root / 'prepared'
            manifest = json.loads((prepared / 'manifest.json').read_text())
            for name, expected in manifest['prepared_sha256'].items():
                if sha256(prepared / name) != expected:
                    raise AssertionError(f'Prepared native artifact changed: {name}')
            codes = torch.from_numpy(np.load(prepared / 'item_codes.npy'))
            self.mapping = {tuple(row['sid']): row['items'] for row in json.loads((prepared / 'sid_items.json').read_text())}
            tokenizer = FrozenSidTokenizer(self.mapping)
            self.training = old.SidDataset(read_jsonl(prepared / 'train.jsonl'), codes, self.config['max_history'])
            self.validation = old.SidDataset(read_jsonl(prepared / 'validation.jsonl'), codes, self.config['max_history'])
            self.model = old.model_for(values, tokenizer).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=values['lr'], weight_decay=values['weight_decay'])
            self.microbatch, self.effective = self.config['microbatch'], self.config['effective_batch']
            self.eval_batch, self.workers, self.collator = self.config['evaluation_batch'], 0, None
            self.clip, self.peak_lrs = values['max_grad_norm'], [values['lr']]
            write_json(run.directory / 'official_resolved.json', {k: v for k, v in values.items() if k != 'accelerator'})
        if self.effective % self.microbatch:
            raise ValueError('Microbatch must divide effective batch')
        self.baseline = manifest['baseline_validation']
        self.model.load_state_dict(source['model'], strict=True)
        self.optimizer.load_state_dict(source['optimizer'])
        if optimizer_steps(self.optimizer.state_dict()) != optimizer_steps(source['optimizer']):
            raise AssertionError('Optimizer step did not restore')
        self.records = self.validation.records
        self.per_epoch = math.ceil(len(self.training) / self.effective)
        self.initial_steps = optimizer_steps(source['optimizer'])
        expected_steps = self.per_epoch * int(source['epoch'])
        if self.initial_steps != expected_steps:
            raise AssertionError(f'Checkpoint not at a complete epoch: {self.initial_steps} != {expected_steps}')
        self.inherited_epoch = int(source['epoch'])
        self.start_lrs = [float(group['lr']) for group in self.optimizer.param_groups]
        self.optimizer.zero_grad(set_to_none=True)

    def loader(self, dataset, batch_size, generator=None):
        return DataLoader(dataset, batch_size=batch_size, shuffle=generator is not None,
                          generator=generator, collate_fn=self.collator, num_workers=self.workers)

    def loss(self, batch):
        if self.family == 'diff_gram':
            return self.model(**self.old.model_batch(batch, self.device, True)).loss
        return self.model(batch).loss

    def size(self, batch):
        key = 'target_item_ids' if self.family == 'diff_gram' else 'target_item_id'
        return len(batch[key])

    @torch.no_grad()
    def predict(self, batch):
        if self.family == 'diff_gram':
            outputs = self.decoder.generate(self.model, self.old.model_batch(batch, self.device, False))
            return [(str(user), int(gold), items, scores) for user, gold, (items, scores)
                    in zip(batch['user_ids'], batch['target_item_ids'].tolist(), outputs)]
        raw = self.model.generate(self.old.inference_batch(batch), n_return_sequences=self.config['beam'], mode='confidence').cpu().tolist()
        return [(str(user), int(gold), self.old.resolve_items(row, self.mapping), None)
                for user, gold, row in zip(batch['user_id'], batch['target_item_id'].tolist(), raw)]


def save_checkpoint(path, backend, **extra):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(dict(model={k: v.detach().cpu() for k, v in backend.model.state_dict().items()},
                    optimizer=backend.optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), **extra), temporary)
    temporary.replace(path)


def restore_probe(backend, run, source):
    backend.model.eval()
    batch = next(iter(backend.loader(Subset(backend.validation, [0]), 1)))
    before = backend.predict(batch)
    path = run.directory / 'restored_start.pt'
    save_checkpoint(path, backend, epoch=backend.inherited_epoch, optimizer_steps=backend.initial_steps,
                    original_config=backend.config, revision_config=run.spec)
    saved = load_checkpoint(path)
    backend.model.load_state_dict(saved['model'], strict=True)
    backend.optimizer.load_state_dict(saved['optimizer'])
    if backend.predict(batch) != before:
        raise AssertionError('Restored checkpoint generation differs')
    if [g['lr'] for g in backend.optimizer.param_groups] != backend.start_lrs:
        raise AssertionError('Restored optimizer learning rate differs')
    backend.model.train()
    train_batch = next(iter(backend.loader(Subset(backend.training, list(range(backend.microbatch))), backend.microbatch)))
    loss = backend.loss(train_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite restored training loss')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(backend.model.parameters(), backend.clip, error_if_nonfinite=True)
    if not float(grad) > 0:
        raise AssertionError('Restored model has no gradient')
    backend.optimizer.zero_grad(set_to_none=True)
    # The probe does not advance Adam or train weights. Restore native RNG if available.
    if 'cpu_rng' in source:
        torch.set_rng_state(source['cpu_rng'])
        torch.cuda.set_rng_state_all(source['cuda_rng'])
    report = dict(state='PASSED', checkpoint_generation_equal=True, optimizer_steps=backend.initial_steps,
                  start_lrs=backend.start_lrs, loss=float(loss.detach()), grad_norm=float(grad),
                  optimizer_updates_during_probe=0, efficacy_evidence=False,
                  rng_policy='restored' if 'cpu_rng' in source else 'new_stage_fixed_seed; original GRAM checkpoint lacks RNG')
    write_json(run.directory / 'restore_check.json', report)
    run.event('restore_check_passed', **report)
    del loss, saved, train_batch, batch
    torch.cuda.empty_cache()


def metric_result(totals, count, baseline, full):
    result = dict(metrics={k: v / count for k, v in totals.items()}, examples=count)
    if full:
        result.update(historical_gram=baseline, delta={k: result['metrics'][k] - v for k, v in baseline.items()})
    else:
        result.update(historical_gram=None, delta=None, interpretation='Fixed subset trend; no comparison with full baseline')
    return result


@torch.no_grad()
def evaluate(backend, dataset, run, stage_epoch, full):
    backend.model.eval()
    torch.cuda.empty_cache()
    scope = 'full_validation' if full else 'trend_subset'
    name = f'{scope}_stage_epoch_{stage_epoch:02d}'
    path = run.directory / f'{name}.jsonl'
    partial = path.with_suffix('.partial.jsonl')
    if path.exists():
        raise FileExistsError(path)
    totals = {f'{metric}@{k}': 0.0 for metric in ('hit', 'ndcg') for k in (5, 10, 50)}
    seen = returned = short = 0
    start = last_update = time.monotonic()
    run.status('VALIDATING', phase=scope, stage_epoch=stage_epoch, epoch=backend.inherited_epoch + stage_epoch,
               validation_scope=scope, validation_total=len(dataset), evaluated=0)
    with partial.open('w') as handle:
        for batch in backend.loader(dataset, backend.eval_batch):
            rows = backend.predict(batch)
            if len(rows) != backend.size(batch):
                raise AssertionError('Prediction batch length mismatch')
            for user, gold, items, scores in rows:
                if len(items) > 50 or len(set(items)) != len(items):
                    raise AssertionError('Invalid actual-item ranking')
                if gold in items:
                    rank = items.index(gold) + 1
                    for k in (5, 10, 50):
                        if rank <= k:
                            totals[f'hit@{k}'] += 1
                            totals[f'ndcg@{k}'] += 1 / math.log2(rank + 1)
                handle.write(json.dumps(dict(split='validation', scope=scope, user_id=user,
                                             gold_item_id=gold, ranked_item_ids=items, sequence_scores=scores)) + '\n')
                seen += 1
                returned += len(items)
                short += len(items) < 50
            current = time.monotonic()
            if current - last_update >= 30:
                handle.flush()
                run.status('VALIDATING', phase=scope, stage_epoch=stage_epoch,
                           epoch=backend.inherited_epoch + stage_epoch, validation_scope=scope,
                           validation_total=len(dataset), evaluated=seen, validation_elapsed_seconds=current-start,
                           validation_eta_seconds=(len(dataset)-seen)*(current-start)/seen)
                last_update = current
    if seen != len(dataset):
        raise AssertionError('Incomplete validation')
    partial.replace(path)
    result = dict(scope=scope, stage_epoch=stage_epoch, **metric_result(totals, seen, backend.baseline, full),
                  mean_returned_items=returned/seen, fewer_than_50_fraction=short/seen,
                  seconds=time.monotonic()-start, predictions_sha256=sha256(path))
    write_json(run.directory / f'{name}.json', result)
    run.event('validation_completed', **result)
    run.context['latest_full_validation' if full else 'latest_trend_validation'] = result
    return result


def train_stage(backend, run, source):
    indices = cohort_indices(backend.records, run.spec['dataset'], backend.config['seed'], run.spec['trend_users'])
    cohort = Subset(backend.validation, indices)
    write_json(run.directory / 'trend_cohort.json', dict(seed=backend.config['seed'], dataset=run.spec['dataset'],
                selection='smallest sha256(seed|dataset|user_id), independent of targets and scores',
                user_ids=[str(backend.records[i]['user_id']) for i in indices], indices=indices,
                total_validation_users=len(backend.validation), no_full_baseline_comparison=True))
    total = backend.per_epoch * run.spec['additional_epochs']
    warmup = run.spec.get('warmup_steps') or max(1, int(total * .05))
    schedule = BudgetSchedule(backend.start_lrs, backend.peak_lrs, warmup, total)
    write_json(run.directory / 'schedule.json', dict(**schedule.state_dict(), steps_per_epoch=backend.per_epoch,
                inherited_steps=backend.initial_steps, first_update_lrs=schedule.at(0)))
    restore_probe(backend, run, source)
    del source
    steps, best, best_epoch = 0, -math.inf, None
    started = time.monotonic()
    schedule.apply(backend.optimizer, 0)
    for epoch in range(1, run.spec['additional_epochs'] + 1):
        backend.model.train()
        generator = torch.Generator().manual_seed(backend.config['seed'] + backend.inherited_epoch + epoch)
        loader = backend.loader(backend.training, backend.microbatch, generator)
        backend.optimizer.zero_grad(set_to_none=True)
        seen, group, total_loss = 0, 0, 0.0
        epoch_start = last_update = time.monotonic()
        run.status('TRAINING', phase='joint' if backend.family == 'diff_gram' else 'native_diffgrm',
                   stage_epoch=epoch, epoch=backend.inherited_epoch+epoch, seen=0, train_total=len(backend.training),
                   optimizer_steps=backend.initial_steps+steps, stage_optimizer_steps=steps,
                   learning_rates=[g['lr'] for g in backend.optimizer.param_groups], epoch_eta_seconds=None)
        for index, batch in enumerate(loader):
            loss = backend.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite screening loss')
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
                steps += 1
                schedule.apply(backend.optimizer, steps)
                backend.optimizer.zero_grad(set_to_none=True)
                group = 0
            current = time.monotonic()
            if current-last_update >= 30:
                progress = dict(stage_epoch=epoch, epoch=backend.inherited_epoch+epoch, seen=seen,
                                train_total=len(backend.training), optimizer_steps=backend.initial_steps+steps,
                                stage_optimizer_steps=steps, loss=total_loss/seen, epoch_elapsed_seconds=current-epoch_start,
                                epoch_eta_seconds=(len(backend.training)-seen)*(current-epoch_start)/seen,
                                learning_rates=[g['lr'] for g in backend.optimizer.param_groups])
                run.status('TRAINING', phase='joint' if backend.family == 'diff_gram' else 'native_diffgrm', **progress)
                run.event('training_progress', **progress)
                last_update = current
        if group or steps != epoch * backend.per_epoch:
            raise AssertionError('Accumulation tail or update count mismatch')
        save_checkpoint(run.directory / 'last.pt', backend, epoch=backend.inherited_epoch+epoch,
                        stage_epoch=epoch, optimizer_steps=backend.initial_steps+steps, stage_optimizer_steps=steps,
                        schedule=schedule.state_dict(), original_config=backend.config, revision_config=run.spec)
        run.event('epoch_completed', stage_epoch=epoch, epoch=backend.inherited_epoch+epoch,
                  optimizer_steps=backend.initial_steps+steps, loss=total_loss/seen, seconds=time.monotonic()-epoch_start)
        if epoch % run.spec['trend_interval'] == 0 or epoch == run.spec['additional_epochs']:
            result = evaluate(backend, cohort, run, epoch, full=False)
            if result['metrics']['ndcg@10'] > best:
                best, best_epoch = result['metrics']['ndcg@10'], epoch
                save_checkpoint(run.directory / 'best_trend.pt', backend, epoch=backend.inherited_epoch+epoch,
                                stage_epoch=epoch, optimizer_steps=backend.initial_steps+steps,
                                original_config=backend.config, revision_config=run.spec, selection=result)
    selected = load_checkpoint(run.directory / 'best_trend.pt')
    backend.model.load_state_dict(selected['model'], strict=True)
    del selected
    backend.optimizer.zero_grad(set_to_none=True)
    full = evaluate(backend, backend.validation, run, best_epoch, full=True)
    result = dict(state='COMPLETED', revision=run.spec['revision'], completion_scope='screening_stage',
                  automatic_additional_training=False, direction_rejected=False,
                  selected_stage_epoch=best_epoch, selected_absolute_epoch=backend.inherited_epoch+best_epoch,
                  additional_epochs=run.spec['additional_epochs'], optimizer_steps=backend.initial_steps+steps,
                  full_validation=full, seconds=time.monotonic()-started,
                  interpretation='Exploratory continuation with revised training budget; not a matched-budget structural attribution')
    write_json(run.directory / 'result.json', result)
    write_json(run.root / 'result.json', result)
    run.status('COMPLETED', phase='screening_stage_complete', result=result)
    run.event('screening_stage_completed', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    run = Run(spec, args.config)
    lock = (run.root / f'.{spec["revision"]}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if run.directory.exists():
        raise FileExistsError(f'Refusing to overwrite revision {run.directory}')
    run.directory.mkdir()
    write_json(run.directory / 'config.json', spec)
    paths = [Path(__file__).resolve(), ROOT / args.config,
             ROOT / 'experiment/phase18/core/screen_budget.py', ROOT / 'experiment/phase18/run_stage18_screen_resume.sh']
    for path in paths:
        dest = run.directory / 'sources' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(run.directory / 'source_manifest.json', {str(p.relative_to(ROOT)): sha256(p) for p in paths})
    switched = False
    try:
        source = handoff(run)
        switched = True
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable for resume')
        backend = Backend(run, source)
        train_stage(backend, run, source)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), switched=switched)
        write_json(run.directory / 'failure.json', failure)
        if switched:
            run.status('FAILED', **failure)
        raise


if __name__ == '__main__':
    main()
