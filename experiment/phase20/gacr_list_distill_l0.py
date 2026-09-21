"""Bounded L0: two C1 restarts, first32 of the original171-step schedule."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from experiment.phase20.gacr_list_distill import Work, tensors_hash
from experiment.phase20.gacr_list_distill_core import sample_backward
from experiment.phase20.gacr_list_distill_diagnostic import Diagnosis, read, write
from experiment.phase20.analyze_gacr_list_distill_l0 import ARMS, analyze, render_report
from experiment.phase18.protocol.s18_diff_gram import seed_everything
from experiment.phase18.core.diff_data import sha256

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'artifacts/phase20/gacr_list_distill_toys_l0_v1'
Q0 = ROOT/'artifacts/phase20/gacr_list_distill_toys_q0_memory_v3'
SOURCE = ROOT/'experiment/phase20/gacr_list_distill_toys_recovery_v3.json'


def optimizer_and_scheduler(parameters, learning_rate):
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01, eps=1e-8)
    return optimizer, get_linear_schedule_with_warmup(optimizer, 9, 171)


def prepare(gpus):
    assert len(gpus) == 2 and len(set(gpus)) == 2
    assert read(Q0/'summary.json')['state'] == 'COMPLETED'
    original = read(SOURCE)
    for path, expected in {**original['source_sha256'], **original['input_sha256']}.items():
        assert sha256(ROOT/path) == expected, ('changed_original', path)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Q0/'panel.json', OUTPUT/'panel.json')
    panel = read(OUTPUT/'panel.json')
    run = ROOT/original['output']
    order = np.load(run/'cache/training_order.npy')
    with np.load(run/'cache/distillation_cache.npz') as z:
        assert sorted(order.tolist()) == np.flatnonzero(z['fit']).tolist()
        short = order[:4096]
        assert len(short) == len(set(short.tolist())) == 4096 and z['fit'][short].all()
        covered = int((z['positions'][short] >= 0).sum())
    np.save(OUTPUT/'training_order.npy', short)
    records = [json.loads(row) for row in (ROOT/original['train_candidate_jsonl']).read_text().splitlines()]
    seen_users = {records[i]['user_id'] for i in short}
    train_set = set(short.tolist())
    subset = dict(examples=4096, optimizer_steps=32, nonzero_lr_steps=31, covered_prefixes=covered,
                  training_users=len(seen_users), original_order_sha256=sha256(run/'cache/training_order.npy'),
                  subset_sha256=sha256(OUTPUT/'training_order.npy'), full_fit_covered_normalizer=21839/12399,
                  fit_panel_training_prefix_overlap=len(train_set & set(panel['fit_indices'])),
                  fit_panel_training_user_overlap=sum(records[i]['user_id'] in seen_users for i in panel['fit_indices']),
                  calibration_training_prefix_overlap=len(train_set & set(panel['calibration_indices'])),
                  calibration_training_user_overlap=len(seen_users & {records[i]['user_id'] for i in panel['calibration_indices']}),
                  start_from_original_C1=True, test_read=False)
    assert subset['calibration_training_prefix_overlap'] == subset['calibration_training_user_overlap'] == 0
    write(OUTPUT/'training_subset.json', subset)
    frozen = {}
    paths = [SOURCE, Q0/'config.json', Q0/'summary.json', Q0/'C1.scores.npz', OUTPUT/'panel.json',
             OUTPUT/'training_order.npy', OUTPUT/'training_subset.json']
    paths += [ROOT/'experiment/phase20'/name for name in
              ('gacr_list_distill_l0.py', 'analyze_gacr_list_distill_l0.py', 'test_gacr_list_distill_l0.py',
               'run_gacr_list_distill_l0.sh', 'gacr_list_distill_diagnostic.py', 'analyze_gacr_list_distill_diagnostic.py')]
    for path in paths:
        frozen[str(path.relative_to(ROOT))] = sha256(path)
    config = dict(id='stage20_gacr_list_distill_toys_l0_v1', source_config=str(SOURCE.relative_to(ROOT)),
                  output=str(OUTPUT.relative_to(ROOT)), source_output=original['output'],
                  report='report/第二十阶段/Stage20_Toys_L0短程学习率筛查_20260916.md',
                  distillation_cache=str((run/'cache/distillation_cache.npz').relative_to(ROOT)),
                  source_q0=str(Q0.relative_to(ROOT)), gpus=dict(zip(ARMS, gpus)),
                  learning_rate=1e-5, optimizer_steps=32, examples=4096, effective_batch_size=128,
                  scheduler_steps=171, warmup_steps=9, covered_normalizer=21839/12399,
                  minimum_free_mib=15000, cuda_memory_fraction=.25, cpu_threads=4,
                  resource_wait_seconds=900, compute_budget_seconds=5400, profile_budget_seconds=600,
                  profile_safety_factor=1.2, profile_fixed_allowance_seconds=180, automatic_retry=False,
                  automatic_next_experiment=False, checkpoint_selection='fixed_step32', test_read=False,
                  gate={'comparison': 'true_teacher_step32 minus CE_anchor_step32',
                        'teacher_kl': 'strictly lower prefix mean in BOTH fit_covered and calibration_covered',
                        'promoted_demoted_satisfaction': 'strictly higher prefix mean in BOTH fit_covered and calibration_covered; empty pairs excluded with denominator reported',
                        'ndcg10': 'strictly higher equal-user mean in calibration_all',
                        'logic': 'all five checks; screening only, no automatic full run'}, frozen_hashes=frozen)
    write(OUTPUT/'config.json', config)
    (OUTPUT/'workers').mkdir()
    for arm in ARMS:
        worker = copy.deepcopy(original)
        worker.update(id=config['id']+'_'+arm, output=str((OUTPUT/'work').relative_to(ROOT)),
                      config_path=str((OUTPUT/'workers'/f'{arm}.json').relative_to(ROOT)),
                      learning_rate=1e-5, gpu_uuid=config['gpus'][arm], minimum_free_mib=15000,
                      cuda_memory_fraction=.25, optimizer_steps_per_arm=32, scheduler_steps=171, warmup_steps=9,
                      train_budget_seconds=5400, resource_wait_seconds=900, report=config['report'],
                      scope='L0 fixed32 calls/4096 FIT; score Q0 panel only at0/32; original171 scheduler',
                      resource_revision={'L0': True, 'allocator_cap': .25, 'profile_includes_AdamW': True})
        worker['source_sha256'].update({p:h for p,h in frozen.items() if p.endswith(('.py', '.sh'))})
        worker['input_sha256'].update({p:h for p,h in frozen.items() if not p.endswith(('.py', '.sh'))})
        worker['input_sha256'][str((OUTPUT/'config.json').relative_to(ROOT))] = sha256(OUTPUT/'config.json')
        write(OUTPUT/'workers'/f'{arm}.json', worker)
    render_report(OUTPUT, 'PREPARED', {})
    print(json.dumps({'state': 'PREPARED', 'output': str(OUTPUT), 'training_subset': subset}))


def timeout_handler(signum, frame):
    raise TimeoutError('L0 compute budget exhausted' if signum == signal.SIGALRM else 'L0 received termination signal')


class Worker:
    score = Diagnosis.score

    def __init__(self, arm):
        self.arm, self.config = arm, read(OUTPUT/'config.json')
        self.path = OUTPUT/arm
        self.path.mkdir(exist_ok=False)
        self.started = time.monotonic()
        self.lock, self.finished = threading.Lock(), threading.Event()
        self.info = dict(state='RUNNING', stage='initialization', arm=arm, pid=os.getpid(),
                         gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), optimizer_steps=0,
                         engineering_optimizer_steps=0, test_read=False)
        self.compute_started = None
        self.status()
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def status(self, **values):
        with self.lock:
            self.info.update(values)
            self.info.update(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), elapsed_seconds=round(time.monotonic()-self.started, 2))
            write(self.path/'status.json', self.info)

    def heartbeat(self):
        while not self.finished.wait(30):
            self.status()

    def verify(self):
        for path, expected in self.config['frozen_hashes'].items():
            assert sha256(ROOT/path) == expected, ('changed_L0_source_input', path)

    def setup(self):
        self.verify()
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == self.config['gpus'][self.arm]
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        self.status(stage='post_cuda_resource_admission')
        stable = 0
        while True:
            free, total = torch.cuda.mem_get_info()
            self.status(free_mib=free/2**20, gpu_name=torch.cuda.get_device_name(0))
            stable = stable+1 if free/2**20 >= self.config['minimum_free_mib'] else 0
            if stable >= 2:
                break
            if time.monotonic()-self.started >= self.config['resource_wait_seconds']:
                raise TimeoutError('L0 15-minute resource wait exhausted before loading model')
            time.sleep(10)
        torch.cuda.set_per_process_memory_fraction(self.config['cuda_memory_fraction'], 0)
        self.compute_started = time.monotonic()
        signal.alarm(self.config['compute_budget_seconds'])
        seed_everything(2023)
        cfg = read(OUTPUT/'workers'/f'{self.arm}.json')
        self.status(stage='load_C1_and_verify_original_inputs')
        self.work = Work(cfg, 'train', self.arm)
        self.work.verify(); self.work.snapshot(); self.work.data(); self.work.load_model(); self.work.snapshot()
        assert not self.work.model.training and self.work.settings['candidate_chunk'] == 4
        assert self.work.settings['covered_normalizer'] == self.config['covered_normalizer']
        self.parameters = [p for p in self.work.model.parameters() if p.requires_grad]
        self.initial = {n:p.detach().cpu().clone() for n,p in self.work.model.named_parameters() if p.requires_grad}
        self.initial_hash = tensors_hash(self.initial.items())
        self.panel = read(OUTPUT/'panel.json')
        with np.load(ROOT/self.config['distillation_cache']) as z:
            self.cache = {k:z[k] for k in z.files}
        for key in ('candidate_ids', 'positions', 'fit'):
            assert np.array_equal(self.cache[key], self.work.arrays[key])
        self.order = np.load(OUTPUT/'training_order.npy')
        assert np.array_equal(self.order, np.load(ROOT/self.config['source_output']/'cache/training_order.npy')[:4096])

    def backward(self, i, divisor=1):
        return sample_backward(self.work.model, self.work.batch(i), self.work.decoder.paths,
                               self.cache['candidate_ids'][i].tolist(), self.cache['parent_scores'][i],
                               self.cache['true_probability'][i], self.cache['positions'][i] >= 0,
                               self.arm, self.work.settings, self.work.device, divisor=divisor)

    def profile(self):
        self.status(stage='profile_scoring_backward_and_AdamW')
        start = time.monotonic()
        timings, scoring, gradients = [], [], []
        for i in self.panel['profile_indices']:
            self.work.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t = time.monotonic()
            scores, ce = self.score(i)
            torch.cuda.synchronize(); scoring.append(time.monotonic()-t)
            np.testing.assert_allclose(scores, self.cache['parent_scores'][i], rtol=0, atol=1e-6)
            torch.cuda.synchronize(); t = time.monotonic()
            self.backward(i)
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True))
            torch.cuda.synchronize(); timings.append(time.monotonic()-t); gradients.append(norm)
            assert norm > 0
            self.work.frozen_unchanged()
            if time.monotonic()-start > self.config['profile_budget_seconds']:
                raise TimeoutError('L0 profile exceeded ten-minute budget')
        # Real AdamW state allocation and serialization, discarded before formal training.
        optimizer = torch.optim.AdamW(self.parameters, lr=1e-5, weight_decay=.01, eps=1e-8)
        optimizer.step(); self.work.engineering_steps = 1
        assert tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad) != self.initial_hash
        checkpoint = self.path/'discarded_engineering_update.pt'
        self.work.save_parameters(checkpoint, discarded_engineering_update=True)
        engineering_hash = tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad)
        self.work.load_parameters(checkpoint)
        assert tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad) == engineering_hash
        with torch.no_grad():
            for n, p in self.work.model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.initial[n])
        del optimizer
        self.work.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        assert tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad) == self.initial_hash
        self.work.frozen_unchanged()
        elapsed = time.monotonic()-self.compute_started
        estimate = self.config['profile_safety_factor']*(np.mean(timings)*4096 + np.mean(scoring)*1024) + self.config['profile_fixed_allowance_seconds']
        result = dict(state='PASSED' if elapsed+estimate <= self.config['compute_budget_seconds'] else 'OVER_BUDGET',
                      seconds_per_training_example=timings, seconds_per_scoring_example=scoring, gradient_norms=gradients,
                      profile_indices=self.panel['profile_indices'], estimated_remaining_seconds=float(estimate),
                      estimated_total_compute_seconds=float(elapsed+estimate),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      engineering_optimizer_steps=1, formal_optimizer_steps=0, restored_C1_sha256=self.initial_hash,
                      original_parent_score_atol=1e-6, save_load_exact=True, frozen_parameters_unchanged=True)
        write(self.path/'profile.json', result)
        self.status(stage='waiting_both_profiles', engineering_optimizer_steps=1, profile=result)
        if result['state'] != 'PASSED':
            raise TimeoutError('L0 profile predicts compute budget overrun; formal training not admitted')
        while not (OUTPUT/'training_admission.json').exists():
            time.sleep(2)
        admission = read(OUTPUT/'training_admission.json')
        assert admission['state'] == 'PASSED'
        assert admission['profile_sha256'][self.arm] == sha256(self.path/'profile.json')
        assert time.monotonic()-self.compute_started+estimate <= self.config['compute_budget_seconds'], 'L0 profile barrier consumed budget'

    def score_panel(self, step):
        self.status(stage=f'fixed_panel_step{step}', scored_prefixes=0)
        indices = self.panel['fit_indices'] + self.panel['calibration_indices']
        scores, ces = [], []
        with (self.path/f'step{step}.scores.jsonl').open('x') as handle:
            for j, i in enumerate(indices):
                row, ce = self.score(i)
                scores.append(row); ces.append(ce)
                handle.write(json.dumps(dict(cache_index=i, scores=row.tolist(), ce=ce))+'\n')
                if (j+1) % 32 == 0:
                    handle.flush(); self.status(scored_prefixes=j+1)
        scores, ces = np.asarray(scores), np.asarray(ces)
        np.savez(self.path/f'step{step}.scores.npz', indices=indices, scores=scores, ce=ces)
        if step == 0:
            np.testing.assert_allclose(scores, self.cache['parent_scores'][indices], rtol=0, atol=1e-6)
            with np.load(Q0/'C1.scores.npz') as q0:
                assert np.array_equal(q0['indices'], indices)
                np.testing.assert_allclose(scores, q0['scores'], rtol=0, atol=1e-6)
                np.testing.assert_allclose(ces, q0['ce'], rtol=0, atol=1e-6)
            write(self.path/'step0_parity.json', dict(state='PASSED', prefixes=512,
                  parent_max_abs_error=float(np.max(np.abs(scores-self.cache['parent_scores'][indices]))), atol=1e-6))
        self.work.frozen_unchanged()

    def train(self):
        # Start with exactly C1 and a fresh optimizer after all engineering work.
        seed_everything(2023)
        assert tensors_hash((n,p) for n,p in self.work.model.named_parameters() if p.requires_grad) == self.initial_hash
        optimizer, scheduler = optimizer_and_scheduler(self.parameters, self.config['learning_rate'])
        assert not optimizer.state and optimizer.param_groups[0]['lr'] == 0
        self.status(stage='decoder_training', examples=0, training_started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        started = time.monotonic(); nonzero = 0
        for start in range(0, 4096, 128):
            optimizer.zero_grad(set_to_none=True)
            batch_indices = self.order[start:start+128]
            assert len(batch_indices) == 128
            totals = {k:0. for k in ('loss', 'ce', 'anchor_kl', 'teacher_kl', 'weighted_teacher', 'covered')}
            for j, i in enumerate(batch_indices):
                values = self.backward(i, divisor=128)
                assert all(np.isfinite(v) for v in values.values())
                for key, value in values.items():
                    totals[key] += value/128
                if (j+1) % 16 == 0:
                    done = start+j+1
                    self.status(examples=done, training_eta_seconds=(time.monotonic()-started)*(4096-done)/done)
            gradient = float(torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True))
            lr = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step(); self.work.steps += 1
            nonzero += int(lr > 0)
            point = dict(step=self.work.steps, examples=start+128, gradient_norm_before_clip=gradient,
                         learning_rate=lr, elapsed_training_seconds=time.monotonic()-started, **totals)
            with (self.path/'training_curve.jsonl').open('a') as handle:
                handle.write(json.dumps(point)+'\n')
            print(json.dumps(dict(arm=self.arm, **point)), flush=True)
            self.status(optimizer_steps=self.work.steps, nonzero_lr_steps=nonzero, learning_rate=lr)
        assert self.work.steps == 32 and nonzero == 31
        self.work.frozen_unchanged(); self.verify(); self.work.verify()
        self.work.save_parameters(self.path/'last.pt', scheduler_horizon=171, warmup_steps=9,
                                  training_order_sha256=sha256(OUTPUT/'training_order.npy'))
        state = optimizer.state_dict()
        for item in state['state'].values():
            for k, v in item.items():
                if torch.is_tensor(v):
                    item[k] = v.detach().cpu()
        torch.save(dict(optimizer=state, scheduler=scheduler.state_dict(), step=32,
                        python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all()), self.path/'optimizer.pt')
        del optimizer, scheduler, state
        self.work.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

    def run(self):
        try:
            self.setup(); self.profile(); self.score_panel(0); self.train(); self.score_panel(32)
            self.verify(); self.work.verify(); self.work.frozen_unchanged()
            result = dict(state='COMPLETED', arm=self.arm, optimizer_steps=32, nonzero_lr_steps=31,
                          engineering_optimizer_steps=1, examples=4096, gpu_uuid=self.info['gpu_uuid'],
                          elapsed_seconds=time.monotonic()-self.started,
                          compute_seconds=time.monotonic()-self.compute_started,
                          checkpoint_sha256=sha256(self.path/'last.pt'),
                          step0_scores_sha256=sha256(self.path/'step0.scores.npz'),
                          step32_scores_sha256=sha256(self.path/'step32.scores.npz'),
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                          frozen_parameters_unchanged=True, test_read=False)
            write(self.path/'summary.json', result)
            self.status(state='COMPLETED', stage='finished')
        except BaseException as error:
            traceback.print_exc()
            self.status(state='FAILED', stage='stopped', error=repr(error))
            raise
        finally:
            signal.alarm(0); self.finished.set()


def controller():
    config = read(OUTPUT/'config.json')
    # Exclusive marker prevents even simultaneous duplicate launches.
    with (OUTPUT/'controller_started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')), handle)
    children, handles, statuses = {}, [], {}
    started = time.monotonic()
    def update(state, error=None, analysis=None):
        for arm in ARMS:
            path = OUTPUT/arm/'status.json'
            if path.exists():
                statuses[arm] = read(path)
        result = dict(state=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      elapsed_seconds=time.monotonic()-started, arms=statuses,
                      children={a:p.pid for a,p in children.items()}, test_read=False)
        if error:
            result['error'] = error
        if analysis:
            result['gate'] = analysis['gate']; result['recommendation'] = analysis['recommendation']
        write(OUTPUT/'status.json', result)
        render_report(OUTPUT, state, statuses, analysis, error)
        return result
    try:
        for path, expected in config['frozen_hashes'].items():
            assert sha256(ROOT/path) == expected, ('changed_before_launch', path)
        for arm in ARMS:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=config['gpus'][arm], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            handle = (OUTPUT/f'{arm}.log').open('x'); handles.append(handle)
            children[arm] = subprocess.Popen([sys.executable, '-u', '-m', 'experiment.phase20.gacr_list_distill_l0', '--worker', arm],
                                              cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            update('RUNNING')
            for arm, process in children.items():
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'{arm} exited {process.returncode}: {statuses.get(arm, {}).get("error", "see worker log")}')
            if not (OUTPUT/'training_admission.json').exists() and all((OUTPUT/a/'profile.json').exists() for a in ARMS):
                profiles = {a:read(OUTPUT/a/'profile.json') for a in ARMS}
                assert all(p['state'] == 'PASSED' for p in profiles.values()), profiles
                write(OUTPUT/'training_admission.json', dict(state='PASSED', admitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      profile_sha256={a:sha256(OUTPUT/a/'profile.json') for a in ARMS}, profiles=profiles))
            if all(p.poll() == 0 for p in children.values()):
                analysis = analyze(OUTPUT)
                summary = update('COMPLETED', analysis=analysis)
                write(OUTPUT/'summary.json', summary)
                break
            if time.monotonic()-started > 6600:
                raise TimeoutError('L0 controller overall deadline reached')
            time.sleep(15)
    except BaseException as error:
        traceback.print_exc()
        # Terminate only the child processes created by this controller.
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        for process in children.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        update('FAILED', error=repr(error))
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--gpus', nargs=2)
    parser.add_argument('--worker', choices=ARMS)
    args = parser.parse_args()
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, timeout_handler)
    if args.prepare:
        prepare(args.gpus)
    elif args.worker:
        Worker(args.worker).run()
    else:
        controller()


if __name__ == '__main__':
    main()
