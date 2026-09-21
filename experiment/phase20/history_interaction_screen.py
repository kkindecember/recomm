"""Bounded four-arm screen with reused, independently checked frozen queries."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.conditional_reference_screen import read, rows, sha, write, metrics, paired
from experiment.phase20.conditional_reference_core import ranks
from experiment.phase20.gacr_fusion_core import standardize_tensor
from experiment.phase20.history_interaction_core import align_text, history_inputs, build_model, global_group_weights, dense_pairwise_losses
from experiment.phase20.personal_popularity_core import reconstruct_training

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiment/phase20/history_interaction_toys_v2.json'


class Worker:
    def __init__(self, config_path, mode):
        self.config_path, self.mode = Path(config_path), mode
        self.c = read(config_path)
        self.config_hash = sha(config_path)
        self.root = ROOT / self.c['output']
        self.path = self.root / mode
        self.path.mkdir(parents=True, exist_ok=False)
        self.started, self.steps = time.monotonic(), 0
        torch.set_num_threads(self.c['cpu_threads'])
        torch.set_num_interop_threads(1)
        torch.manual_seed(self.c['seed'])
        np.random.seed(self.c['seed'])
        for name, digest in {**self.c['input_sha256'], **self.c['source_sha256']}.items():
            assert sha(ROOT / name) == digest, ('Frozen source/input changed', name)
        write(self.path / 'protocol.json', self.c)
        self.status('initializing')

    def status(self, stage, state='RUNNING', **extra):
        elapsed = time.monotonic() - self.started
        if state == 'RUNNING' and elapsed > self.c['budgets_seconds'][self.mode]:
            raise TimeoutError('Predeclared worker budget exceeded')
        value = {'state': state, 'stage': stage, 'pid': os.getpid(),
                 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 'elapsed_seconds': elapsed, 'optimizer_steps': self.steps,
                 'config_sha256': self.config_hash, **extra}
        write(self.path / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def finish(self, summary):
        summary = {**summary, 'config_sha256': self.config_hash,
                   'elapsed_seconds': time.monotonic()-self.started,
                   'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   'optimizer_steps': self.steps, 'test_read': False}
        write(self.path / 'summary.json', summary)
        write(self.path / 'manifest.json', {p.name: sha(p) for p in sorted(self.path.iterdir())
                                          if p.is_file() and p.name not in ('status.json', 'manifest.json')})
        self.status('finished', 'COMPLETED')

    def verify_mode(self, mode):
        path = self.root / mode
        assert read(path / 'status.json')['state'] == 'COMPLETED'
        assert read(path / 'summary.json')['config_sha256'] == self.config_hash
        for name, digest in read(path / 'manifest.json').items():
            assert sha(path / name) == digest, (mode, name)

    def prepare(self):
        d = self.c['data']
        original = read(ROOT / d['reference_protocol'])
        # Cached B_plain contains only existing scores. No old experimental
        # semantic-reference features or fitted correction are imported.
        manifest = read(ROOT / d['reference_manifest'])
        assert manifest['feature_cache.npz'] == sha(ROOT / d['feature_cache'])
        for name, digest in original['input_sha256'].items():
            assert sha(ROOT / name) == digest, name
        self.status('reconstruct_visible_histories_and_cache_lineage')
        with np.load(ROOT / d['feature_cache']) as z:
            a = {k: z[k] for k in ('base_scores', 'candidate_ids', 'positions', 'fit', 'head', 'users', 'ntrain')}
            a['features'] = z['B_plain']
        n = int(a['ntrain'])
        assert n == 22095 and len(a['users']) == n+19412
        catalog = read(ROOT / d['catalog'])['items']
        train = list(rows(ROOT / d['train']))
        _, sequences, _, _ = reconstruct_training(train)
        frequencies = np.bincount([i for h in sequences.values() for i in h], minlength=len(catalog))
        histories, fit_users, cal_users = [], set(), set()
        with np.load(ROOT / d['training_cache']) as z:
            source_cache = {k: z[k] for k in ('fit', 'head', 'candidate_ids')}
        np.testing.assert_array_equal(a['fit'], source_cache['fit'])
        np.testing.assert_array_equal(a['head'], source_cache['head'])
        for j, record in enumerate(rows(ROOT / d['training_candidates'])):
            source = train[record['train_index']]
            assert source['user_id'] == record['user_id'] == a['users'][j]
            assert source['history'] == record['history'] and source['target'] == record['target']
            ids = a['candidate_ids'][j].tolist()
            assert set(ids) == set(record['union_ids'][:50]) == set(source_cache['candidate_ids'][j])
            assert len(set(ids)) == 50
            p = ids.index(source['target']) if source['target'] in ids else -1
            assert p == a['positions'][j]
            assert (record['partition'] == 'fit') == bool(a['fit'][j])
            histories.append(source['history'])
            (fit_users if a['fit'][j] else cal_users).add(source['user_id'])
        assert len(histories) == n and fit_users.isdisjoint(cal_users)
        val = list(rows(ROOT / d['validation']))
        assert len(val) == 19412 and len({r['user_id'] for r in val}) == 19412
        for j, record in enumerate(val, n):
            assert record['user_id'] == a['users'][j]
            assert record['history'] == sequences[record['user_id']][-20:]
            ids = a['candidate_ids'][j].tolist()
            assert len(set(ids)) == 50
            p = ids.index(record['target']) if record['target'] in ids else -1
            assert p == a['positions'][j]
            histories.append(record['history'])
        with (ROOT / d['validation_ranks']).open() as handle:
            known = {r['user_id']: int(r['B1']) for r in csv.DictReader(handle, delimiter='\t')}
        a['known_ranks'] = np.asarray([known[r['user_id']] for r in val])
        np.testing.assert_array_equal(ranks(a['base_scores'][n:], a['positions'][n:]), a['known_ranks'])
        a['histories'], a['history_attributes'] = history_inputs(histories, frequencies)
        a['target_frequencies'] = frequencies[[r['target'] for r in val]]
        a['frequencies'] = frequencies
        assert np.isfinite(a['features']).all() and np.isfinite(a['base_scores']).all()
        assert a['candidate_ids'].min() > 0 and a['candidate_ids'].max() < len(catalog)
        text = torch.load(ROOT / d['text_embeddings'], map_location='cpu')
        aligned = align_text(catalog, text['item_ids'], text['embeddings'])
        checkpoint = torch.load(ROOT / d['cf_checkpoint'], map_location='cpu')
        cf = checkpoint['model_state_dict']['item_embedding.weight'].detach().clone()
        assert cf.shape == (len(catalog), self.c['cf_dim'])
        assert aligned.shape == (len(catalog), self.c['text_dim'])
        cf = F.normalize(cf, dim=-1)
        with np.load(ROOT / d['reference_item_vectors']) as z:
            np.testing.assert_allclose(cf.numpy(), z['item_vectors'], rtol=0, atol=1e-6)
        # Unit-normalized vectors are centered/scaled without trainable affine
        # terms to avoid an artificially tiny initial dot-product attention.
        assets = {'text': F.layer_norm(aligned, (aligned.shape[-1],)),
                  'cf': F.layer_norm(cf, (cf.shape[-1],))}
        assets['text'][0].zero_(); assets['cf'][0].zero_()
        assert all(torch.isfinite(x).all() for x in assets.values())
        torch.save(assets, self.path / 'assets.pt')
        np.savez(self.path / 'queries.npz', **a)
        parameters, initial = {}, {}
        for arm in self.c['arms']:
            model = build_model(self.c, arm)
            parameters[arm] = sum(x.numel() for x in model.parameters())
            torch.save(model.state_dict(), self.path / f'{arm}.initial.pt')
            initial[arm] = model.state_dict()
        common = set(initial['C_attention']) & set(initial['D_joint'])
        assert all(torch.equal(initial['C_attention'][k], initial['D_joint'][k]) for k in common)
        assert parameters['C_attention'] >= parameters['D_joint']
        valid = a['fit'] & (a['positions'][:n] >= 0)
        self.finish({'state': 'PASSED', 'training_rows': n, 'fit_rows': int(a['fit'].sum()),
                     'fit_covered': int(valid.sum()), 'fit_covered_head': int((valid & a['head']).sum()),
                     'calibration_rows': int((~a['fit']).sum()), 'validation_users': len(val),
                     'parameters': parameters, 'common_initial_tensors_equal': len(common),
                     'fit_calibration_users_disjoint': True, 'all_query_targets_and_candidates_checked': True,
                     'frozen_B1_ranks_exact': True, 'embedding_ASIN_mapping_complete': True,
                     'embedding_row_order_matches_catalog': text['item_ids'] == catalog[1:],
                     'cf_embedding_matches_verified_reference': True,
                     'item_tables_frozen': True, 'preparation_has_new_model_validation': False})

    def load_tensors(self, device):
        self.verify_mode('prepare')
        with np.load(self.root / 'prepare/queries.npz') as z:
            self.a = {k: z[k] for k in z.files}
        self.n = int(self.a['ntrain'])
        self.t = {k: torch.as_tensor(self.a[k], device=device) for k in
                  ('features', 'candidate_ids', 'histories', 'history_attributes', 'base_scores', 'positions')}
        assets = torch.load(self.root / 'prepare/assets.pt', map_location=device)
        self.text, self.cf = assets['text'], assets['cf']
        assert not self.text.requires_grad and not self.cf.requires_grad
        self.fit_indices = np.flatnonzero(self.a['fit'] & (self.a['positions'][:self.n] >= 0))
        w = np.zeros(self.n, dtype=np.float64)
        w[self.fit_indices] = global_group_weights(self.a['head'][self.fit_indices])
        self.weights = torch.as_tensor(w, device=device)

    def cuda(self):
        assert torch.cuda.is_available(), 'CUDA unavailable in current execution environment'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(self.c['cuda_memory_fraction'], 0)
        free, _ = torch.cuda.mem_get_info()
        assert free / 2**20 >= self.c['minimum_free_mib'], 'Insufficient GPU memory'
        return torch.device('cuda:0')

    def model(self, arm, device):
        m = build_model(self.c, arm).to(device)
        m.load_state_dict(torch.load(self.root / 'prepare' / f'{arm}.initial.pt', map_location=device))
        return m

    def prediction(self, model, indices):
        index = torch.as_tensor(indices, device=self.t['features'].device, dtype=torch.long)
        return model(self.t['features'][index], self.t['candidate_ids'][index],
                     self.t['histories'][index], self.t['history_attributes'][index], self.text, self.cf)

    def update(self, model, optimizer, indices):
        optimizer.zero_grad(set_to_none=True)
        index = torch.as_tensor(indices, device=self.weights.device, dtype=torch.long)
        correction = self.prediction(model, indices)
        score = self.t['base_scores'][index] + correction.double()
        loss = (dense_pairwise_losses(standardize_tensor(score), self.t['positions'][index]) * self.weights[index]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.c['gradient_clip'])
        assert torch.isfinite(norm)
        optimizer.step(); self.steps += 1
        return float(loss.detach()), float(norm)

    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.c['learning_rate'],
                                 weight_decay=self.c['weight_decay'])

    def score(self, model, indices):
        result = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), self.c['batch_size']):
                ids = indices[start:start+self.c['batch_size']]
                correction = self.prediction(model, ids).cpu().numpy().astype(np.float64)
                result.append(self.a['base_scores'][ids] + correction)
        return np.concatenate(result)

    def profile(self):
        device = self.cuda(); self.load_tensors(device)
        batch = self.c['batch_size']
        length_order = np.argsort(-np.count_nonzero(self.a['histories'][self.fit_indices], axis=1), kind='stable')
        ids = self.fit_indices[length_order[:batch]]
        results = {}
        for arm in self.c['arms']:
            model = self.model(arm, device)
            with torch.no_grad():
                assert torch.count_nonzero(self.prediction(model, ids)).item() == 0
            opt = self.optimizer(model)
            torch.cuda.reset_peak_memory_stats()
            self.update(model, opt, ids)  # warmup; discarded along with all profile updates
            torch.cuda.synchronize(); start = time.monotonic()
            for _ in range(self.c['profile_steps']):
                loss, norm = self.update(model, opt, ids)
            torch.cuda.synchronize(); elapsed = time.monotonic()-start
            grads = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
            assert all(np.isfinite(list(grads.values())))
            if arm != 'B_scores':
                assert all(grads[name+'.weight'] > 0 for name in ('sq','sk','sv','cq','ck','cv'))
                assert grads['joint.0.weight' if arm == 'D_joint' else 'pooled_control.0.weight'] > 0
            results[arm] = {'seconds_per_step_longest_histories': elapsed/self.c['profile_steps'],
                            'last_fit_loss': loss, 'last_gradient_norm': norm, 'gradient_norms': grads,
                            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
            self.status('profile_'+arm, result=results[arm])
            del model, opt
        updates = self.c['epochs'] * int(np.ceil(len(self.fit_indices)/batch))
        predicted = sum(v['seconds_per_step_longest_histories'] for v in results.values()) * updates
        assert predicted < self.c['profile_training_prediction_limit_seconds'], 'Profile exceeds training budget'
        self.finish({'state': 'PASSED', 'arms': results, 'steps_per_formal_arm': updates,
                     'estimated_training_seconds_from_longest_histories': predicted,
                     'device': torch.cuda.get_device_name(0), 'gpu_uuid': self.c['gpu_uuid'],
                     'profile_weights_discarded': True, 'validation_predictions_computed': False})

    def train(self):
        self.verify_mode('profile')
        device = self.cuda(); self.load_tensors(device)
        calibration = np.flatnonzero(~self.a['fit'])
        results = {}
        for arm in self.c['arms']:
            self.status('train_'+arm)
            model, curve = self.model(arm, device), []
            initial = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            optimizer = self.optimizer(model)
            generator = np.random.default_rng(self.c['seed'])
            arm_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, self.c['epochs']+1):
                model.train()
                sequence = generator.permutation(self.fit_indices)
                losses, gradient, last_log = [], [], time.monotonic()
                for offset in range(0, len(sequence), self.c['batch_size']):
                    ids = sequence[offset:offset+self.c['batch_size']]
                    loss, norm = self.update(model, optimizer, ids)
                    losses.append((loss,len(ids))); gradient.append(norm)
                    if time.monotonic()-last_log > 20:
                        self.status('train_'+arm, epoch=epoch, rows=offset+len(ids))
                        last_log = time.monotonic()
                cal_scores = self.score(model, calibration)
                cal_ranks = ranks(cal_scores, self.a['positions'][calibration])
                row = {'epoch': epoch, 'online_weighted_fit_loss': sum(l*n for l,n in losses)/sum(n for _,n in losses),
                       'mean_gradient_norm': float(np.mean(gradient)), 'calibration': metrics(cal_ranks)}
                curve.append(row)
                self.status('train_'+arm, **row)
                write(self.path / f'{arm}.curve.json', curve)
            final = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(final, self.path / f'{arm}.pt')
            torch.save(optimizer.state_dict(), self.path / f'{arm}.optimizer.pt')
            np.savez(self.path / f'{arm}.calibration.npz', scores=cal_scores, indices=calibration)
            results[arm] = {'training_seconds': time.monotonic()-arm_start,
                            'parameters': sum(p.numel() for p in model.parameters()),
                            'parameter_change_l2': float(sum((final[k]-initial[k]).square().sum() for k in final).sqrt()),
                            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                            'selected_epoch': self.c['epochs'], 'selection': 'fixed_last_epoch',
                            'optimizer_steps': self.c['epochs']*int(np.ceil(len(self.fit_indices)/self.c['batch_size']))}
            del model, optimizer
        # All three fixed-last checkpoints exist before any new validation score.
        scores = {'A_frozen': self.a['base_scores'][self.n:]}
        for arm in self.c['arms']:
            self.status('full_validation_'+arm)
            model = self.model(arm, device)
            model.load_state_dict(torch.load(self.path / f'{arm}.pt', map_location=device))
            scores[arm] = self.score(model, np.arange(self.n, len(self.a['users'])))
            assert np.isfinite(scores[arm]).all()
            del model
        np.savez(self.path / 'validation_scores.npz', **scores)
        self.finish({'state':'COMPLETED', 'arms':results, 'all_arms_trained_before_validation':True,
                     'gpu_uuid':self.c['gpu_uuid'], 'backbone_updates':0, 'new_beam':False})

    def evaluate(self):
        self.verify_mode('prepare'); self.verify_mode('train')
        a = np.load(self.root / 'prepare/queries.npz'); n = int(a['ntrain'])
        scores = np.load(self.root / 'train/validation_scores.npz')
        rr = {arm:ranks(scores[arm], a['positions'][n:]) for arm in scores.files}
        np.testing.assert_array_equal(rr['A_frozen'], a['known_ranks'])
        assert all(np.array_equal(r<=50, a['known_ranks']<=50) for r in rr.values())
        masks = {'all':np.ones(len(a['known_ranks']),dtype=bool),
                 'low_frequency':a['target_frequencies']<=self.c['q1'],
                 'other_frequency':a['target_frequencies']>self.c['q1']}
        groups = {g:{arm:metrics(r[mask]) for arm,r in rr.items()} for g,mask in masks.items()}
        comparisons = {}
        for arm,ref in [('B_scores','A_frozen'),('C_attention','A_frozen'),('D_joint','A_frozen'),
                        ('C_attention','B_scores'),('D_joint','B_scores'),('D_joint','C_attention')]:
            comparisons[arm+'_minus_'+ref] = {g:paired(rr[arm][mask],rr[ref][mask],self.c) for g,mask in masks.items()}
        with (self.path / 'per_user.tsv').open('w') as handle:
            writer=csv.writer(handle,delimiter='\t')
            writer.writerow(['user_id','target_frequency',*rr])
            for j,u in enumerate(a['users'][n:]):
                writer.writerow([u,int(a['target_frequencies'][j]),*[int(r[j]) for r in rr.values()]])
        allm=groups['all']; base=allm['A_frozen']['ndcg@10']; main=allm['D_joint']['ndcg@10']
        above = all(main>allm[k]['ndcg@10'] for k in ('A_frozen','B_scores','C_attention'))
        if above and main/base-1 >= self.c['practical_relative_gain'] and allm['D_joint']['hit@10']>=allm['A_frozen']['hit@10']:
            decision='REVIEW_JOINT_INCREMENT_BEFORE_FOLLOWUP'
        elif above:
            decision='SMALL_OR_MIXED_JOINT_INCREMENT_NO_AUTOMATIC_FOLLOWUP'
        elif allm['C_attention']['ndcg@10']>base:
            decision='ORDINARY_ATTENTION_ONLY_JOINT_NOT_ESTABLISHED'
        else:
            decision='STOP_THIS_HISTORY_INTERACTION_RECIPE'
        self.finish({'state':'ANALYZED','decision':decision,'groups':groups,'comparisons':comparisons,
                     'joint_above_all_N10_controls':above,'new_method_confirmed':False,
                     'selection':'fixed_last_epoch_no_validation_checkpoint_selection',
                     'uncertainty':'descriptive paired user bootstrap on repeatedly used development validation'})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',required=True,choices=['prepare','profile','train','evaluate'])
    p.add_argument('--config',default=str(DEFAULT)); args=p.parse_args()
    worker=None
    try:
        worker=Worker(args.config,args.mode)
        getattr(worker,args.mode)()
    except Exception as e:
        if worker is not None:
            worker.status('error','FAILED',error=str(e),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
