"""Matched CF event-exposure screen; no test access or checkpoint search.

Modes are separate exclusive attempts. Prepared prefixes, candidates and frozen
deployment scores are hash checked. Only CF training uses CUDA.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment/phase9'))
from train_cf0_b2_item_head import CF0B2ItemHead, set_seed, scheduler_lambda
from experiment.phase20.conditional_reference_core import Correction, frozen_fusion, arm_features, ranks
from experiment.phase20.conditional_reference_screen import rows, read, write, sha, metrics, paired
from experiment.phase20.gacr_fusion_core import standardize_tensor, pairwise_losses
from experiment.phase20.personal_popularity_core import reconstruct_training


def split_users(users, seed):
    return np.asarray([int.from_bytes(hashlib.sha256(
        f'{seed}|fusion-fit|{u}'.encode()).digest()[:8], 'big') % 5 != 0 for u in users])


def terminal_indices(records):
    result = {}
    for i, r in enumerate(records):
        result[r['user_id']] = i
    return result


def padded_histories(records):
    result = np.zeros((len(records), 20), dtype=np.int64)
    for i, r in enumerate(records):
        h = r['history']
        assert 0 < len(h) <= 20 and all(x > 0 for x in h)
        result[i, :len(h)] = h
    return result


def masked_ce(logits, targets, keep):
    if not bool(keep.any()):
        raise ValueError('No retained events in batch')
    losses = F.cross_entropy(logits, targets - 1, reduction='none')
    return (losses * keep.to(losses.dtype)).sum() / keep.sum()


def cf_features(data, cf):
    score, common, positions, order, _ = frozen_fusion(
        data['g'], cf, data['frequencies'], data['raw_positions'], data['base_rank'], 5)
    return score, arm_features(common, cf, order), positions, order


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for k, v in model.state_dict().items():
        digest.update(k.encode())
        digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Job:
    def __init__(self, config_path, mode):
        self.config_path = config_path
        self.config = read(config_path)
        self.mode = mode
        self.root = ROOT / self.config['output']
        self.path = self.root / mode
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.config_hash = sha(config_path)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        self.status('RUNNING', 'verify')
        self.verify()

    def verify(self):
        assert sha(self.config_path) == self.config_hash
        for name, expected in self.config['input_sha256'].items():
            assert sha(ROOT / name) == expected, name
        for name, expected in self.config['source_sha256'].items():
            assert sha(ROOT / name) == expected, name

    def status(self, state='RUNNING', stage='', **extra):
        elapsed = time.monotonic() - self.started
        if state == 'RUNNING' and elapsed > self.config['budgets'][self.mode]:
            raise TimeoutError('Predeclared worker time budget')
        record = dict(state=state, stage=stage, pid=os.getpid(),
                      elapsed_seconds=elapsed, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)
        write(self.path / 'status.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def complete(self, result):
        self.verify()
        result.update(state='COMPLETED', elapsed_seconds=time.monotonic() - self.started,
                      config_sha256=self.config_hash, test_read=False, new_beam=False,
                      automatic_retry=False)
        write(self.path / 'summary.json', result)
        write(self.path / 'manifest.json', {p.name: sha(p) for p in self.path.iterdir()
              if p.is_file() and p.name not in ('manifest.json', 'status.json')})
        self.status('COMPLETED', 'finished')

    def load_artifact(self, mode, filename):
        manifest = read(self.root / mode / 'manifest.json')
        path = self.root / mode / filename
        assert sha(path) == manifest[filename]
        assert read(self.root / mode / 'summary.json')['config_sha256'] == self.config_hash
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def prepare(self):
        d = self.config['data']
        train = list(rows(ROOT / d['train']))
        _, sequences, _, _ = reconstruct_training(train)
        terminal = terminal_indices(train)
        catalog = read(ROOT / d['catalog'])
        assert catalog['items'][1:] == sorted(catalog['items'][1:])
        frequencies = np.bincount([x for seq in sequences.values() for x in seq],
                                  minlength=len(catalog['items']))
        with np.load(ROOT / d['training_cache']) as z:
            a = {k: z[k] for k in z.files}
        selected, records, cached_fit_users = [], [], set()
        for i, r in enumerate(rows(ROOT / d['training_candidates'])):
            source = train[r['train_index']]
            assert all(r[k] == source[k] for k in ('user_id', 'history', 'target'))
            assert bool(a['fit'][i]) == (r['partition'] == 'fit')
            if a['fit'][i]:
                cached_fit_users.add(r['user_id'])
            if a['fit'][i] and r['train_index'] == terminal[r['user_id']]:
                assert r['union_ids'][:50] == a['candidate_ids'][i].tolist()
                assert r['target_position'] == int(a['positions'][i])
                selected.append(i)
                records.append({k: r[k] for k in ('user_id', 'history', 'target', 'train_index')})
        selected = np.asarray(selected)
        assert len(records) == len(cached_fit_users) == 3882
        assert {r['user_id'] for r in records} == cached_fit_users
        assert len(train) == 109361 and int((a['positions'][selected] >= 0).sum()) == 2121
        event_ids = np.asarray([r['train_index'] for r in records])
        excluded = np.zeros(len(train), dtype=bool)
        excluded[event_ids] = True
        assert excluded.sum() == len(records)
        fit = split_users([r['user_id'] for r in records], self.config['fusion_seed'])
        assert set(r['user_id'] for r, f in zip(records, fit) if f).isdisjoint(
            r['user_id'] for r, f in zip(records, fit) if not f)
        b1 = Correction(.2).eval()
        b1.load_state_dict(torch.load(ROOT / d['b1_checkpoint'], map_location='cpu'))
        train_g = []
        with torch.no_grad():
            for start in range(0, len(selected), 128):
                ids = selected[start:start + 128]
                train_g.append((torch.from_numpy(a['base'][ids]) +
                                b1(torch.from_numpy(a['features'][ids]))).numpy().astype(np.float64))
        validation = list(rows(ROOT / d['validation']))
        vu = [r['user_id'] for r in validation]
        assert len(set(vu)) == len(vu) == 19412
        assert all(r['history'] == sequences[r['user_id']][-20:] for r in validation)
        with np.load(ROOT / d['validation_cf']) as vcf, np.load(ROOT / d['validation_b1_scores']) as vb1:
            assert vcf['users'].tolist() == vb1['users'].tolist() == vu
            np.testing.assert_array_equal(vcf['candidate_ids'], vb1['candidate_ids'])
            vc = vcf['candidate_ids']
            vp = np.asarray([list(c).index(r['target']) if r['target'] in c else -1
                             for c, r in zip(vc, validation)])
            candidates = np.concatenate([a['candidate_ids'][selected], vc])
            queries = dict(g=np.concatenate([np.concatenate(train_g), vb1['residual_scores']]),
                original_cf=np.concatenate([a['cf'][selected], vcf['scores']]),
                candidate_ids=candidates, frequencies=frequencies[candidates],
                raw_positions=np.r_[a['positions'][selected], vp],
                base_rank=np.concatenate([a['base'][selected], np.broadcast_to(1. / np.arange(1, 51), vc.shape)]),
                histories=padded_histories(records + validation),
                targets=np.asarray([r['target'] for r in records + validation]),
                users=np.asarray([r['user_id'] for r in records + validation]),
                target_frequencies=frequencies[[r['target'] for r in records + validation]],
                fit=fit, head=a['head'][selected], selected_cache_rows=selected,
                event_ids=event_ids, ntrain=np.asarray(len(records)), num_items=np.asarray(len(frequencies) - 1))
        score, features, positions, order = cf_features(queries, queries['original_cf'])
        with (ROOT / d['validation_ranks']).open() as handle:
            known = {r['user_id']: int(r['B1']) for r in csv.DictReader(handle, delimiter='\t')}
        known_ranks = np.asarray([known[u] for u in vu])
        np.testing.assert_array_equal(ranks(score[len(records):], positions[len(records):]), known_ranks)
        np.testing.assert_array_equal(frequencies[a['candidate_ids'][selected]], a['frequencies'][selected])
        queries.update(original_base=score, original_features=features, positions=positions,
                       order=order, known_ranks=known_ranks)
        np.savez(self.path / 'queries.npz', **queries)
        np.savez(self.path / 'cf_train.npz', histories=padded_histories(train),
                 targets=np.asarray([r['target'] for r in train]), excluded=excluded)
        with (self.path / 'selected_events.jsonl').open('x') as handle:
            for r, f in zip(records, fit):
                handle.write(json.dumps(dict(r, fusion_partition='fit' if f else 'calibration')) + '\n')
        previous = {}
        duplicate_events = 0
        for i, r in enumerate(train):
            key = (r['user_id'], tuple(r['history']), r['target'])
            if excluded[i] and key in previous:
                duplicate_events += 1
            previous[key] = i
        self.complete(dict(train_events=len(train), excluded_events=int(excluded.sum()),
            selected_covered=2121, fusion_fit=int(fit.sum()), fusion_calibration=int((~fit).sum()),
            fusion_fit_covered=int((fit & (positions[:len(records)] >= 0)).sum()),
            fusion_calibration_covered=int((~fit & (positions[:len(records)] >= 0)).sum()),
            selected_target_in_history=sum(r['target'] in r['history'] for r in records),
            selected_duplicate_earlier_user_history_target=duplicate_events,
            terminal_event_proof=True, frozen_baseline_ranks_exact=True,
            baseline_validation=metrics(known_ranks)))

    def make_cf(self):
        cfg = self.config['cf']
        return CF0B2ItemHead(11924, 20, 512, 2, 4, .1, .07).cuda()

    def train_one(self, arm, data, profile=False):
        c = self.config['cf']
        set_seed(c['seed'])
        loader = DataLoader(np.arange(len(data['targets'])), batch_size=c['batch_size'],
                            shuffle=True, generator=torch.Generator().manual_seed(c['seed']), num_workers=0)
        model = self.make_cf()
        initial = state_digest(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=.01)
        total = len(loader) * c['epochs']
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: scheduler_lambda(step, total, int(total * .05)))
        digest = hashlib.sha256()
        curves, steps, masked_count = [], 0, 0
        timed_steps = []
        started = time.monotonic()
        for epoch in range(1, c['epochs'] + 1):
            model.train()
            loss_sum, n = 0., 0
            for raw_index in loader:
                index = raw_index.numpy()
                digest.update(index.tobytes())
                h = data['histories'][index]
                h = torch.from_numpy(h[:, :int((h > 0).sum(1).max())]).cuda()
                t = torch.from_numpy(data['targets'][index]).cuda()
                keep = ~data['excluded'][index] if arm == 'excluded' else np.ones(len(index), dtype=bool)
                masked_count += int((~keep).sum())
                keep = torch.from_numpy(keep).cuda()
                torch.cuda.synchronize()
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss = masked_ce(model.score(h, h.ne(0)), t, keep)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite CF loss')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize()
                timed_steps.append(time.monotonic() - tick)
                steps += 1
                count = int(keep.sum())
                loss_sum += float(loss.detach()) * count
                n += count
                if profile and steps == 12:
                    return model, dict(timed_steps=timed_steps, projected_two_arm_seconds=float(
                        np.mean(timed_steps[2:]) * total * 2 * 1.3), steps=steps,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
                if steps % 100 == 0:
                    self.status(stage='train_' + arm, epoch=epoch, steps=steps, loss=float(loss.detach()))
            curves.append(dict(epoch=epoch, training_loss=loss_sum / n, retained_events=n,
                               seconds=time.monotonic() - started))
            write(self.path / f'{arm}.training.json', curves)
            self.status(stage='train_' + arm, epoch=epoch, steps=steps, loss=curves[-1]['training_loss'])
        assert steps == total == 2140
        assert masked_count == (3882 * 10 if arm == 'excluded' else 0)
        final = state_digest(model)
        assert final != initial
        torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(), epoch=10, model_config=self.config['cf']),
                   self.path / f'{arm}.pt')
        return model, dict(steps=steps, initial_sha256=initial, final_sha256=final,
            batch_order_sha256=digest.hexdigest(), masked_events_over_epochs=masked_count,
            curve=curves, seconds=time.monotonic() - started)

    def score_cf(self, model, queries):
        model.eval()
        scores = np.empty_like(queries['original_cf'])
        with torch.no_grad():
            for start in range(0, len(scores), 512):
                stop = start + 512
                h = queries['histories'][start:stop]
                h = torch.from_numpy(h[:, :int((h > 0).sum(1).max())]).cuda()
                ids = torch.from_numpy(queries['candidate_ids'][start:stop] - 1).cuda()
                scores[start:stop] = model.score(h, h.ne(0)).gather(1, ids).cpu().numpy()
        assert np.isfinite(scores).all()
        return scores

    def profile(self):
        assert torch.cuda.is_available(), 'CUDA unavailable; use approved GPU execution environment'
        torch.cuda.set_per_process_memory_fraction(.125)
        torch.cuda.reset_peak_memory_stats()
        data = self.load_artifact('prepare', 'cf_train.npz')
        _, result = self.train_one('excluded', data, profile=True)
        result['passed'] = result['projected_two_arm_seconds'] < self.config['budgets']['train'] - 300
        result['gpu'] = torch.cuda.get_device_name(0)
        self.complete(result)

    def train(self):
        assert read(self.root / 'profile/summary.json')['passed']
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.125)
        torch.cuda.reset_peak_memory_stats()
        data = self.load_artifact('prepare', 'cf_train.npz')
        queries = self.load_artifact('prepare', 'queries.npz')
        results, source_scores = {}, {}
        for arm in ('seen', 'excluded'):
            model, results[arm] = self.train_one(arm, data)
            self.status(stage='score_' + arm)
            source_scores[arm] = self.score_cf(model, queries)
            del model
            torch.cuda.empty_cache()
        for key in ('initial_sha256', 'batch_order_sha256', 'steps'):
            assert results['seen'][key] == results['excluded'][key]
        np.savez(self.path / 'source_scores.npz', **source_scores)
        self.complete(dict(arms=results, matched_initialization_and_batches=True,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, gpu=torch.cuda.get_device_name(0)))

    def fit_fuser(self, features, base, positions, fit, head, arm, deployment_features, deployment_base):
        torch.manual_seed(self.config['fusion_seed'])
        model = Correction(.5)
        initial = state_digest(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        x, b, p = torch.from_numpy(features), torch.from_numpy(base), torch.from_numpy(positions)
        valid = fit & (positions >= 0)
        groups = [np.flatnonzero(valid & (head == h)) for h in (True, False)]
        curve = []
        for step in range(1, 31):
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for group in groups:
                assert len(group) > 0
                for start in range(0, len(group), 512):
                    ix = torch.from_numpy(group[start:start + 512])
                    loss = pairwise_losses(standardize_tensor(b[ix] + model(x[ix]).double()), p[ix]).sum() / (2 * len(group))
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    value += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            curve.append(dict(step=step, loss_before_step=value, gradient=float(grad)))
        assert state_digest(model) != initial
        torch.save(model.state_dict(), self.path / f'{arm}.pt')
        torch.save(optimizer.state_dict(), self.path / f'{arm}.optimizer.pt')
        model.eval()
        def infer(features, base):
            result = base.copy()
            with torch.no_grad():
                for start in range(0, len(base), 512):
                    result[start:start + 512] += model(torch.from_numpy(features[start:start + 512])).numpy()
            return result
        self.status(stage='fuser_' + arm, steps=30, loss=curve[-1]['loss_before_step'])
        return infer(deployment_features, deployment_base), infer(features, base), dict(
            curve=curve, initial_sha256=initial, head_fit_covered=len(groups[0]), other_fit_covered=len(groups[1]))

    def evaluate(self):
        q = self.load_artifact('prepare', 'queries.npz')
        sources = self.load_artifact('train', 'source_scores.npz')
        sources['historical'] = q['original_cf']
        n = int(q['ntrain'])
        p = q['positions']
        output = {'baseline': q['original_base'][n:]}
        train_summary, diagnostics = {}, {}
        validation_masks = dict(all=np.ones(len(p) - n, dtype=bool),
            low_frequency=q['target_frequencies'][n:] <= 5,
            excluded_users=np.isin(q['users'][n:], q['users'][:n]),
            other_users=~np.isin(q['users'][n:], q['users'][:n]))
        for arm in ('historical', 'seen', 'excluded'):
            base, features, positions, order = cf_features(q, sources[arm])
            np.testing.assert_array_equal(positions, p)
            np.testing.assert_array_equal(order, q['order'])
            output[arm], fitted, train_summary[arm] = self.fit_fuser(
                features[:n], base[:n], p[:n], q['fit'], q['head'], arm,
                q['original_features'][n:], q['original_base'][n:])
            cf_rank = ranks(sources[arm], q['raw_positions'])
            source_rank = ranks(base, p)
            raw_rank = ranks(q['g'], q['raw_positions'])
            fit_rank = ranks(fitted, p[:n])
            diagnostics[arm] = {}
            for group, ids in [('fit', np.flatnonzero(q['fit'])),
                               ('calibration', np.flatnonzero(~q['fit'])),
                               ('validation', np.arange(n, len(p)))]:
                covered = ids[p[ids] >= 0]
                nd = lambda r: np.where(r <= 10, 1. / np.log2(r + 1), 0.)
                delta = nd(source_rank[ids]) - nd(raw_rank[ids])
                diagnostics[arm][group] = dict(cf_all=metrics(cf_rank[ids]),
                    cf_covered=metrics(cf_rank[covered]), source_pcrf=metrics(source_rank[ids]),
                    pcrf_ndcg10_improved=int((delta > 0).sum()), pcrf_ndcg10_harmed=int((delta < 0).sum()))
                if group != 'validation':
                    diagnostics[arm][group]['source_fuser'] = metrics(fit_rank[ids])
        assert len(set(s['initial_sha256'] for s in train_summary.values())) == 1
        rr = {a: ranks(s, p[n:]) for a, s in output.items()}
        np.testing.assert_array_equal(rr['baseline'], q['known_ranks'])
        assert all(np.array_equal(r <= 50, rr['baseline'] <= 50) for r in rr.values())
        groups = {g: {a: metrics(r[mask]) for a, r in rr.items()} for g, mask in validation_masks.items()}
        comparisons = {}
        for a, b in [('historical', 'baseline'), ('seen', 'baseline'), ('excluded', 'baseline'),
                     ('excluded', 'seen'), ('excluded', 'historical')]:
            comparisons[a + '_minus_' + b] = {g: paired(rr[a][m], rr[b][m], self.config)
                                             for g, m in validation_masks.items()}
            self.status(stage='paired_comparison', comparison=a + '_minus_' + b)
        points = groups['all']
        passes = all(points['excluded']['ndcg@10'] > points[b]['ndcg@10']
                     for b in ('baseline', 'seen', 'historical')) and points['excluded']['hit@10'] >= points['baseline']['hit@10']
        np.savez(self.path / 'validation_scores.npz', **output)
        with (self.path / 'per_user.tsv').open('x') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', *rr])
            for i, u in enumerate(q['users'][n:]):
                writer.writerow([u, *[int(r[i]) for r in rr.values()]])
        self.complete(dict(groups=groups, comparisons=comparisons, training=train_summary,
            diagnostics=diagnostics, outcome='REVIEW_NEXT_SCREEN' if passes else 'STOP_THIS_RECIPE',
            shared_deployment_base_sha256=array_digest(q['original_base'][n:]),
            shared_deployment_features_sha256=array_digest(q['original_features'][n:]),
            fixed_baseline_ranks_exact=True, fixed_hit50_exact=True,
            statistical_scope='Repeated development validation, single training seed. Paired bootstrap is descriptive, unadjusted; not final confirmation.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiment/phase20/cf_exposure_toys_v1.json')
    parser.add_argument('--mode', choices=['prepare', 'profile', 'train', 'evaluate'], required=True)
    args = parser.parse_args()
    job = Job(ROOT / args.config, args.mode)
    try:
        getattr(job, args.mode)()
    except BaseException as error:
        job.status('FAILED', 'exception', error=repr(error))
        (job.path / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
