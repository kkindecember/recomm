"""One bounded revision: differentiate the exact candidate Top10 threshold."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, z, ranks, metrics, sha
from experiment.phase20.future_fusion_alignment import standardize
from experiment.phase20.future_label_multiseed import write, fingerprint, describe, compare

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/talos_exact_threshold_20260917'
ADAPTER = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'
PREV = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'
FACTOR = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
PAIR = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
BASELINE = ROOT / 'artifacts/phase20/strong_baseline_20260917/baseline.json'
CONTRACT = ROOT / 'plan/第二十阶段/Talos精确阈值梯度修订_冻结合同_20260917.md'
ARMS = ('talos_exact',)
REFERENCES = ('normalized.next_only', 'fusion_mean.next_only', 'legacy.soft_future', 'softmax', 'negative_softmax', 'talos')
SEEDS = (20260917, 20260918, 20260919)
DEPLOYMENTS = ('raw', 'current', 'seed2024', 'seed2025', 'mean')
CONTRASTS = [(a, b) for a in ARMS for b in REFERENCES]


def row_loss(scores, positions, arm, temperature=1., threshold=None):
    assert arm == 'talos_exact'
    if threshold is None:
        threshold = torch.topk(scores, k=10, dim=1).values[:, -1:]
    transformed = F.logsigmoid(scores - threshold) / temperature
    positive = transformed.gather(1, positions[:, None]).squeeze(1)
    mask = torch.arange(scores.shape[1])[None] == positions[:, None]
    return torch.logsumexp(transformed.masked_fill(mask, -torch.inf), 1) - positive


def checks():
    from experiment.phase20.talos_candidate_screen import row_loss as original
    rng = np.random.RandomState(716)
    x = torch.tensor(rng.normal(size=(4, 50)), requires_grad=True)
    pos = torch.tensor([0, 9, 20, 49])
    out = row_loss(x, pos, 'talos_exact', .7)
    np.testing.assert_allclose(out.detach(), original(x, pos, 'talos', .7).detach(), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda a: row_loss(a, pos, 'talos_exact', .7).sum(), (x,), eps=1e-6, atol=1e-5)
    np.testing.assert_allclose(out.detach(), row_loss(x+100, pos, 'talos_exact', .7).detach(), atol=1e-12, rtol=0)
    perm = torch.randperm(50); inv = torch.argsort(perm)
    np.testing.assert_allclose(out.detach(), row_loss(x[:,perm], inv[pos], 'talos_exact', .7).detach(), atol=1e-12, rtol=0)
    for values in (torch.zeros(4,50,dtype=torch.float64), torch.linspace(-1e4,1e4,200).reshape(4,50).double()):
        values.requires_grad_(); loss=row_loss(values,pos,'talos_exact').sum();loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(values.grad).all()
    features=torch.tensor(rng.normal(size=(4,50,6)),dtype=torch.float32)
    prior=torch.tensor(z(rng.normal(size=(4,50)))-np.log(49)); cf=torch.tensor(rng.normal(size=(4,50)))
    gradients=[]
    for chunk in (4,2):
        torch.manual_seed(SEEDS[0]); model=Factor('next_only')
        for offset in range(0,4,chunk):
            sl=slice(offset,offset+chunk); l,_=model(features[sl],prior[sl])
            np.testing.assert_allclose(l.detach(),prior[sl],atol=1e-12,rtol=0)
            (row_loss(standardize(l)+cf[sl],pos[sl],'talos_exact').sum()/200).backward()
        gradients.append(np.concatenate([p.grad.numpy().ravel() for p in model.parameters()]))
    np.testing.assert_allclose(*gradients,atol=1e-7,rtol=1e-5)
    before=fingerprint(model);torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01).step()
    assert before!=fingerprint(model) and sum(p.numel() for p in model.parameters())==258
    return dict(original_forward_parity=True, finite_difference_recomputed_threshold=True,
        translation_invariance=True, candidate_permutation=True, extreme_and_tied_finite_subgradient=True,
        full_vs_chunk_gradient=True, initial_B1_capacity_and_update=True)


def practical_gate(contrast, individual=None):
    overall = contrast['all']; tail = contrast['tail']; nonfit = contrast['nonfit']
    ndcg_route = overall['relative_N10']['mean'] >= .005 and overall['hit_net']['mean'] >= 0
    hit_route = overall['delta_H10']['mean'] >= .001 and overall['delta_N10']['mean'] >= 0
    protection = tail['delta_N10']['mean'] >= 0 and tail['hit_net']['mean'] >= 0 and nonfit['delta_N10']['mean'] >= 0
    directions = sum(x >= 0 for x in overall['delta_N10']['values']) >= 2
    return dict(ndcg_route=bool(ndcg_route), hit_route=bool(hit_route), protection=bool(protection),
                directions=bool(directions), passed=bool((ndcg_route or hit_route) and protection and directions))


def specificity_gate(contrast):
    overall = contrast['all']
    gain = ((overall['relative_N10']['mean'] >= .005 and overall['hit_net']['mean'] >= 0) or
            (overall['delta_H10']['mean'] >= .001 and overall['delta_N10']['mean'] >= 0))
    protection = contrast['tail']['hit_net']['mean'] >= 0 and contrast['nonfit']['delta_N10']['mean'] >= 0
    return dict(gain=bool(gain), protection=bool(protection), passed=bool(gain and protection))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    health = checks()
    if args.check_only: print(json.dumps(health)); return
    OUT.mkdir(exist_ok=False); started = time.monotonic(); steps = 0; hashes = {}; domains = {}

    def use(p):
        hashes[str(p.relative_to(ROOT))] = sha(p); return p

    def read(folder, filename):
        with np.load(use(folder / filename), allow_pickle=False) as data: return {key: data[key] for key in data.files}

    def status(state, **kwargs):
        write(OUT / 'status.json', dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            elapsed_seconds=time.monotonic() - started, pid=os.getpid(), training_steps=steps, planned_training_steps=180,
            gpu_used=False, test_read=False, automatic_next_experiment=False, **kwargs))

    def timeout(signum, frame): raise TimeoutError('30 minute CPU hard limit')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(1800)
    try:
        status('RUNNING', stage='verify_inputs')
        for p in (Path(__file__), CONTRACT, BASELINE, ROOT / 'experiment/phase20/future_factor_screen.py',
                  ROOT / 'experiment/phase20/future_fusion_alignment.py', ROOT / 'experiment/phase20/future_label_multiseed.py'): use(p)
        old = json.loads(use(PREV / 'summary.json').read_text())
        audit = json.loads(use(PREV / 'verification.json').read_text()); assert audit['state'] == 'VERIFIED'
        for mapping in (old['input_sha256'], audit['output_sha256']):
            for p, h in mapping.items(): use(ROOT / p); assert hashes[p] == h, p
        previous_adapter = json.loads(use(ADAPTER / 'summary.json').read_text())
        adapter_audit = json.loads(use(ADAPTER / 'verification.json').read_text()); assert adapter_audit['state'] == 'VERIFIED'
        use(ADAPTER / 'threshold_direction.json')
        use(ROOT / 'experiment/phase20/check_talos_threshold_direction.py')
        for mapping in (previous_adapter['input_sha256'], adapter_audit['output_sha256']):
            for p, h in mapping.items(): use(ROOT / p); assert hashes[p] == h, p
        registered = json.loads(BASELINE.read_text()); assert registered['canonical_arm'] == 'normalized.next_only.mean'
        for p, h in registered['sha256'].items(): use(ROOT / p); assert hashes[p] == h, p
        initial_map = json.loads(use(ROOT / 'artifacts/phase20/future_label_multiseed_20260917/protocol.json').read_text())['initialization_sha256']
        write(OUT / 'protocol.json', dict(contract=str(CONTRACT.relative_to(ROOT)), checks=health, seeds=SEEDS,
            arms=ARMS, references=REFERENCES, deployments=DEPLOYMENTS, contrasts=CONTRASTS, temperature=1., top_k=10,
            loss_divisor=50, gradient_clipping=10., lr=.01, weight_decay=.01, steps_per_arm=30, chunk=256,
            new_training_steps=180, new_checkpoints=6, threads=4, max_seconds=1800, development_reused=True,
            required_strong_baseline=registered['canonical_arm'], source_sha256=hashes.copy()))
        for domain in ('Toys', 'Beauty'):
            status('RUNNING', stage='load_data', dataset=domain)
            a = read(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}', 'training_cache.npz')
            ctx = read(PREV, f'{domain}.training_context.npz'); ix = ctx['fit_indices']; pos = a['positions'][ix]
            assert a['fit'][ix].all() and (pos >= 0).all()
            np.testing.assert_array_equal(ctx['candidate_ids'], a['candidate_ids'][ix])
            np.testing.assert_array_equal(ctx['targets'], ctx['candidate_ids'][np.arange(len(ix)), pos])
            x = torch.as_tensor(a['features'][ix]); prior = torch.as_tensor(ctx['prior']); cf = torch.as_tensor(ctx['cf_mean']); target = torch.as_tensor(pos)
            head = a['head'][ix]; weights = torch.as_tensor(np.where(head, .5 / head.sum(), .5 / (~head).sum()))
            v = read(FACTOR, f'{domain}.audit_validation_cache.npz'); pair = read(PAIR, f'{domain}.predictions.npz')
            for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(v[key], pair[key])
            parts = {f: pair['scores'][list(pair['arm_names']).index(f)] - (v['prior'] + np.log(49)) for f in DEPLOYMENTS[1:]}
            for f in ('current', 'mean'):
                np.testing.assert_allclose(parts[f], v['cf_' + f], atol=1e-11, rtol=0); parts[f] = v['cf_' + f]
            in_fit = np.isin(v['users'], np.unique(ctx['users']))
            masks = dict(all=np.ones(len(in_fit), bool), tail=v['tail_mask'], non_tail=~v['tail_mask'], fit=in_fit, nonfit=~in_fit)
            results = {}
            for seed in SEEDS:
                before = read(ADAPTER, f'{domain}.{seed}.predictions.npz')
                for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(v[key], before[key])
                np.testing.assert_array_equal(before['in_fit'], in_fit)
                raw = {name: before['raw_scores'][list(before['raw_names']).index(name)] for name in REFERENCES}
                training = {}
                for arm in ARMS:
                    torch.manual_seed(seed); model = Factor('next_only'); initial = fingerprint(model)
                    assert initial == initial_map[str(seed)]
                    initial_state = {k: t.clone() for k, t in model.state_dict().items()}
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
                    curve = []
                    for step in range(1, 31):
                        model.train(); optimizer.zero_grad(set_to_none=True); total = 0.
                        for offset in range(0, len(x), 256):
                            sl = slice(offset, offset + 256); logits, _ = model(x[sl], prior[sl]); score = standardize(logits) + cf[sl]
                            loss = (row_loss(score, target[sl], arm) * weights[sl]).sum() / 50
                            assert torch.isfinite(loss); loss.backward(); total += float(loss.detach())
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                        optimizer.step(); steps += 1
                        curve.append(dict(step=step, scaled_loss=total, loss=total * 50, grad_norm=float(grad)))
                        status('RUNNING', stage='training', dataset=domain, seed=seed, arm=arm, step=step)
                    model.eval(); final_loss = 0.; threshold_min = float('inf'); threshold_max = -float('inf')
                    with torch.no_grad():
                        for offset in range(0, len(x), 256):
                            sl = slice(offset, offset + 256); logits, _ = model(x[sl], prior[sl]); score = standardize(logits) + cf[sl]
                            final_loss += float((row_loss(score, target[sl], arm) * weights[sl]).sum())
                            beta = torch.topk(score, 10, dim=1).values[:, -1]
                            threshold_min = min(threshold_min, float(beta.min())); threshold_max = max(threshold_max, float(beta.max()))
                    delta = float(torch.sqrt(sum((t - initial_state[k]).square().sum() for k, t in model.state_dict().items())))
                    assert np.isfinite(delta) and delta > 0
                    training[arm] = dict(initial_sha256=initial, curve=curve, final_loss=final_loss,
                        parameter_delta_l2=delta, final_threshold_range=[threshold_min, threshold_max])
                    torch.save(model.state_dict(), OUT / f'{domain}.{seed}.{arm}.pt')
                    prediction = []
                    with torch.no_grad():
                        for offset in range(0, len(v['users']), 256):
                            sl = slice(offset, offset + 256); logits, _ = model(torch.as_tensor(v['features'][sl]), torch.as_tensor(v['prior'][sl]))
                            prediction.append(logits.numpy())
                    raw[arm] = z(np.concatenate(prediction))
                rr = {name + '.' + f: ranks(values if f == 'raw' else values + parts[f], v['positions']) for name, values in raw.items() for f in DEPLOYMENTS}
                for name in REFERENCES:
                    for f in DEPLOYMENTS: np.testing.assert_array_equal(rr[name + '.' + f], before['ranks'][list(before['rank_names']).index(name + '.' + f)])
                met = {name: {group: metrics(rank[mask]) for group, mask in masks.items()} for name, rank in rr.items()}
                comps = {a + '__vs__' + b: {f: {group: compare(rr[a + '.' + f][mask], rr[b + '.' + f][mask]) for group, mask in masks.items()} for f in DEPLOYMENTS} for a, b in CONTRASTS}
                results[str(seed)] = dict(metrics=met, comparisons=comps)
                write(OUT / f'{domain}.{seed}.training.json', training)
                np.savez_compressed(OUT / f'{domain}.{seed}.predictions.npz', users=v['users'], candidate_ids=v['candidate_ids'], positions=v['positions'],
                    tail_mask=v['tail_mask'], in_fit=in_fit, raw_names=np.asarray(list(raw)), raw_scores=np.stack(list(raw.values())),
                    rank_names=np.asarray(list(rr)), ranks=np.stack(list(rr.values())))
            aggregate = {c: {f: {group: {key: describe([s['comparisons'][c][f][group][key] for s in results.values()]) for key in ('delta_N10', 'relative_N10', 'delta_H10', 'hit_net')} for group in masks} for f in DEPLOYMENTS} for c in comps}
            average = {name: {group: {key: describe([s['metrics'][name][group][key] for s in results.values()]) for key in ('N10', 'H10')} for group in masks} for name in rr}
            utility = {a: practical_gate(aggregate[a + '__vs__normalized.next_only']['mean']) for a in ARMS}
            specific = {a: specificity_gate(aggregate['talos_exact__vs__' + a]['mean']) for a in ('softmax', 'negative_softmax')}
            domains[domain] = dict(seeds=results, aggregate_metrics=average, aggregate_comparisons=aggregate,
                utility_gates=utility, specificity_gates=specific, topk_gate=bool(utility['talos_exact']['passed'] and all(c['passed'] for c in specific.values())))
            write(OUT / f'{domain}.summary.json', domains[domain])
        assert steps == 180
        for path, h in hashes.items(): assert sha(ROOT / path) == h, path
        retained = [a for a in ARMS if all(d['utility_gates'][a]['passed'] for d in domains.values())]
        specific = all(d['topk_gate'] for d in domains.values())
        decision = 'EXACT_THRESHOLD_INCREMENT_SUPPORTED' if specific else ('PRACTICAL_BASELINE_ONLY' if retained else 'NO_ADVANCEMENT_GATE_PASSED')
        write(OUT / 'summary.json', dict(state='COMPLETED', domains=domains, checks=health, input_sha256=hashes,
            decision=decision, retained_arms=retained, training_steps=steps, test_read=False, development_reused=True))
        status('COMPLETED', decision=decision, retained_arms=retained); signal.alarm(0)
        print(json.dumps(dict(decision=decision, retained_arms=retained)))
    except BaseException as exc:
        status('FAILED', error=repr(exc)); raise


if __name__ == '__main__': main()
