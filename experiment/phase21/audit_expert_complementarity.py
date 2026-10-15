"""Read-only source-cache audit: expert opportunities versus realized guidance."""
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT / 'artifacts/phase21/expert_complementarity_20260923_v1'
PROTOCOL = ROOT / 'plan/第二十一阶段/21.3A_专家互补机会与实际兑现诊断_20260923.md'
EXPERTS = ('SASRec', 'ItemCF', 'semantic')
ABSENT = 2**30


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sorted_rank(ids, scores, target, excluded):
    ids, scores = np.asarray(ids), np.asarray(scores)
    valid = np.isfinite(scores) & ~np.isin(ids, excluded)
    ranked = ids[valid][np.lexsort((ids[valid], -scores[valid]))]
    pos = np.flatnonzero(ranked == target)
    return int(pos[0] + 1) if len(pos) else ABSENT


def directory_rank(scores, target, eligible):
    if not eligible[target - 1]:
        return ABSENT
    target_score = scores[target - 1]
    # Array order is ascending item ID. Excluded items never count as above.
    better = np.count_nonzero((scores > target_score) & eligible)
    tied = np.count_nonzero((scores[:target - 1] == target_score) & eligible[:target - 1])
    return int(1 + better + tied)


def utility(ranks, k=10):
    ranks = np.asarray(ranks)
    return np.where(ranks <= k, 1 / np.log2(ranks + 1), 0.)


def metrics(ranks):
    ranks = np.asarray(ranks)
    return {f'{m}{k}': float((ranks <= k).mean() if m == 'H' else utility(ranks, k).mean())
            for k in (5, 10, 20, 50) for m in ('H', 'N')}


def contrast(new, old):
    new, old = np.asarray(new), np.asarray(old)
    dn = utility(new) - utility(old)
    return dict(delta_N10=float(dn.mean()),
                gained10=int(((new <= 10) & (old > 10)).sum()),
                lost10=int(((old <= 10) & (new > 10)).sum()),
                n10_improved=int((dn > 1e-12).sum()), n10_worsened=int((dn < -1e-12).sum()))


def analyze_cohort(rows):
    names = ('GRAM', 'fixed_late', *EXPERTS, 'LIME')
    rr = {name: np.asarray([r[name] for r in rows]) for name in names}
    has_guided = all('fixed_decode' in r for r in rows)
    if has_guided:
        rr['fixed_decode'] = np.asarray([r['fixed_decode'] for r in rows])
    out = dict(users=len(rows), ineligible_targets=sum(r['ineligible_target'] for r in rows),
               metrics={k: metrics(v) for k, v in rr.items()}, opportunity={})
    for k in (10, 50):
        g = rr['GRAM'] <= k
        hits = {n: rr[n] <= k for n in (*EXPERTS, 'LIME')}
        eh = np.stack([hits[n] for n in EXPERTS])
        table = {}
        for name, h in hits.items():
            row = dict(both_hit=int((g & h).sum()), GRAM_only=int((g & ~h).sum()),
                       source_only=int((~g & h).sum()), neither=int((~g & ~h).sum()))
            assert sum(row.values()) == len(rows)
            if name in EXPERTS:
                row['unique_vs_GRAM_and_other_experts'] = int((~g & h & (eh.sum(0) == 1)).sum())
            table[name] = row
        opportunity = ~g & eh.any(0)
        table['any_expert'] = dict(GRAM_missed=int((~g).sum()),
                                 expert_opportunities=int(opportunity.sum()),
                                 GRAM_and_all_experts_miss=int((~g & ~eh.any(0)).sum()))
        if has_guided:
            guided = rr['fixed_decode'] <= k
            table['any_expert']['realized_by_fixed_decode'] = int((opportunity & guided).sum())
            table['any_expert']['unrealized_by_fixed_decode'] = int((opportunity & ~guided).sum())
            table['guided_misses'] = dict(total=int((~guided).sum()),
                all_experts_outside_k=int((~guided & ~eh.any(0)).sum()),
                at_least_one_expert_inside_k=int((~guided & eh.any(0)).sum()))
        out['opportunity'][str(k)] = table
    out['realized'] = {'fixed_late_minus_GRAM': contrast(rr['fixed_late'], rr['GRAM'])}
    if has_guided:
        out['realized']['fixed_decode_minus_GRAM'] = contrast(rr['fixed_decode'], rr['GRAM'])
        out['realized']['fixed_decode_minus_fixed_late'] = contrast(rr['fixed_decode'], rr['fixed_late'])
        dn = utility(rr['fixed_decode']) - utility(rr['fixed_late'])
        absent = rr['GRAM'] > 50
        out['generation_diagnosis'] = dict(
            target_outside_original50_then_guided_top10=int((absent & (rr['fixed_decode'] <= 10)).sum()),
            target_outside_original50_then_guided_top50=int((absent & (rr['fixed_decode'] <= 50)).sum()),
            target_original50_lost_from_guided50=int((~absent & (rr['fixed_decode'] > 50)).sum()),
            n10_improved_vs_late_target_originally_outside50=int(((dn > 1e-12) & absent).sum()),
            n10_improved_vs_late_target_originally_inside50=int(((dn > 1e-12) & ~absent).sum()),
            outside50_contribution_to_delta_N10=float(np.where(absent, dn, 0).mean()),
            inside50_contribution_to_delta_N10=float(np.where(~absent, dn, 0).mean()))
    return out


def run_domain(domain, manifest):
    folder = SOURCE / domain
    for name in ('samples.json', 'data.npz', 'candidates.jsonl', 'experts.npy', 'lime.npy',
                 'ranks.npz', 'analysis.json', 'data_config.json'):
        path = folder / name
        manifest[str(path.relative_to(ROOT))] = sha(path)
    samples = json.loads((folder / 'samples.json').read_text())
    with np.load(folder / 'data.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    with np.load(folder / 'ranks.npz', allow_pickle=False) as f:
        previous = {k: f[k] for k in f.files}
    analysis = json.loads((folder / 'analysis.json').read_text())
    experts = np.load(folder / 'experts.npy', mmap_mode='r', allow_pickle=False)
    lime = np.load(folder / 'lime.npy', mmap_mode='r', allow_pickle=False)
    assert experts.shape[0] == len(samples) == lime.shape[0]
    assert experts.shape[1] == 3 and experts.shape[2] == lime.shape[1]
    catalog_ids = np.arange(1, lime.shape[1] + 1)
    guided_by_user = {}
    with (folder / 'candidates.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if 'guided' in r:
                assert r['role'] == 'development'
                assert r['user_id'] not in guided_by_user
                guided_by_user[r['user_id']] = r
    assert len(guided_by_user) == 512
    check_indices = {0, len(data['users']) // 2, len(data['users']) - 1}
    result = []
    assert len(set(data['users'])) == len(data['users'])
    for j, user in enumerate(data['users']):
        i = int(data['sample_indices'][j]); s = samples[i]
        assert s['user_id'] == user and s['target'] == int(data['targets'][j])
        assert s['role'] == data['roles'][j] != 'gate_calibration'
        excluded = s['history'][-20:]
        assert excluded == s['full_history'][-20:]
        target = int(s['target']); eligible = ~np.isin(catalog_ids, excluded)
        row = dict(domain=domain, user_id=str(user), role=s['role'], target=target,
                   ineligible_target=not bool(eligible[target - 1]))
        ids = data['ids'][j]
        np.testing.assert_array_equal(data['valid'][j], ~np.isin(ids, excluded))
        row['GRAM'] = sorted_rank(ids, data['native'][j], target, excluded)
        row['fixed_late'] = sorted_rank(ids, data['mixed'][j], target, excluded)
        for name, score in [*zip(EXPERTS, experts[i]), ('LIME', lime[i])]:
            assert np.isfinite(score).all()
            row[name] = directory_rank(score, target, eligible)
            if j in check_indices:
                assert row[name] == sorted_rank(catalog_ids, score, target, excluded)
        if s['role'] == 'development':
            r = guided_by_user[str(user)]; g = r['guided']
            assert r['sample_index'] == i and r['target'] == target
            row['fixed_decode'] = sorted_rank(g['ids'], g['scores'], target, excluded)
        result.append(row)
        if (j + 1) % 256 == 0:
            write(OUT / 'status.json', dict(state='RUNNING', domain=domain,
                  completed=j + 1, total=len(data['users']), test_read=False))
    by_role = {}
    for role, prefix in [('ranker_fit', 'oof'), ('development', 'dev')]:
        selected = [r for r in result if r['role'] == role]
        old_users = previous['train_users' if role == 'ranker_fit' else 'development_users']
        np.testing.assert_array_equal([r['user_id'] for r in selected], old_users)
        for name in ('GRAM', 'fixed_late', *(('fixed_decode',) if role == 'development' else ())):
            np.testing.assert_array_equal([r[name] for r in selected], previous[f'{prefix}_{name}'])
        by_role[role] = analyze_cohort(selected)
        reference = analysis['oof' if role == 'ranker_fit' else 'development']['metrics']
        for name in ('GRAM', 'fixed_late', *(('fixed_decode',) if role == 'development' else ())):
            for k, value in reference[name].items():
                assert abs(by_role[role]['metrics'][name][k] - value) < 1e-12
    return by_role, result


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT / 'status.json', dict(state='RUNNING', stage='source_hashes', test_read=False))
    (OUT / 'protocol.md').write_text(PROTOCOL.read_text())
    (OUT / 'audit_source.py').write_text(Path(__file__).read_text())
    manifest = {str(PROTOCOL.relative_to(ROOT)): sha(PROTOCOL),
                str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    domains, records = {}, []
    try:
        for domain in ('Toys', 'Beauty'):
            domains[domain], rows = run_domain(domain, manifest)
            records.extend(rows)
            print(domain, 'complete', flush=True)
        fields = ['domain', 'user_id', 'role', 'target', 'ineligible_target',
                  'GRAM', 'fixed_late', *EXPERTS, 'LIME', 'fixed_decode']
        with (OUT / 'per_user.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(records)
        write(OUT / 'manifest.json', manifest)
        summary = dict(state='COMPLETED', domains=domains, users=len(records),
            seconds=time.monotonic() - started, test_read=False, optimizer_steps=0,
            evaluation='exploratory_cache_diagnostic', verification=dict(
                all_existing_GRAM_late_and_guided_ranks_exact=True,
                all_existing_metrics_exact_within_1e_12=True,
                full_catalog_count_ranks_checked_by_sort_on_three_fixed_rows_per_domain=True),
            limitations=['Expert-only hits are opportunities, not implementable gains.',
                        'Different candidate scopes; raw expert/GRAM scores are not compared.',
                        'No new mechanism, independent test, or causal attribution validated.'])
        write(OUT / 'summary.json', summary)
        write(OUT / 'status.json', {k: summary[k] for k in ('state','seconds','users','test_read','optimizer_steps','verification')})
        print(json.dumps(dict(state='COMPLETED', seconds=summary['seconds']), ensure_ascii=False))
    except BaseException as e:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        write(OUT / 'status.json', dict(state='FAILED', error=repr(e), test_read=False))
        raise


if __name__ == '__main__':
    main()
