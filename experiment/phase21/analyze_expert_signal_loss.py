"""CPU-only audit of existing expert/fusion/GRAM caches; no model execution."""
import csv
import hashlib
import json
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
REFERENCE = ROOT / 'artifacts/phase21/expert_complementarity_20260923_v1/per_user.csv'
OUT = ROOT / 'artifacts/phase21/expert_signal_loss_20260924_v1'
PROTOCOL = ROOT / 'plan/第二十一阶段/21.7_专家意见丢失位置缓存分析_20260924.md'
EXPERTS = ('SASRec', 'ItemCF', 'semantic')
ABSENT = 2**30


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def full_ranking(score, eligible):
    # Stable sort preserves ascending item ID for exactly equal scores.
    ordered = np.flatnonzero(eligible)[np.argsort(-score[eligible], kind='stable')] + 1
    ranks = np.full(len(score) + 1, ABSENT, dtype=np.int32)
    ranks[ordered] = np.arange(1, len(ordered) + 1)
    return ordered, ranks


def candidate_rank(ids, scores, target, excluded):
    ids, scores = np.asarray(ids), np.asarray(scores)
    valid = np.isfinite(scores) & ~np.isin(ids, excluded)
    ordered = ids[valid][np.lexsort((ids[valid], -scores[valid]))]
    pos = np.flatnonzero(ordered == target)
    return int(pos[0] + 1) if len(pos) else ABSENT


def utility(rank):
    rank = np.asarray(rank)
    return np.where(rank <= 10, 1 / np.log2(rank + 1), 0.)


def compare(new, old):
    new, old = np.asarray(new), np.asarray(old)
    delta = utility(new) - utility(old)
    rng = np.random.default_rng(20260924)
    boot = []
    for start in range(0, 2000, 100):
        ix = rng.integers(0, len(delta), size=(min(100, 2000-start), len(delta)))
        boot.extend(delta[ix].mean(1).tolist())
    return dict(users=len(new), new_H10=float((new <= 10).mean()),
                old_H10=float((old <= 10).mean()), new_N10=float(utility(new).mean()),
                old_N10=float(utility(old).mean()), delta_N10=float(delta.mean()),
                delta_N10_ci95=np.quantile(boot, [.025, .975]).tolist(),
                gained10=int(((new <= 10) & (old > 10)).sum()),
                lost10=int(((new > 10) & (old <= 10)).sum()),
                n10_improved=int((delta > 1e-12).sum()),
                n10_worsened=int((delta < -1e-12).sum()))


def length_bin(n):
    return '1-4' if n < 5 else '5-9' if n < 10 else '10-19' if n < 20 else '20+'


def rank_bin(rank):
    return '1' if rank == 1 else '2-3' if rank <= 3 else '4-10'


def passage_fields(passage):
    matches = list(re.finditer(r'(?:^|;\s*)(item|similar items|title|categories|brand|description|price|salesrank)\s*:', passage))
    return {m.group(1): passage[m.end():matches[j+1].start() if j+1 < len(matches) else len(passage)].strip()
            for j, m in enumerate(matches)}


class Groups:
    def __init__(self):
        self.data = defaultdict(lambda: dict(candidates=0, target_hits=0, users=set()))

    def add(self, key, user, hit):
        value = self.data[tuple(key)]
        value['candidates'] += 1
        value['target_hits'] += int(hit)
        value['users'].add(user)

    def rows(self):
        return [dict(group=list(k), candidates=v['candidates'], users=len(v['users']),
                     target_hits=v['target_hits'], target_hits_per_1000_candidates=1000*v['target_hits']/v['candidates'])
                for k, v in sorted(self.data.items())]


def cohort_summary(rows):
    names = ['GRAM', 'fixed_late', *EXPERTS, 'LIME', *['reserve_'+e for e in EXPERTS]]
    guided = rows[0]['role'] == 'development'
    if guided:
        names += ['fixed_decode']
    ranks = {n: np.asarray([r[n] for r in rows]) for n in names}
    expert_hit = np.stack([ranks[e] <= 10 for e in EXPERTS])
    fused = ranks['LIME'] <= 10
    opportunity = (ranks['GRAM'] > 10) & expert_hit.any(0)
    out = dict(users=len(rows), metrics={n: dict(H10=float((r <= 10).mean()), N10=float(utility(r).mean()))
                for n, r in ranks.items()}, experts_versus_fusion={}, reserve_versus_fusion={})
    for e in EXPERTS:
        out['experts_versus_fusion'][e] = compare(ranks[e], ranks['LIME'])
        out['reserve_versus_fusion'][e] = compare(ranks['reserve_'+e], ranks['LIME'])
    out['suppression'] = dict(expert_any_hit=int(expert_hit.any(0).sum()),
        expert_hit_fusion_miss=int((expert_hit.any(0) & ~fused).sum()),
        no_expert_top10_fusion_hit=int((~expert_hit.any(0) & fused).sum()),
        GRAM_miss_expert_hit=int(opportunity.sum()))
    for e, hits in zip(EXPERTS, expert_hit):
        out['suppression'][e] = dict(expert_hit=int(hits.sum()), fusion_rejected=int((hits & ~fused).sum()),
            GRAM_miss_expert_hit=int(((ranks['GRAM'] > 10) & hits).sum()))
    if guided:
        decoded = ranks['fixed_decode'] <= 10
        pending = opportunity & ~decoded
        out['opportunity_outcomes'] = dict(total=int(opportunity.sum()), realized=int((opportunity & decoded).sum()),
            unrealized=int(pending.sum()),
            fused_hit_guided_hit=int((opportunity & fused & decoded).sum()),
            fused_hit_guided_miss=int((opportunity & fused & ~decoded).sum()),
            fused_miss_guided_hit=int((opportunity & ~fused & decoded).sum()),
            fused_miss_guided_miss=int((opportunity & ~fused & ~decoded).sum()),
            unresolved_fusion_outside10=int((pending & ~fused).sum()),
            unresolved_fusion_inside10_guided_outside50=int((pending & fused & (ranks['fixed_decode'] > 50)).sum()),
            unresolved_fusion_inside10_guided_rank11_50=int((pending & fused & (ranks['fixed_decode'] <= 50)).sum()))
        parts = out['opportunity_outcomes']
        assert sum(parts[k] for k in ('unresolved_fusion_outside10',
            'unresolved_fusion_inside10_guided_outside50', 'unresolved_fusion_inside10_guided_rank11_50')) == parts['unrealized']
        out['guided_minus_GRAM'] = compare(ranks['fixed_decode'], ranks['GRAM'])
        out['guided_minus_late'] = compare(ranks['fixed_decode'], ranks['fixed_late'])
        for k in (10, 50):
            out['fusion_guided_cross_'+str(k)] = {
                'both_hit': int(((ranks['LIME'] <= k) & (ranks['fixed_decode'] <= k)).sum()),
                'fusion_only': int(((ranks['LIME'] <= k) & (ranks['fixed_decode'] > k)).sum()),
                'guided_only': int(((ranks['LIME'] > k) & (ranks['fixed_decode'] <= k)).sum()),
                'neither': int(((ranks['LIME'] > k) & (ranks['fixed_decode'] > k)).sum())}
        new_coverage = (ranks['GRAM'] > 50) & (ranks['fixed_decode'] <= 50)
        dn = utility(ranks['fixed_decode'])-utility(ranks['fixed_late'])
        out['coverage_vs_ranking'] = dict(new_target_in_50=int(new_coverage.sum()),
            new_target_in_10=int((new_coverage & decoded).sum()),
            target_dropped_from_50=int(((ranks['GRAM'] <= 50) & (ranks['fixed_decode'] > 50)).sum()),
            delta_N10_originally_outside50=float(np.where(ranks['GRAM'] > 50, dn, 0).mean()),
            delta_N10_originally_inside50=float(np.where(ranks['GRAM'] <= 50, dn, 0).mean()))
    # Predeclared observable strata. This is descriptive, not a learned selector.
    out['history_strata'] = {}
    for b in ('1-4', '5-9', '10-19', '20+'):
        ix = np.asarray([r['history_bin'] == b for r in rows])
        out['history_strata'][b] = dict(users=int(ix.sum()),
            opportunities=int((opportunity & ix).sum()),
            fusion_rejected_opportunities=int((opportunity & ~fused & ix).sum()))
        if guided:
            out['history_strata'][b]['unrealized'] = int((opportunity & ~decoded & ix).sum())
    return out


def run_domain(domain, reference, manifest):
    folder = SOURCE / domain
    source_names = ['samples.json', 'catalog.json', 'experts.npy', 'lime.npy', 'gate_prediction.npz',
                    'data.npz', 'candidates.jsonl', 'ranks.npz', 'analysis.json']
    initial_stats = {name: (folder/name).stat() for name in source_names}
    for name in source_names:
        manifest[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    samples = json.loads((folder/'samples.json').read_text())
    catalog = json.loads((folder/'catalog.json').read_text())
    experts = np.load(folder/'experts.npy', mmap_mode='r', allow_pickle=False)
    lime = np.load(folder/'lime.npy', mmap_mode='r', allow_pickle=False)
    with np.load(folder/'gate_prediction.npz', allow_pickle=False) as f:
        weights, penalties = f['weights'], f['penalty']
    with np.load(folder/'data.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    guided = {}
    with (folder/'candidates.jsonl').open() as f:
        for line in f:
            record = json.loads(line)
            if 'guided' in record:
                assert record['user_id'] not in guided
                guided[record['user_id']] = record
    assert len(guided) == 512
    assert set(data['roles']) == {'ranker_fit', 'development'}
    assert len(set(data['users'])) == len(data['users'])
    assert experts.shape == (len(samples), 3, len(catalog['items'])-1)
    group = Groups()
    rows, details = [], []
    max_error = 0.
    candidate_columns = ['domain','role','user_id','item','target_hit','support_count','support_experts',
                         'fusion_admitted','history_bin','SASRec_rank','ItemCF_rank','semantic_rank','LIME_rank',
                         'SASRec_score','ItemCF_score','semantic_score','LIME_score',
                         'SASRec_weight','ItemCF_weight','semantic_weight']
    with (OUT/(domain+'_candidates.csv')).open('w', newline='') as cf:
        writer = csv.DictWriter(cf, candidate_columns); writer.writeheader()
        for j, user in enumerate(data['users']):
            i = int(data['sample_indices'][j]); sample = samples[i]
            target, role = int(data['targets'][j]), str(data['roles'][j])
            assert sample['user_id'] == user and sample['target'] == target and sample['role'] == role
            assert sample['history'][-20:] == sample['full_history'][-20:]
            excluded = sample['history'][-20:]
            eligible = np.ones(lime.shape[1], bool); eligible[np.asarray(excluded)-1] = False
            assert eligible[target-1]
            hist_mask = np.zeros(lime.shape[1], bool)
            hist_mask[np.asarray(sample['full_history'])-1] = True
            recovered = (weights[i,:,None]*experts[i]).sum(0)-penalties[i]*hist_mask
            err = float(np.max(np.abs(recovered-lime[i])))
            max_error = max(err, max_error)
            assert err < 2e-7, (domain, j, err)
            ordered, rank_maps = {}, {}
            for name, score in [*zip(EXPERTS, experts[i]), ('LIME', lime[i])]:
                ordered[name], rank_maps[name] = full_ranking(score, eligible)
            row = dict(domain=domain, role=role, user_id=str(user), sample_index=i, target=target,
                       history_length=len(sample['full_history']), history_bin=length_bin(len(sample['full_history'])))
            row.update({name: int(rr[target]) for name, rr in rank_maps.items()})
            row['GRAM'] = candidate_rank(data['ids'][j], data['native'][j], target, excluded)
            row['fixed_late'] = candidate_rank(data['ids'][j], data['mixed'][j], target, excluded)
            if role == 'development':
                record = guided[str(user)]
                assert record['sample_index'] == i and record['target'] == target
                g = record['guided']
                row['fixed_decode'] = candidate_rank(g['ids'], g['scores'], target, excluded)
            ref = reference[(domain, str(user))]
            assert ref['role'] == role and int(ref['target']) == target
            for name in ['GRAM','fixed_late',*EXPERTS,'LIME'] + (['fixed_decode'] if role == 'development' else []):
                assert row[name] == int(ref[name]), (domain, user, name, row[name], ref[name])
            fused10 = set(ordered['LIME'][:10].tolist())
            for e in EXPERTS:
                candidate = int(next(x for x in ordered[e] if x not in fused10))
                row['reserve_'+e+'_item'] = candidate
                row['reserve_'+e] = row['LIME'] if row['LIME'] <= 9 else 10 if candidate == target else ABSENT
            target_support = [e for e in EXPERTS if row[e] <= 10]
            row['target_support_experts'] = '+'.join(target_support)
            row['target_support_count'] = len(target_support)
            opportunity = row['GRAM'] > 10 and bool(target_support)
            if role != 'development':
                row['outcome'] = 'training_no_guided_cache'
            elif not opportunity:
                row['outcome'] = 'outside_expert_opportunity_subset'
            elif row['fixed_decode'] <= 10:
                row['outcome'] = 'realized'
            elif row['LIME'] > 10:
                row['outcome'] = 'fusion_outside10'
            elif row['fixed_decode'] > 50:
                row['outcome'] = 'fusion_inside10_guided_outside50'
            else:
                row['outcome'] = 'fusion_inside10_guided_rank11_50'
            if target_support and row['LIME'] > 10:
                boundary = int(ordered['LIME'][9])
                margins = weights[i].astype(float)*(experts[i,:,target-1].astype(float)-experts[i,:,boundary-1].astype(float))
                penalty_margin = -float(penalties[i])*(int(hist_mask[target-1])-int(hist_mask[boundary-1]))
                gap = float(lime[i,target-1]-lime[i,boundary-1])
                assert abs(float(margins.sum())+penalty_margin-gap) < 2e-7
                row['fusion_boundary_item'] = boundary
                row['fusion_boundary_gap'] = gap
                row['history_penalty_margin'] = penalty_margin
                for e, w, m in zip(EXPERTS, weights[i], margins):
                    row[e+'_weight'] = float(w)
                    row[e+'_boundary_margin'] = float(m)
            candidate_union = sorted(set().union(*(set(o[:10].tolist()) for o in ordered.values())))
            for item in candidate_union:
                support = [e for e in EXPERTS if rank_maps[e][item] <= 10]
                pattern = '+'.join(support) or 'none'
                admitted = item in fused10
                hit = item == target
                key = [role, 'pattern', pattern, str(admitted)]
                group.add(key, user, hit)
                group.add([role, 'support_count', str(len(support)), str(admitted)], user, hit)
                group.add([role, 'history', row['history_bin'], pattern, str(admitted)], user, hit)
                if len(support) == 1:
                    group.add([role, 'exclusive_rank', support[0], rank_bin(int(rank_maps[support[0]][item])), str(admitted)], user, hit)
                c = dict(domain=domain, role=role, user_id=str(user), item=item, target_hit=int(hit),
                    support_count=len(support), support_experts=pattern, fusion_admitted=int(admitted), history_bin=row['history_bin'])
                c.update({e+'_rank': int(rank_maps[e][item]) for e in (*EXPERTS, 'LIME')})
                for e, s, w in zip(EXPERTS, experts[i,:,item-1], weights[i]):
                    c[e+'_score'], c[e+'_weight'] = float(s), float(w)
                c['LIME_score'] = float(lime[i,item-1]); writer.writerow(c)
            rows.append(row)
            # Save one deterministically selected case per stage and expert contrast.
            case_tags = [row['outcome']] if role == 'development' and opportunity else []
            if role == 'development':
                for e in EXPERTS:
                    if row[e] <= 10 and row['LIME'] > 10:
                        case_tags.append(e+'_correct_fusion_miss')
                    if row[e] > 10 and row['LIME'] <= 10:
                        case_tags.append(e+'_miss_fusion_correct')
            if case_tags:
                fields = passage_fields(catalog['passages'][target])
                d = dict(row, tags=case_tags, target_title=fields.get('title'), target_brand=fields.get('brand'),
                         target_categories=fields.get('categories'),
                         weights={e: float(w) for e,w in zip(EXPERTS,weights[i])},
                         history_last5=[dict(item=h, title=passage_fields(catalog['passages'][h]).get('title')) for h in sample['history'][-5:]])
                if 'fusion_boundary_item' in row:
                    d['fusion_boundary_title'] = passage_fields(catalog['passages'][row['fusion_boundary_item']]).get('title')
                details.append(d)
            if (j+1) % 256 == 0:
                write(OUT/'status.json', dict(state='RUNNING', domain=domain, completed=j+1, total=len(data['users']),
                                            pid=os.getpid(), test_read=False, gpu_inferences=0, optimizer_steps=0))
                print(domain, j+1, '/', len(data['users']), flush=True)
    examples = {}
    for d in sorted(details, key=lambda r: r['user_id']):
        for tag in d['tags']:
            examples.setdefault(tag, d)
    for name in source_names:
        after = (folder/name).stat(); before = initial_stats[name]
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    result = {role: cohort_summary([r for r in rows if r['role'] == role]) for role in ('ranker_fit','development')}
    expected = json.loads((ROOT/'artifacts/phase21/expert_complementarity_20260923_v1/summary.json').read_text())['domains'][domain]
    for role in result:
        assert result[role]['users'] == expected[role]['users']
        for name in ['GRAM','fixed_late',*EXPERTS,'LIME'] + (['fixed_decode'] if role == 'development' else []):
            for m, val in result[role]['metrics'][name].items():
                assert abs(val-expected[role]['metrics'][name][m]) < 1e-12
    result['candidate_groups'] = group.rows()
    result['verification'] = dict(all_reference_ranks_exact=True, baseline_metrics_exact=True,
        max_fusion_reconstruction_error=max_error, sources_unmodified=True, users=len(rows))
    write(OUT/(domain+'_summary.json'), result)
    write(OUT/(domain+'_cases.json'), examples)
    write(OUT/(domain+'_all_development_cases.json'), details)
    return result, rows


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='RUNNING', stage='hash_inputs', pid=os.getpid(), test_read=False,
                                gpu_inferences=0, optimizer_steps=0))
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    (OUT/'analysis_source.py').write_text(Path(__file__).read_text())
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in (REFERENCE, PROTOCOL, Path(__file__))}
    try:
        with REFERENCE.open() as f:
            reference = {(r['domain'], r['user_id']): r for r in csv.DictReader(f)}
        results, rows = {}, []
        for domain in ('Toys','Beauty'):
            results[domain], part = run_domain(domain, reference, manifest)
            rows.extend(part)
        assert len(rows) == len(reference) == 7107
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/'per_user.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, columns); writer.writeheader(); writer.writerows(rows)
        write(OUT/'manifest.json', manifest)
        summary = dict(state='COMPLETED', domains=results, users=len(rows), seconds=time.monotonic()-start,
                       gpu_inferences=0, optimizer_steps=0, test_read=False, exploratory=True,
                       actual_guided_step_trace_available=False, bootstrap_replicates=2000,
                       bootstrap_unit='user_within_domain_and_role', independent_confirmation=False)
        write(OUT/'summary.json', summary)
        write(OUT/'status.json', {k:v for k,v in summary.items() if k != 'domains'})
        print(json.dumps({k:v for k,v in summary.items() if k != 'domains'}), flush=True)
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write(OUT/'status.json', dict(state='FAILED', error=repr(error), pid=os.getpid(), test_read=False))
        raise


if __name__ == '__main__':
    main()
