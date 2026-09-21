"""Post-result rank accounting from frozen caches; no model calls or fitting."""
import csv
import json
import math
import time
from pathlib import Path

from experiment.phase20.guided_decoding_job import ROOT, now, read, sha, write


SOURCE = ROOT / 'artifacts/phase20/guided_decoding_20260919_v1'
OUT = SOURCE / 'rank_damage_diagnostic'


def ordered(row, scores):
    return [item for item, _ in sorted(zip(row['candidate_ids'], scores), key=lambda x: -x[1])]


def gain(rank):
    return 1 / math.log2(rank + 1) if rank <= 10 else 0.0


def analyze(domain):
    source = SOURCE / domain / 'evaluation.jsonl'
    before_hash = sha(source)
    assert before_hash == read(SOURCE / domain / 'verification.json')['output_sha256']['evaluation.jsonl']
    pairs = {}
    for line in source.open():
        row = json.loads(line)
        if row['arm'] in ('vanilla', 'real_cf'):
            pair = pairs.setdefault(row['user_id'], {})
            assert row['arm'] not in pair
            pair[row['arm']] = row
    assert len(pairs) == 512
    results = []
    stage_changes = {s: dict(improved=0, unchanged=0, worsened=0) for s in ('native', 'b1', 'raw_softmax', 'softmax')}
    max_feature_diff = [0.0] * 6
    shared_items = rank_feature_changed = reliability_changed = 0
    max_cf_diff = 0.0
    for user, pair in pairs.items():
        b, c = pair['vanilla'], pair['real_cf']
        assert b['target'] == c['target']
        target = b['target']
        bi = {item: i for i, item in enumerate(b['candidate_ids'])}
        ci = {item: i for i, item in enumerate(c['candidate_ids'])}
        common = bi.keys() & ci.keys()
        reliability_changed += int(abs(b['reliability'] - c['reliability']) > 1e-12)
        for item in common:
            shared_items += 1
            diffs = [abs(x-y) for x, y in zip(b['features'][bi[item]], c['features'][ci[item]])]
            max_feature_diff = [max(x, y) for x, y in zip(max_feature_diff, diffs)]
            rank_feature_changed += int(diffs[1] > 1e-6)
            max_cf_diff = max(max_cf_diff, *(abs(x[bi[item]]-y[ci[item]]) for x, y in zip(b['cf_scores'], c['cf_scores'])))
        old_order = ordered(b, b['scores']['softmax'])
        new_order = ordered(c, c['scores']['softmax'])
        old_rank = old_order.index(target)+1 if target in bi else 51
        new_rank = new_order.index(target)+1 if target in ci else 51
        assert (old_rank, new_rank) == (b['ranks']['softmax'], c['ranks']['softmax'])
        category = ('both' if target in common else 'newly_covered' if target in ci
                    else 'lost_coverage' if target in bi else 'neither')
        r = dict(user_id=user, target=target, coverage=category,
                 old_rank=old_rank, new_rank=new_rank, N10_change=gain(new_rank)-gain(old_rank),
                 top10_gained=int(new_rank <= 10 < old_rank), top10_lost=int(old_rank <= 10 < new_rank),
                 shared_candidates=len(common), old_reliability=b['reliability'], new_reliability=c['reliability'],
                 old_shared_above=None, new_shared_above=None, removed_above=None, added_above=None,
                 shared_rank_delta=None, exclusive_rank_delta=None)
        if target in common:
            old_above, new_above = old_order[:old_rank-1], new_order[:new_rank-1]
            r.update(old_shared_above=sum(i in common for i in old_above),
                     new_shared_above=sum(i in common for i in new_above),
                     removed_above=sum(i not in common for i in old_above),
                     added_above=sum(i not in common for i in new_above))
            r['shared_rank_delta'] = r['new_shared_above']-r['old_shared_above']
            r['exclusive_rank_delta'] = r['added_above']-r['removed_above']
            assert new_rank-old_rank == r['shared_rank_delta']+r['exclusive_rank_delta']
            for stage in stage_changes:
                def scores(row):
                    if stage == 'b1': return row['b1_scores']
                    if stage == 'raw_softmax': return row['raw_head_scores']['softmax']
                    return row['scores'][stage]
                ranks = [[i for i in ordered(row, scores(row)) if i in common].index(target)+1 for row in (b, c)]
                delta = ranks[1]-ranks[0]
                stage_changes[stage]['worsened' if delta > 0 else 'improved' if delta < 0 else 'unchanged'] += 1
        results.append(r)
    assert sha(source) == before_hash
    with (OUT / f'{domain}.users.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    transitions = [r for r in results if r['top10_gained'] or r['top10_lost']]
    write(OUT / f'{domain}.top10_transitions.json', transitions)
    actual_delta = sum(r['N10_change'] for r in results)/512
    expected = read(SOURCE / domain / 'summary.json')['contrasts']['real_cf.softmax__vs__vanilla.softmax']['delta_N10']
    assert abs(actual_delta-expected) < 1e-12
    return dict(users=512, source_sha256=before_hash, delta_N10=actual_delta,
                top10_gained=sum(r['top10_gained'] for r in results),
                top10_lost=sum(r['top10_lost'] for r in results),
                shared_items=shared_items, shared_item_rank_feature_changed=rank_feature_changed,
                shared_item_max_feature_difference=max_feature_diff,
                shared_item_max_raw_cf_difference=max_cf_diff,
                reliability_changed_users=reliability_changed,
                shared_target_rank_changes_by_stage=stage_changes,
                lost_top10_cases=[r for r in transitions if r['top10_lost']])


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT / 'status.json', dict(state='RUNNING', started_at=now(), new_gpu_inference=0, new_training_steps=0))
    write(OUT / 'protocol.json', dict(created_at=now(), analysis_type='post_result_descriptive_accounting',
          question='Separate shared-item reordering from incoming/outgoing competitors above the target.',
          identity='new_rank-old_rank = new_shared_above-old_shared_above + added_above-removed_above',
          population='All 512 evaluation users in each domain; real_cf versus vanilla; frozen softmax primary.',
          restrictions='No parameter selection, new scoring rule, model fitting, new generation, or test read.',
          limitation='Rank terms add exactly; this does not identify separate causal effects or additive NDCG contributions.'))
    try:
        summary = dict(domains={d: analyze(d) for d in ('Toys', 'Beauty')},
                       analysis_source_sha256=sha(Path(__file__)), new_gpu_inference=0, new_training_steps=0,
                       test_read=False, reused_development_validation=True)
        write(OUT / 'summary.json', summary)
        write(OUT / 'status.json', dict(state='COMPLETED', finished_at=now(), elapsed_seconds=time.monotonic()-started,
                                      verified='Cache hashes, 1024 target ranks, shared-target rank identities, aggregate N10 deltas.',
                                      new_gpu_inference=0, new_training_steps=0))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        write(OUT / 'status.json', dict(state='FAILED', finished_at=now(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
