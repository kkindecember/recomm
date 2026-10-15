"""Post-hoc diagnostic; this is not a new equal-budget decoding experiment."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ('original', 'lime_decode_screen_20260921_v1/full_parallel_v4', 'lime'),
    ('mechanism', 'lime_mechanism_screen_20260921_v1', 'lime'),
    ('prefix', 'lime_prefix_expert_20260921_v1', 'fixed_cal'),
)


def rank(items, target):
    return items.index(target) + 1 if target in items else 51


def ndcg10(r):
    return 1 / math.log2(r + 1) if r <= 10 else 0.


def ordered(scores, history):
    return sorted((i for i in scores if i not in history),
                  key=lambda i: (-scores[i], i))[:50]


def main():
    report = {
        'purpose': 'Check whether union of two cached searches helps with unchanged scores',
        'limitations': [
            'Post-hoc exploratory analysis on previously inspected development cohorts',
            'Union uses more candidates/compute; not an equal-budget method comparison',
            'A null result cannot rule out other search strategies or scoring changes',
            'Oracle per-user choice uses target and is not a deployable method',
        ],
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'results': [],
    }
    for cohort, directory, arm in SOURCES:
        for domain in ('Toys', 'Beauty'):
            path = ROOT / 'artifacts/phase20' / directory / domain / 'evaluation.jsonl'
            content = path.read_bytes()
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            totals = dict(guided_ndcg10=0., union_ndcg10=0.,
                          oracle_choice_ndcg10=0., mean_raw_union_size=0.)
            max_shared_score_error = 0.
            gained50 = lost50 = net_hits10 = 0
            for row in rows:
                history, target = set(row['history']), row['target']
                c = row['candidate_scores'][arm]
                guided = dict(zip(c['ids'], c['scores']))
                c = row['candidate_scores'][arm + '_late']
                original = dict(zip(c['ids'], c['scores']))
                assert len(guided) == len(original) == 50
                for i in guided.keys() & original.keys():
                    max_shared_score_error = max(max_shared_score_error,
                                                 abs(guided[i] - original[i]))
                merged = {**original, **guided}
                base_items = ordered(guided, history)
                assert base_items == row['arms'][arm]['items']
                base_rank = rank(base_items, target)
                assert base_rank == row['arms'][arm]['rank']
                union_rank = rank(ordered(merged, history), target)
                original_rank = rank(ordered(original, history), target)
                totals['guided_ndcg10'] += ndcg10(base_rank)
                totals['union_ndcg10'] += ndcg10(union_rank)
                totals['oracle_choice_ndcg10'] += ndcg10(min(base_rank, original_rank))
                totals['mean_raw_union_size'] += len(merged)
                gained50 += int(base_rank > 50 and union_rank <= 50)
                lost50 += int(base_rank <= 50 and union_rank > 50)
                net_hits10 += int(union_rank <= 10) - int(base_rank <= 10)
            assert rows and max_shared_score_error <= 5e-6
            totals = {k: v / len(rows) for k, v in totals.items()}
            result = dict(cohort=cohort, domain=domain, users=len(rows),
                          input=str(path.relative_to(ROOT)),
                          input_sha256=hashlib.sha256(content).hexdigest(),
                          max_shared_score_error=max_shared_score_error,
                          gained_hits50=gained50, lost_hits50=lost50,
                          net_hits10=net_hits10, **totals)
            result['relative_ndcg10_change'] = totals['union_ndcg10'] / totals['guided_ndcg10'] - 1
            report['results'].append(result)
    output = ROOT / 'artifacts/phase20/lime_next_mechanism_review_20260921_v1/cached_union.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'rows': [
        {k: row[k] for k in ('cohort', 'domain', 'guided_ndcg10', 'union_ndcg10',
                             'relative_ndcg10_change')} for row in report['results']
    ]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
