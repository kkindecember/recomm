"""Read-only input audit: align public review dates to frozen training histories.

No model fitting, validation/test metrics, sequence reordering, or dataset changes.
Only dates for user/item pairs in the existing upstream training histories are kept.
Timestamp availability and variation do not establish predictive usefulness.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    return {
        'count': len(values), 'mean': statistics.mean(values),
        'median': statistics.median(values), 'min': ordered[0],
        'p90_nearest_rank': ordered[max(0, (9 * len(values) + 9) // 10 - 1)],
        'max': ordered[-1],
    }


def audit(domain, raw):
    history_path = (ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1'
                    / domain / 'upstream_sequences_without_terminal.jsonl')
    catalog_path = (ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
                    / domain / 'catalog.json')
    catalog = json.loads(catalog_path.read_text())['items']
    records = [json.loads(line) for line in history_path.open() if line.strip()]
    histories = {r['user_id']: [catalog[i] for i in r['history']] for r in records}
    assert len(histories) == len(records)
    wanted = {(user, item) for user, hist in histories.items() for item in hist}
    matched = defaultdict(list)
    source_rows = 0
    with gzip.open(raw, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            source_rows += 1
            key = (row['reviewerID'], row['asin'])
            if key in wanted:
                matched[key].append(row.get('unixReviewTime'))
    missing = wanted - matched.keys()
    ambiguous = {key for key, vals in matched.items() if len(vals) != 1}
    invalid = {key for key, vals in matched.items()
               if len(vals) == 1 and (not isinstance(vals[0], int) or vals[0] <= 0)}
    usable = {key: vals[0] for key, vals in matched.items()
              if key not in ambiguous and key not in invalid}
    gaps, spans, unique_dates, lengths = [], [], [], []
    full_users = nonmonotonic_users = same_date_users = tied_users = 0
    at_least_three_dates = 0
    for user, hist in histories.items():
        lengths.append(len(hist))
        if not all((user, item) in usable for item in hist):
            continue
        full_users += 1
        times = [usable[(user, item)] for item in hist]
        delta = [b - a for a, b in zip(times, times[1:])]
        if any(d < 0 for d in delta):
            nonmonotonic_users += 1
            continue
        distinct = len(set(times))
        unique_dates.append(distinct)
        same_date_users += distinct == 1
        at_least_three_dates += distinct >= 3
        tied_users += any(d == 0 for d in delta)
        gaps.extend(d / 86400 for d in delta)
        spans.append((times[-1] - times[0]) / 86400)
    n = len(histories)
    return {
        'domain': domain,
        'scope': 'Frozen upstream training histories only; original order preserved.',
        'timestamp_semantics': 'Amazon unixReviewTime: review date, not purchase/click time.',
        'source_rows_scanned': source_rows,
        'inputs': {str(p): sha256(p) for p in [raw, history_path, catalog_path]},
        'training_users': n, 'training_events': sum(lengths),
        'training_unique_user_item_pairs': len(wanted),
        'repeated_pairs_in_training_histories': sum(lengths) - len(wanted),
        'matched_unique_pairs': len(matched), 'missing_pairs': len(missing),
        'ambiguous_pairs': len(ambiguous), 'invalid_timestamp_pairs': len(invalid),
        'users_fully_aligned': full_users, 'nonmonotonic_users': nonmonotonic_users,
        'fraction_timestamps_on_utc_day_boundary': (
            sum(t % 86400 == 0 for t in usable.values()) / len(usable) if usable else None),
        'history_length': distribution(lengths),
        'fraction_histories_at_most_10': sum(k <= 10 for k in lengths) / n,
        'adjacent_gap_days': distribution(gaps),
        'fraction_adjacent_gaps_zero': sum(d == 0 for d in gaps) / len(gaps) if gaps else None,
        'fraction_adjacent_gaps_over_30_days': sum(d > 30 for d in gaps) / len(gaps) if gaps else None,
        'history_span_days': distribution(spans),
        'unique_review_dates_per_history': distribution(unique_dates),
        'users_all_history_on_one_date': same_date_users,
        'users_with_any_adjacent_date_tie': tied_users,
        'users_with_at_least_three_dates': at_least_three_dates,
        'predictive_value_tested': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beauty', type=Path, required=True)
    parser.add_argument('--toys', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / 'status.json'
    status.write_text(json.dumps({'state': 'RUNNING', 'kind': 'CPU timestamp alignment audit'}))
    results = [audit('Beauty', args.beauty), audit('Toys', args.toys)]
    payload = {'kind': 'Data availability/descriptive audit, not a model experiment',
               'domains': results, 'predictive_value_tested': False}
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    status.write_text(json.dumps({'state': 'COMPLETED', 'summary': 'summary.json',
                                 'predictive_value_tested': False}, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
