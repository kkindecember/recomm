"""Separate aggregate and scalar event replay of the expert evidence audit."""
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from analyze_selective_evidence import sha, write
from analyze_history_evidence import fields
from evidence_decomposition_core import read_users

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/expert_evidence_alignment_20260926_v1'
RAW = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
PRIOR = ROOT / 'artifacts/phase21/selective_evidence_20260925_v1'
BLIND = ROOT / 'artifacts/phase21/blind_disagreements_20260925_v1'


def read(p):
    return json.loads(p.read_text())


def decode(row):
    parsed = {}
    for k, v in row.items():
        if v in ('True', 'False'):
            parsed[k] = v == 'True'
        elif v == '':
            parsed[k] = None
        elif k in ('domain', 'user_id', 'stratum', 'a_title', 'b_title') or k.endswith('_vote'):
            parsed[k] = v
        else:
            parsed[k] = float(v)
    return parsed


def check_group(rows, summary):
    assert len(rows) == summary['users']
    for label in ('a_hit', 'b_hit', 'neither'):
        assert sum(r['stratum'] == label for r in rows) == summary[label]
    for e in ('sas', 'cf', 'text'):
        s = summary['experts'][e]
        votes = Counter(r[e + '_vote'] for r in rows)
        for role in ('A', 'B', 'tie'):
            assert votes[role] == s[role]
        answers = [r for r in rows if r['stratum'] != 'neither']
        agree = sum(r[e + '_vote'].lower() + '_hit' == r['stratum'] for r in answers)
        ties = sum(r[e + '_vote'] == 'tie' for r in answers)
        assert (agree, len(answers) - agree - ties, ties) == (s['target_agree'], s['target_disagree'], s['target_tie'])
        if e == 'sas':
            continue
        for key in ('conflicting_events', 'no_contribution', 'single_event_flip'):
            assert sum(r[e + '_' + key] for r in rows) == s[key]
        values = [r[e + '_cancellation'] for r in rows if r[e + '_cancellation'] is not None]
        assert len(values) == s['cancellation_n']
        if values:
            np.testing.assert_allclose(np.percentile(values, [0, 25, 50, 75, 100]), s['cancellation_quantiles'], atol=1e-12)


def manual_counts(cards):
    result = {}
    for name, subset in [('all24', [c for c in cards if c['review']['preference'] != 'abstain']),
                         ('target_in_pair', [c for c in cards if c['review']['preference'] != 'abstain' and c['review']['stratum'] != 'neither']),
                         ('manual_agrees', [c for c in cards if c['review']['agrees_with_observed_target'] is True]),
                         ('manual_disagrees', [c for c in cards if c['review']['agrees_with_observed_target'] is False]),
                         ('neither', [c for c in cards if c['review']['preference'] != 'abstain' and c['review']['stratum'] == 'neither'])]:
        one = dict(cards=len(subset), experts={})
        for e in ('sas', 'cf', 'text'):
            aligned = [c for c in subset if c['row'][e + '_vote'] == c['review']['chosen_role']]
            ties = sum(c['row'][e + '_vote'] == 'tie' for c in subset)
            s = dict(agrees_manual=len(aligned), opposes_manual=len(subset) - len(aligned) - ties, tie=ties)
            if e != 'sas':
                overlap = 0
                for c in aligned:
                    role = c['review']['chosen_role']; sign = 1 if role == 'A' else -1
                    sources = sorted(c[e]['sources'], key=lambda r: (-sign * r['difference'], r['ref']))
                    assert sources[0]['difference'] * sign > 0
                    overlap += sources[0]['ref'] in c['review'][c['review']['preference'] + '_history']
                s['top_support_in_cited_history'] = overlap
            one['experts'][e] = s
        result[name] = one
    return result


def main():
    start = time.monotonic(); manifest = read(OUT / 'input_manifest.json')
    for path, digest in manifest.items():
        assert sha(ROOT / path) == digest, path
    summary = read(OUT / 'summary.json'); cards = read(OUT / 'cards.json')
    sealed = read(BLIND / 'sealed_labels.json')
    previous = {r['card_id']: r for r in read(BLIND / 'unblinded_reviews.json')}
    checks = Counter(); max_cf = max_text = 0.
    for domain in ('Toys', 'Beauty'):
        with (OUT / (domain + '_users.csv')).open() as f:
            rows = [decode(r) for r in csv.DictReader(f)]
        assert len({r['user_id'] for r in rows}) == len(rows)
        by_user = {r['user_id']: r for r in rows}
        users = read_users(PRIOR / domain / 'validation/users.csv')
        with np.load(PRIOR / domain / 'validation/pairs.npz') as z:
            items = z['items']
        scores = np.load(PRIOR / domain / 'validation/score.npy', mmap_mode='r')[:len(items)]
        expected_users = {r['user_id'] for r in users if r['cohort'] == 'primary' and r['user_id'] not in sealed['excluded_previously_shown'][domain] and r['end'] > r['start']}
        assert set(by_user) == expected_users
        for u in users:
            if u['user_id'] not in by_user:
                continue
            ranked = sorted(range(u['start'], u['end']), key=lambda j: (-float(scores[j, 18]), int(items[j])))
            j = ranked[0]; a = int(items[j]); row = by_user[u['user_id']]
            assert (a, u['boundary'], u['target']) == (row['a'], row['b'], row['target'])
            label = 'a_hit' if a == u['target'] else 'b_hit' if u['boundary'] == u['target'] else 'neither'
            assert row['stratum'] == label
            for k, e in enumerate(('sas', 'cf', 'text')):
                diff = float(scores[j, k] - scores[j, k + 3])
                v = 'A' if diff > 1e-6 else 'B' if diff < -1e-6 else 'tie'
                assert diff == row[e + '_margin'] and v == row[e + '_vote']
            checks['target_free_pair_and_votes'] += 1
        s = summary['domains'][domain]
        groups = [(rows, s['all']), ([r for r in rows if r['reviewed']], s['reviewed']),
                  ([r for r in rows if not r['reviewed']], s['not_reviewed'])]
        groups += [([r for r in rows if r['stratum'] == label], s['strata'][label]) for label in ('a_hit', 'b_hit', 'neither')]
        for group, result in groups:
            check_group(group, result); checks['aggregate_sections'] += 1
        subset = [c for c in cards if c['row']['domain'] == domain]
        assert manual_counts(subset) == s['manual_alignment']
        cfg = read(RAW / domain / 'data_config.json'); catalog = read(RAW / domain / 'catalog.json')
        with np.load(ROOT / cfg['embedding']) as z:
            emb = z['embeddings']; assert z['item_ids'].tolist() == catalog['items'][1:]
        cf = read(ROOT / cfg['itemcf'])['neighbors']
        for c in subset:
            assert c['review'] == previous[c['card_id']]
            r = c['row']; a, b, h = r['a'], r['b'], c['history']
            assert (a, b, h) == (sealed['cards'][c['card_id']]['a'], sealed['cards'][c['card_id']]['b'], sealed['cards'][c['card_id']]['history'])
            # Match the scorer's float32 row reduction. A one-dimensional norm
            # may use a different BLAS reduction and promote the scalar epsilon.
            a_unit = emb[a - 1] / (np.sqrt(np.sum(emb[a - 1] ** 2)) + np.float32(1e-8))
            b_unit = emb[b - 1] / (np.sqrt(np.sum(emb[b - 1] ** 2)) + np.float32(1e-8))
            w = np.asarray([np.exp(-.1 * t) for t in range(len(h) - 1, -1, -1)], np.float32)
            mean = (emb[np.asarray(h) - 1] * w[:, None]).sum(0) / w.sum()
            denom = float(w.sum()) * float(np.linalg.norm(mean) + 1e-8)
            expected_cf, expected_text = [], []
            for j, item in enumerate(h):
                neighbors = {v['item_id']: float(np.float32(v['score'])) for v in cf.get(catalog['items'][item], [])}
                weight = (j - max(0, len(h) - 20) + 1) / min(20, len(h)) if j >= len(h) - 20 else 0.
                expected_cf.append([weight * neighbors.get(catalog['items'][v], 0.) for v in (a, b)])
                expected_text.append([float(w[j]) * float(np.dot(emb[item - 1].astype(float), v.astype(float))) / denom for v in (a_unit, b_unit)])
                checks['scalar_event_replays'] += 1
            for e, expected in [('cf', expected_cf), ('text', expected_text)]:
                values = np.asarray(expected); saved = np.asarray(c[e]['events'])
                error = float(np.max(np.abs(values - saved)))
                if e == 'cf': max_cf = max(max_cf, error)
                else: max_text = max(max_text, error)
                np.testing.assert_allclose(values, saved, atol=1e-8, rtol=1e-6)
                delta = values[:, 0] - values[:, 1]; total = float(delta.sum()); mass = float(np.abs(delta).sum())
                tol = max(1e-12, mass * 1e-6)
                assert abs(total - r[e + '_raw_margin']) < 1e-8, (c['card_id'], e, total, r[e + '_raw_margin'])
                assert int((delta > tol).sum()) == r[e + '_positive_events']
                assert int((delta < -tol).sum()) == r[e + '_negative_events']
                flip = False
                for j in range(len(h)):
                    remainder = sum(float(d) for k, d in enumerate(delta) if k != j)
                    if abs(total) > tol and remainder * np.sign(total) < -tol:
                        flip = True
                    if e == 'text':
                        # Re-form the pooled vector after deletion; positive normalization
                        # cannot change the pairwise sign. Remaining weights are held fixed.
                        keep = [k for k in range(len(h)) if k != j]
                        vector = sum((float(w[k]) * emb[h[k] - 1].astype(float) for k in keep), np.zeros(emb.shape[1]))
                        new_margin = float(vector @ (a_unit.astype(float) - b_unit.astype(float)))
                        if abs(remainder) > 1e-10:
                            assert new_margin * remainder > 0
                    checks['fixed_weight_event_deletions'] += 1
                assert flip == r[e + '_single_event_flip']
                for source in c[e]['sources']:
                    pos = [k for k, item in enumerate(h) if item == source['item']]
                    assert source['positions'] == [k + 1 for k in pos]
                    assert source['title'] == fields(catalog['passages'][source['item']]).get('title', '')
                    np.testing.assert_allclose(values[pos].sum(0), [source['a'], source['b']], atol=1e-8)
                    checks['source_rows'] += 1
            checks['review_cards'] += 1
    assert manual_counts(cards) == summary['manual_alignment']
    frozen = read(BLIND / 'review_freeze.json')
    assert sha(BLIND / 'frozen_review.json') == frozen['frozen_review_sha256']
    assert sha(BLIND / 'review_entries.json') == frozen['raw_review_sha256']
    assert read(ROOT / 'artifacts/phase21/expert_feedback_20260924_v1/status.json')['state'] == 'USER_PAUSED'
    result = dict(state='VERIFIED', source_hashes=len(manifest), checks=dict(checks),
                  scalar_replay_max_absolute_error=dict(cf=max_cf, text=max_text),
                  frozen_blind_review_unchanged=True, semantic_validity_independently_reviewed=False,
                  sasrec_history_attribution=False, seconds=time.monotonic() - start,
                  numerical_note='Scalar verifier uses the original float32 row norm/epsilon; no tolerance relaxation or analysis-result changes.')
    write(OUT / 'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
