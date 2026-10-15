"""Frozen CPU attribution audit; no fitting, no altered recommendation policy."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from analyze_history_evidence import fields
from analyze_selective_evidence import sha, write, csv_write
from evidence_decomposition_core import read_users
from expert_evidence_alignment_core import Attribution, EXPERTS, normalized, vote, summarize_contributions, support_card, top_evidence
from prepare_blind_disagreements import select_a

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/expert_evidence_alignment_20260926_v1'
PRIOR = ROOT / 'artifacts/phase21/selective_evidence_20260925_v1'
BLIND = ROOT / 'artifacts/phase21/blind_disagreements_20260925_v1'
RAW = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL = ROOT / 'artifacts/phase21/full_validation_20260924_v1'
PROTOCOL = ROOT / 'plan/第二十一阶段/21.18_专家实际依据与语义解释对齐检查_20260926.md'
sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
from lime_rec.experts import ItemCFExpert, LLMSemanticExpert


def status(stage, **kw):
    write(OUT / 'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(),
                                  utc=datetime.now(timezone.utc).isoformat(), **kw))
    print(stage, kw, flush=True)


def counts(rows):
    result = dict(users=len(rows), a_hit=sum(r['stratum'] == 'a_hit' for r in rows),
                  b_hit=sum(r['stratum'] == 'b_hit' for r in rows),
                  neither=sum(r['stratum'] == 'neither' for r in rows), experts={})
    for expert in EXPERTS:
        v = [r[expert + '_vote'] for r in rows]
        info = [r for r in rows if r['stratum'] != 'neither']
        one = dict(A=v.count('A'), B=v.count('B'), tie=v.count('tie'),
                   target_agree=sum(r[expert + '_vote'] == r['stratum'][0].upper() for r in info),
                   target_disagree=sum(r[expert + '_vote'] not in ('tie', r['stratum'][0].upper()) for r in info),
                   target_tie=sum(r[expert + '_vote'] == 'tie' for r in info))
        if expert != 'sas':
            available = [r[expert + '_cancellation'] for r in rows if r[expert + '_cancellation'] is not None]
            one.update(conflicting_events=sum(r[expert + '_conflicting_events'] for r in rows),
                       no_contribution=sum(r[expert + '_no_contribution'] for r in rows),
                       single_event_flip=sum(r[expert + '_single_event_flip'] for r in rows),
                       cancellation_n=len(available),
                       cancellation_quantiles=np.quantile(available, [0, .25, .5, .75, 1]).tolist() if available else None)
        result['experts'][expert] = one
    return result


def manual_alignment(cards):
    definite = [c for c in cards if c['review']['preference'] != 'abstain']
    groups = {}
    for name, subset in [('all24', definite), ('target_in_pair', [c for c in definite if c['review']['stratum'] != 'neither']),
                         ('manual_agrees', [c for c in definite if c['review']['agrees_with_observed_target'] is True]),
                         ('manual_disagrees', [c for c in definite if c['review']['agrees_with_observed_target'] is False]),
                         ('neither', [c for c in definite if c['review']['stratum'] == 'neither'])]:
        group = dict(cards=len(subset), experts={})
        for e in EXPERTS:
            agrees = [c for c in subset if c['row'][e + '_vote'] == c['review']['chosen_role']]
            one = dict(agrees_manual=len(agrees),
                       opposes_manual=sum(c['row'][e + '_vote'] not in ('tie', c['review']['chosen_role']) for c in subset),
                       tie=sum(c['row'][e + '_vote'] == 'tie' for c in subset))
            if e != 'sas':
                one['top_support_in_cited_history'] = sum(
                    bool(c[e]['manual_support']) and c[e]['manual_support'][0]['ref'] in
                    c['review'][c['review']['preference'] + '_history'] for c in agrees)
            group['experts'][e] = one
        groups[name] = group
    return groups


def run():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    manifest = {}
    prior_hashes = json.loads((BLIND / 'completion_manifest.json').read_text())

    def use(path):
        relative = str(path.relative_to(ROOT))
        digest = sha(path)
        if relative in prior_hashes:
            assert digest == prior_hashes[relative], relative
        manifest[relative] = digest
        return path

    for p in [Path(__file__), Path(__file__).with_name('expert_evidence_alignment_core.py'),
              Path(__file__).with_name('prepare_blind_disagreements.py'), PROTOCOL,
              ROOT / 'third_party/LIME-Rec/lime_rec/experts.py']:
        use(p)
        shutil.copy2(p, OUT / p.name)
    status('verify_inputs')
    reviews = {r['card_id']: r for r in json.loads(use(BLIND / 'unblinded_reviews.json').read_text())}
    review_by_user = {(r['domain'], r['user_id']): r for r in reviews.values()}
    sealed = json.loads(use(BLIND / 'sealed_labels.json').read_text())
    all_cards, domain_results, all_rows = [], {}, []
    parity = {}
    for domain in ('Toys', 'Beauty'):
        folder = PRIOR / domain / 'validation'
        users = read_users(use(folder / 'users.csv'))
        with np.load(use(folder / 'pairs.npz')) as z:
            items = z['items']
        score = np.load(use(folder / 'score.npy'), mmap_mode='r')[:len(items)]
        samples_list = json.loads(use(FULL / domain / 'samples.json').read_text())
        samples = {r['user_id']: r for r in samples_list}
        source_indices = {r['user_id']: i for i, r in enumerate(samples_list)}
        cfg = json.loads(use(RAW / domain / 'data_config.json').read_text())
        catalog = json.loads(use(RAW / domain / 'catalog.json').read_text())
        titles = [fields(p).get('title', '') for p in catalog['passages']]
        with np.load(use(ROOT / cfg['embedding'])) as z:
            assert z['item_ids'].tolist() == catalog['items'][1:]
            embeddings = z['embeddings']
        cfpath = use(ROOT / cfg['itemcf'])
        attr = Attribution(catalog['items'][1:], embeddings, json.loads(cfpath.read_text()))
        official_cf = ItemCFExpert(str(cfpath), catalog['items'][1:])
        official_text = LLMSemanticExpert(catalog['items'][1:], {}, embeddings=embeddings)
        fullpath = use(FULL / domain / 'experts.npy')
        cache_manifest = json.loads(use(FULL / domain / 'score_cache_manifest.json').read_text())
        assert manifest[str(fullpath.relative_to(ROOT))] == cache_manifest['sha256']['experts.npy']
        experts = np.load(fullpath, mmap_mode='r')
        write(OUT / 'input_manifest.json', manifest)
        excluded = set(sealed['excluded_previously_shown'][domain])
        rows = []
        domain_cards = []
        errors = dict(cf_catalog=0., text_catalog=0., cf_pair=0., text_pair=0., cached_pair=0.)
        expected_count = sealed['population'][domain]['primary_after_exclusions']
        for u in users:
            if u['cohort'] != 'primary' or u['user_id'] in excluded:
                continue
            sl = slice(u['start'], u['end'])
            a = select_a(items[sl], score[sl, 18])
            if not a:
                continue
            b = u['boundary']
            pair_index = u['start'] + int(np.flatnonzero(items[sl] == a)[0])
            x = score[pair_index]
            h = samples[u['user_id']]['full_history']
            pair = attr.pair(a, b, h)
            review = review_by_user.get((domain, u['user_id']))
            row = dict(domain=domain, user_id=u['user_id'], a=a, b=b,
                       target=u['target'], stratum='a_hit' if a == u['target'] else 'b_hit' if b == u['target'] else 'neither',
                       reviewed=review is not None, history_events=len(h), a_title=titles[a], b_title=titles[b])
            for j, e in enumerate(EXPERTS):
                row[e + '_a'] = float(x[j]); row[e + '_b'] = float(x[j + 3])
                row[e + '_margin'] = float(x[j] - x[j + 3]); row[e + '_vote'] = vote(row[e + '_margin'])
                if e != 'sas':
                    stats = summarize_contributions(pair[e][:, 0] - pair[e][:, 1])
                    row.update({e + '_' + k: v for k, v in stats.items()})
                    if row[e + '_vote'] != 'tie':
                        assert stats['raw_margin'] * row[e + '_margin'] > 0, (domain, u['user_id'], e)
            rows.append(row)
            if review:
                assert (review['a'], review['b']) == (a, b)
                source_i = source_indices[u['user_id']]
                # numpy advanced indexing changes axis order; explicit stack fixes it.
                cached_pair = np.stack([experts[source_i, :, a - 1], experts[source_i, :, b - 1]])
                assert np.array_equal(cached_pair.ravel(), np.r_[x[:3], x[3:6]])
                raw_cf = official_cf.score(u['user_id'], [catalog['items'][i] for i in h])
                raw_text = official_text.score(u['user_id'], [catalog['items'][i] for i in h])
                for e, raw, expert_i in [('cf', raw_cf, 1), ('text', raw_text, 2)]:
                    nr = normalized(raw)
                    delta = float(np.max(np.abs(nr - experts[source_i, expert_i])))
                    errors[e + '_catalog'] = max(errors[e + '_catalog'], delta)
                    np.testing.assert_allclose(nr, experts[source_i, expert_i], atol=5e-6, rtol=1e-5)
                    error = float(abs((raw[a - 1] - raw[b - 1]) - row[e + '_raw_margin']))
                    errors[e + '_pair'] = max(errors[e + '_pair'], error)
                    assert error < 5e-6
                card = dict(card_id=review['card_id'], row=row, review=review, history=h,
                            a=dict(item=a, title=titles[a]), b=dict(item=b, title=titles[b]))
                for e in ('cf', 'text'):
                    support = support_card(h, pair[e], titles)
                    role = review['chosen_role']
                    card[e] = dict(events=pair[e].tolist(), sources=support,
                                   supports_a=top_evidence(support, 'A'), supports_b=top_evidence(support, 'B'),
                                   manual_support=top_evidence(support, role) if role in ('A', 'B') else [])
                domain_cards.append(card)
            if len(rows) % 2048 == 0:
                status('attribution', domain=domain, completed=len(rows), total=expected_count)
        assert len(rows) == expected_count and len(domain_cards) == 64
        all_rows.extend(rows); all_cards.extend(domain_cards)
        csv_write(OUT / (domain + '_users.csv'), rows)
        domain_results[domain] = dict(
            all=counts(rows), reviewed=counts([r for r in rows if r['reviewed']]),
            not_reviewed=counts([r for r in rows if not r['reviewed']]),
            strata={s: counts([r for r in rows if r['stratum'] == s]) for s in ('a_hit', 'b_hit', 'neither')},
            manual_alignment=manual_alignment(domain_cards))
        parity[domain] = dict(cards=64, catalog_item_scores_checked=64 * 2 * (len(catalog['items']) - 1),
                              maximum_absolute_errors=errors)
        status('domain_complete', domain=domain, seconds=time.monotonic() - started)
    assert json.loads((ROOT / 'artifacts/phase21/expert_feedback_20260924_v1/status.json').read_text())['state'] == 'USER_PAUSED'
    result = dict(state='COMPLETED', domains=domain_results, manual_alignment=manual_alignment(all_cards),
                  parity=parity, users=len(all_rows), reviewed_cards=len(all_cards), seconds=time.monotonic() - started,
                  statistical_fits=0, gpu_inferences=0, test_read=False, semantic_training_state='USER_PAUSED',
                  new_recommendation_policy=False, causal_user_intent_claim=False)
    write(OUT / 'cards.json', all_cards)
    write(OUT / 'summary.json', result)
    write(OUT / 'input_manifest.json', manifest)
    write(OUT / 'status.json', {k: v for k, v in result.items() if k not in ('domains', 'manual_alignment')})
    print(json.dumps(dict(users=result['users'], seconds=result['seconds'], parity=parity,
                          manual_alignment=result['manual_alignment']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        if OUT.exists():
            write(OUT / 'status.json', dict(state='FAILED', traceback=traceback.format_exc(), pid=os.getpid()))
        raise
