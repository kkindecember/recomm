"""Prepare train-only terminal holdout inputs; never launches upstream training.

This is a within-user event split, not a global-time split. Holding out an event
does not exclude the same item from other users' legitimate training histories.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now
from experiment.phase20.personal_popularity_core import reconstruct_training

DATA = {'Toys':'artifacts/phase18/diff_gram/prepared_v2_categories',
        'Beauty':'artifacts/phase18/diff_gram/beauty/prepared_v1'}


def main():
    parent = ROOT/'artifacts/phase20/lime_learnability_20260920_v1'
    assert read(parent/'summary.json')['continue_to_data_design']
    assert read(parent/'independent_verification.json')['state'] == 'VERIFIED'
    out = ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1'
    out.mkdir(parents=True,exist_ok=False)
    hashes = {}; domains = {}
    for domain, source in DATA.items():
        folder = out/domain; folder.mkdir()
        path = ROOT/source/'train.jsonl'
        hashes[str(path.relative_to(ROOT))] = sha(path)
        raw = path.read_text().splitlines(keepends=True)
        rows = [json.loads(x) for x in raw]
        _, sequences, _, _ = reconstruct_training(rows)
        cache_path = ROOT/'artifacts/phase20/source_competition_20260920_v2'/domain/'train.npz'
        hashes[str(cache_path.relative_to(ROOT))] = sha(cache_path)
        old = read(ROOT/'artifacts/phase20/source_competition_20260920_v2/summary.json')
        assert hashes[str(cache_path.relative_to(ROOT))] == old['output_sha256'][str(cache_path.relative_to(ROOT))]
        with np.load(cache_path,allow_pickle=False) as z:
            users, indices, targets = z['users'],z['train_indices'],z['targets']
        cohort = set(map(str,users))
        assert len(cohort) == {'Toys':3494,'Beauty':4085}[domain]
        terminal = {r['user_id']:i for i,r in enumerate(rows)}
        cache_latest = {}
        for j in np.argsort(indices): cache_latest[str(users[j])] = int(j)
        held_indices = {terminal[u] for u in cohort}
        cleaned = [r for i,r in enumerate(rows) if i not in held_indices]
        _, rebuilt, _, _ = reconstruct_training(cleaned)
        clean_sequences = {u:(s[:-1] if u in cohort else s) for u,s in sequences.items()}
        for u,s in clean_sequences.items():
            assert rebuilt.get(u,s[:1]) == s
        held = []
        for u in sorted(cohort):
            ix = terminal[u]; row = rows[ix]; j = cache_latest[u]
            assert int(indices[j]) == ix and int(targets[j]) == row['target']
            assert row['history'] == clean_sequences[u][-20:]
            role_hash = hashlib.sha256(f'20260920|lime-clean-fit-role|{domain}|{u}'.encode()).digest()
            role = 'gate_calibration' if int.from_bytes(role_hash[:8],'big') % 5 == 0 else 'ranker_fit'
            held.append(dict(row, original_train_index=ix, role=role))
        with (folder/'upstream_train_without_terminal.jsonl').open('x') as f:
            for i,line in enumerate(raw):
                if i not in held_indices: f.write(line)
        with (folder/'terminal_holdout.jsonl').open('x') as f:
            for row in held: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        with (folder/'upstream_sequences_without_terminal.jsonl').open('x') as f:
            for u in sorted(clean_sequences):
                f.write(json.dumps(dict(user_id=u,history=clean_sequences[u]),ensure_ascii=False)+'\n')
        # Independent read-back of exact retained lines and role separation.
        actual = (folder/'upstream_train_without_terminal.jsonl').read_text().splitlines(keepends=True)
        assert actual == [line for i,line in enumerate(raw) if i not in held_indices]
        saved = [json.loads(line) for line in (folder/'terminal_holdout.jsonl').open()]
        gate = {r['user_id'] for r in saved if r['role']=='gate_calibration'}
        ranker = {r['user_id'] for r in saved if r['role']=='ranker_fit'}
        assert not gate & ranker and gate | ranker == cohort
        domains[domain] = dict(original_train_prefixes=len(rows),clean_train_prefixes=len(cleaned),
            heldout_events=len(held),gate_calibration_users=len(gate),ranker_fit_users=len(ranker),
            sequence_users=len(clean_sequences),heldout_followed_by_training_prefix=False,
            exact_retained_line_readback=True,old_candidate_cache_reusable=False,
            old_trained_upstream_reusable=False,global_time_split=False)
    for name,digest in hashes.items(): assert sha(ROOT/name) == digest
    write(out/'summary.json',dict(state='PREPARED',completed_at=now(),domains=domains,
        input_sha256=hashes,output_sha256={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*.jsonl')},
        source_sha256=sha(__file__),upstream_training_started=False,new_beam=False,
        development_labels_used_for_split=False,test_read=False,
        limitation='Only event lists are prepared. All exposed upstreams, collaborative hints, counts and score caches still require reconstruction; existing weights cannot make these heldout targets unseen.'))
    print(read(out/'summary.json'))


if __name__ == '__main__': main()
