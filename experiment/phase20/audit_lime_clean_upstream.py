"""Independent clean-input lineage and saved probe audit, without fitting."""
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from experiment.phase20.guided_decoding_job import ROOT,read,write,sha,now


def main():
    root=ROOT/'artifacts/phase20/lime_clean_upstream_toys_20260920_v1'
    prepared=read(root/'prepare/summary.json');assert prepared['state']=='COMPLETED'
    data=ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1/Toys'
    sequences={r['user_id']:r['history'] for r in map(json.loads,(data/'upstream_sequences_without_terminal.jsonl').open())}
    held=list(map(json.loads,(data/'terminal_holdout.jsonl').open()))
    near=ROOT/'artifacts/phase20/nearest_comparison_20260918'
    items=list(map(json.loads,(near/'Toys.items.jsonl').open()));ids=[r['source_item_id'] for r in items]
    count=np.zeros(len(ids)+1,dtype=np.int64)
    for seq in sequences.values():
        for i in seq:count[i]+=1
    np.testing.assert_array_equal(count,np.load(root/'prepare/frequencies.npy'))
    # Full original train bank minus heldout target multiplicity must exactly
    # match new counts; retaining identical items at other events is legitimate.
    with np.load(near/'Toys.train_sequences.npz',allow_pickle=False) as old:
        original=np.bincount(old['items'],minlength=len(ids)+1)
        for j,u in enumerate(old['users']):
            seq=old['items'][old['offsets'][j]:old['offsets'][j+1]].tolist()
            expected=seq[:-1] if any_held(str(u),held) else seq
            assert sequences[str(u)]==expected
    removed=np.bincount([r['target'] for r in held],minlength=len(ids)+1)
    np.testing.assert_array_equal(original-count,removed)
    cf=read(root/'prepare/itemcf.json')
    assert cf['num_training_interactions']==int(count.sum())==125279
    anchors=np.linspace(1,len(ids),12,dtype=int).tolist();pairs={a:collections.Counter() for a in anchors}
    for seq in sequences.values():
        for right,y in enumerate(seq):
            for x in seq[max(0,right-20):right]:
                if x==y:continue
                if x in pairs:pairs[x][y]+=1
                if y in pairs:pairs[y][x]+=1
    neighbors=0
    for anchor,tallies in pairs.items():
        expected=[(ids[k-1],v/math.sqrt(count[anchor]*count[k])) for k,v in tallies.items()]
        expected.sort(key=lambda x:(-x[1],x[0]));expected=expected[:100]
        actual=cf['neighbors'].get(ids[anchor-1],[])
        assert len(expected)==len(actual)
        for (item,value),got in zip(expected,actual):
            assert item==got['item_id'] and abs(value-got['score'])<1e-12;neighbors+=1
    reports={};hashes=0
    for action in ('prepare','cpu_smoke','profile'):
        path=root/action/'summary.json'
        if not path.exists():continue
        summary=read(path);assert summary['state']=='COMPLETED'
        for rel,digest in summary['input_sha256'].items():assert sha(ROOT/rel)==digest;hashes+=1
        for name,digest in summary['outputs_sha256'].items():assert sha(root/action/name)==digest;hashes+=1
        if action=='prepare':continue
        sas=torch.load(root/action/'sasrec_probe.pt',map_location='cpu')
        gram=torch.load(root/action/'gram_probe.pt',map_location='cpu')
        c1=torch.load(root/action/'c1_probe.pt',map_location='cpu')
        assert sas['probe_only'] and not sas['train_complete'] and sas['steps']==40
        assert gram['probe_only'] and not gram['formal_parent']
        assert c1['probe_only'] and not c1['formal_parent']
        for cp in (sas,gram,c1):assert all(torch.isfinite(v).all() for v in cp['state_dict'].values())
        assert summary['checkpoint_replay_exact']
        reports[action]=dict(device=summary['device'],sasrec_steps=40,gram_steps=len(summary['gram']),
            c1_steps=len(summary['c1']),all_saved_tensors_finite=True,checkpoint_replay_exact=True,
            reusable_as_formal_parents=False)
        del sas,gram,c1
    report=dict(state='VERIFIED',completed_at=now(),sequence_users=len(sequences),removed_events=len(held),
        frequency_entries=len(count),checked_itemcf_anchors=len(anchors),checked_itemcf_neighbors=neighbors,
        checked_hashes=hashes,profiles=reports,test_read=False,development_labels_read=False,
        original_training_bank_used_for_readonly_audit=True,source_sha256=sha(__file__))
    write(root/'independent_verification.json',report);print(report)


def any_held(user,held):
    # Cache user membership once; no target selection enters reconstruction.
    if not hasattr(any_held,'users'):any_held.users={r['user_id'] for r in held}
    return user in any_held.users


if __name__=='__main__':main()
