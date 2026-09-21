"""Verify Beauty's event exclusion and rebuild its independent ItemCF assets."""
import collections
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiment.phase20.guided_decoding_job import ROOT,read,write,sha,now
from experiment.phase20.lime_clean_upstream_core import validate_rows
sys.path.insert(0,str(ROOT/'third_party/LIME-Rec'))
from lime_rec.itemcf import build_itemcf_model
from scripts.training.train_sasrec import SeqDataset


def main():
    torch.set_num_threads(4);limit=threadpool_limits(4);started=time.monotonic()
    out=ROOT/'artifacts/phase20/lime_clean_beauty_preflight_20260920_v1';out.mkdir(exist_ok=False)
    inputs={}
    def use(path):
        inputs[str(path.relative_to(ROOT))]=sha(path);return path
    parent=read(use(ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1/summary.json'))
    data=ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1/Beauty'
    for name in ('upstream_sequences_without_terminal.jsonl','upstream_train_without_terminal.jsonl','terminal_holdout.jsonl'):
        p=use(data/name);assert inputs[str(p.relative_to(ROOT))]==parent['output_sha256'][str(p.relative_to(ROOT))]
    sequences={r['user_id']:r['history'] for r in map(json.loads,(data/'upstream_sequences_without_terminal.jsonl').open())}
    rows=list(map(json.loads,(data/'upstream_train_without_terminal.jsonl').open()))
    assert validate_rows(rows,sequences)==127328;del rows
    held=list(map(json.loads,(data/'terminal_holdout.jsonl').open()))
    cohort={r['user_id'] for r in held};assert len(cohort)==len(held)==4085
    assert all(r['history']==sequences[r['user_id']][-20:] for r in held)
    gate={r['user_id'] for r in held if r['role']=='gate_calibration'}
    fit={r['user_id'] for r in held if r['role']=='ranker_fit'}
    assert not gate & fit and gate | fit == cohort and len(gate)==775 and len(fit)==3310
    near=ROOT/'artifacts/phase20/nearest_comparison_20260918'
    items=list(map(json.loads,use(near/'Beauty.items.jsonl').open()));ids=[r['source_item_id'] for r in items]
    assert len(ids)==len(set(ids))==12101
    assert [r['item_id'] for r in items]==list(range(1,len(ids)+1))
    count=np.bincount(np.concatenate(list(sequences.values())),minlength=len(ids)+1)
    assert int(count.sum())==149691 and len(sequences)==22363
    with np.load(use(near/'Beauty.train_sequences.npz'),allow_pickle=False) as original:
        for j,u in enumerate(original['users']):
            seq=original['items'][original['offsets'][j]:original['offsets'][j+1]].tolist()
            assert sequences[str(u)]==(seq[:-1] if str(u) in cohort else seq)
        old_count=np.bincount(original['items'],minlength=len(ids)+1)
    np.testing.assert_array_equal(old_count-count,np.bincount([r['target'] for r in held],minlength=len(ids)+1))
    ds=SeqDataset([sequences[u] for u in sorted(sequences)],50,len(ids))
    assert len(ds)==22363
    # Verify the actual SASRec adapter on every sequence, including truncation.
    token_count=0
    for i,seq in enumerate(ds.data):
        x,y,negative=ds[i];truncated=seq[-51:]
        assert x[x!=0].tolist()==truncated[:-1] and y[y!=0].tolist()==truncated[1:]
        assert not set(negative[negative!=0].tolist()) & set(seq)
        token_count+=int((y!=0).sum())
    np.save(out/'frequencies.npy',count)
    tic=time.monotonic()
    cf=build_itemcf_model(dataset_name='Beauty',histories={u:[ids[j-1] for j in s] for u,s in sequences.items()},
        item_ids=ids,top_k=100,window_size=20)
    assert cf['training_only'] and cf['num_training_interactions']==149691
    write(out/'itemcf.json',cf);cf_seconds=time.monotonic()-tic
    # Independently check sparse co-occurrences for 12 predetermined anchors.
    anchors=np.linspace(1,len(ids),12,dtype=int);pairs={int(a):collections.Counter() for a in anchors}
    for seq in sequences.values():
        for right,y in enumerate(seq):
            for x in seq[max(0,right-20):right]:
                if x==y:continue
                if x in pairs:pairs[x][y]+=1
                if y in pairs:pairs[y][x]+=1
    checked=0
    for anchor,counts in pairs.items():
        expected=sorted([(ids[j-1],n/math.sqrt(count[anchor]*count[j])) for j,n in counts.items()],key=lambda p:(-p[1],p[0]))[:100]
        got=cf['neighbors'].get(ids[anchor-1],[]);assert len(got)==len(expected)
        for (item,value),row in zip(expected,got):
            assert row['item_id']==item and abs(row['score']-value)<1e-12;checked+=1
    for name,digest in inputs.items():assert sha(ROOT/name)==digest
    result=dict(state='VERIFIED',domain='Beauty',completed_at=now(),seconds=time.monotonic()-started,
        dataset_boundary_checked=True,training_users=22363,training_prefixes=127328,heldout_events=4085,
        catalog_items=12101,training_interactions=149691,sasrec_users_checked=len(ds),
        sasrec_supervised_positions_per_epoch=token_count,itemcf_build_seconds=cf_seconds,
        independent_itemcf_neighbors=checked,input_sha256=inputs,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        source_sha256=sha(__file__),development_labels_read=False,test_read=False,
        original_training_bank_read_for_audit_only=True)
    write(out/'verification.json',result);print(result)


if __name__=='__main__':main()
