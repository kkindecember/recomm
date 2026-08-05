#!/usr/bin/env python3
"""P0-T: train one SASRec drafter and audit target-free candidate coverage."""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import torch
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiment.phase4.gcdh_p0 import ROOT, collate, prepare, read_users, sha256, write_json, normalized_sequence
from experiment.phase4.gacr_p0 import split_training_users
from experiment.phase4.gacr_s0 import select_stratified_samples, target_free_union
from experiment.phase4.rpcd_t0 import SASRec, rank_users, train_dataset
from utils import generation_trie as gt

def hit(items, target, k): return float(target in items[:k])
def ndcg(items, target):
    return 0.0 if target not in items[:10] else 1/math.log2(items.index(target)+2)
def dedup(items): return list(dict.fromkeys(items))

@torch.no_grad()
def gram_union(sample, prepared, cfg, device):
    model=prepared['model']; probe=dict(sample); probe['output']=prepared['item2lexid'][prepared['catalog'][0]]
    batch=collate(prepared['collator'],[probe]); ids=batch['item_text_ids'].to(device); mask=batch['item_text_masks'].to(device)
    if '_f0_trie' not in prepared:
        prepared['_f0_trie']=gt.Trie(prepared['encoded_candidates']); prepared['_f0_len']=max(map(len,prepared['encoded_candidates']))
    out=model.backbone.generate(input_ids=ids,attention_mask=mask,max_length=prepared['_f0_len'],prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(prepared['_f0_trie']),num_beams=50,num_return_sequences=50)
    beam=[prepared['sequence_to_item'].get(normalized_sequence(x.tolist())) for x in out]
    if any(x is None for x in beam) or len(set(beam))!=50: raise ValueError('invalid beam mapping')
    model.backbone.encoder.n_passages=ids.size(1); hidden=model.backbone.encoder(input_ids=ids.view(1,-1),attention_mask=mask.view(1,-1),return_dict=True)[0]
    pooled=model.pool_coarse(hidden,mask,ids.shape[-1])[0]; logits=model.catalog_head(pooled)
    for item in sample['history_items']:
        if item in model.item_to_index: logits[model.item_to_index[item]]=-torch.inf
    catalog=[prepared['catalog'][i] for i in torch.topk(logits,k=50).indices.tolist()]
    return beam,catalog,target_free_union(beam,catalog)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('P0-T requires CUDA')
    cfg=json.loads(a.config.read_text()); device=torch.device('cuda:0'); p0=json.loads((ROOT/cfg['p0_config']).read_text())
    trained={}; states={}; selected={}
    for d in cfg['datasets']:
        prepared=prepare(d,p0,device); rows,ss=train_dataset(d,prepared['sequences'],prepared['catalog'],cfg['drafter'],int(cfg['seed']),device); trained[d]=rows; states[d]=ss; del prepared; torch.cuda.empty_cache()
    epoch=max(range(len(next(iter(trained.values())))),key=lambda i:sum(trained[d][i]['internal_calibration_ndcg@10'] for d in cfg['datasets']))
    for d in cfg['datasets']: selected[d]=epoch+1
    all_rows=[]; metrics={}
    for d in cfg['datasets']:
        prepared=prepare(d,p0,device); ckpt=ROOT/cfg['checkpoint_root']/d/'C1/model.pt'; prepared['model'].load_state_dict(torch.load(ckpt,map_location=device),strict=True); prepared['model'].eval()
        users=read_users(ROOT/cfg['split_root']/d/'train_users.txt'); _,cal_users=split_training_users(users,int(cfg['seed']),d)
        from experiment.phase4.gcdh_p0 import build_train_samples
        pool=build_train_samples(prepared['sequences'],cal_users,prepared['item2input'],prepared['item2lexid']); samples=select_stratified_samples(pool,prepared['heads'],int(cfg['seed']),f'{d}|f0t',128,128)
        m=SASRec(len(prepared['catalog']),64,50,2,2,.2).to(device); m.load_state_dict(states[d][epoch]); idx={x:i+1 for i,x in enumerate(prepared['catalog'])}
        ranked=rank_users(m,[(s['sample_key'],s['history_items'],s['positive_item']) for s in samples],idx,['<padding>']+prepared['catalog'],50,256,50,device)
        rows=[]
        for s in samples:
            beam,cat,base=gram_union(s,prepared,cfg,device); sas=dedup(ranked[s['sample_key']]['items']); ext=dedup(base+sas); t=s['positive_item']; group='head' if t in prepared['heads'] else 'tail'
            row={'dataset':d,'sample_key':s['sample_key'],'group':group,'base_hit50':hit(base,t,50),'sasrec_hit50':hit(sas,t,50),'extended_hit50':hit(ext,t,50),'sasrec_unique_hit50':float(t in sas and t not in base),'base_oracle_ndcg10':ndcg([t] if t in base else [],t),'extended_oracle_ndcg10':ndcg([t] if t in ext else [],t),'base_size':len(base),'extended_size':len(ext)}; rows.append(row); all_rows.append(row)
        def avg(rs,k): return sum(r[k] for r in rs)/len(rs)
        metrics[d]={'n':len(rows),'base_recall50':avg(rows,'base_hit50'),'extended_recall50':avg(rows,'extended_hit50'),'delta_recall50':avg(rows,'extended_hit50')-avg(rows,'base_hit50'),'sasrec_unique_recall50':avg(rows,'sasrec_unique_hit50'),'tail_delta_recall50':avg([r for r in rows if r['group']=='tail'],'extended_hit50')-avg([r for r in rows if r['group']=='tail'],'base_hit50'),'unique_users':sum(r['sasrec_unique_hit50'] for r in rows)}
        del prepared,m; torch.cuda.empty_cache()
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'per_user.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    passed=all(metrics[d]['sasrec_unique_recall50']>0 and metrics[d]['unique_users']>=10 and metrics[d]['delta_recall50']>=.01 and metrics[d]['tail_delta_recall50']>=.005 for d in cfg['datasets']) and any(metrics[d]['delta_recall50']>=.02 for d in cfg['datasets'])
    summary={'experiment_id':cfg['experiment_id'],'decision':'F1_DESIGN_ALLOWED' if passed else 'STOP_CANDIDATE_DRAFTING','selected_shared_epoch':epoch+1,'metrics':metrics,'integrity':{'test_data_read':False,'sports_data_read':False,'verifier_trained':False},'per_user_sha256':sha256(a.output_root/'per_user.csv')}; write_json(a.output_root/'summary.json',summary)
if __name__=='__main__': main()
