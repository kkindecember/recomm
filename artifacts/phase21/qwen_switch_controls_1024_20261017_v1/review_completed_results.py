"""CPU-only independent review of the completed matched Qwen switch experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REC = OUT/'recovery_20261017_v1'
ARMS = ('pair_qwen_raw','pair_qwen_loo','pair_qwen_raw_penalty','pair_qwen_loo_penalty',
        'qwen_raw','qwen_loo','qwen_raw_penalty','qwen_loo_penalty')
FRESH = tuple(k for k in ARMS if k not in ('pair_qwen_raw','qwen_raw'))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def metrics(rr):
    rr = np.asarray(rr)
    return {f'{kind}{k}':float(((rr<=k) if kind=='H' else (rr<=k)/np.log2(rr+1)).mean())
            for k in (5,10,20,50) for kind in ('H','N')}


def contrast(a,b):
    dn = (a<=10)/np.log2(a+1)-(b<=10)/np.log2(b+1)
    dh = (a<=10).astype(float)-(b<=10)
    rng = np.random.RandomState(20261017)
    values = np.empty((2000,2))
    for j in range(len(values)):
        ix = rng.randint(0,len(a),len(a))
        values[j] = dn[ix].mean(),dh[ix].mean()
    return dict(delta_N10=float(dn.mean()),delta_H10=float(dh.mean()),
        descriptive_95pct_N10=np.percentile(values[:,0],[2.5,97.5]).tolist(),
        descriptive_95pct_H10=np.percentile(values[:,1],[2.5,97.5]).tolist(),
        top10_gained=int(((a<=10)&(b>10)).sum()),top10_lost=int(((a>10)&(b<=10)).sum()))


def score_vectors(raw, loo, history):
    values = dict(pair_qwen_raw=np.float32(.5)*raw[:,0]+np.float32(.5)*raw[:,1],
                  pair_qwen_loo=np.float32(.5)*raw[:,0]+np.float32(.5)*loo,
                  qwen_raw=raw[:,1].copy(),qwen_loo=loo.copy())
    ix = np.unique(history)-1
    for key in tuple(values):
        value = values[key].copy(); value[ix]-=np.float32(.05)
        values[key+'_penalty'] = value
    return values


def review(domain):
    folder = OUT/domain
    samples = read(folder/'samples.json')
    assert len(samples)==len({s['user_id'] for s in samples})==1024
    hashes = read(folder/'output_manifest.json')['sha256']
    for rel,digest in hashes.items():
        assert sha(ROOT/rel)==digest,rel
    baselines = [json.loads(line) for line in (folder/'comparison_predictions.jsonl').read_text().splitlines()]
    rr = dict(np.load(folder/'ranks.npz'))
    np.testing.assert_array_equal(rr.pop('users'),[s['user_id'] for s in samples])
    assert len(rr)==19
    pure = dict(np.load(folder/'standalone_ranks.npz')); assert set(pure)==set(ARMS)
    raw = np.load(folder/'raw_experts.npy',mmap_mode='r')
    loo = np.load(folder/'loo_text.npy',mmap_mode='r')
    job_rows = {}
    for shard in range(4):
        job = OUT/'jobs'/f'{domain}_{shard}'
        assert read(job/'status.json')['state']=='COMPLETED'
        assert read(job/'lease_status.json')['state']=='released'
        checks = read(job/'runtime_checks.json')
        assert checks['state']=='PASSED' and checks['users']==256 and checks['gram_digest_unchanged']
        assert read(job/'smoke.json')['state']=='PASSED'
        records = [json.loads(line) for line in (job/'predictions.jsonl').read_text().splitlines()]
        assert [r['sample_index'] for r in records]==list(range(shard,1024,4))
        for row in records:
            assert row['sample_index'] not in job_rows
            job_rows[row['sample_index']] = row
    hp = {a:[] for a in rr}
    records = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    assert len(records)==1024 and sorted(job_rows)==list(range(1024))
    for j,row in enumerate(records):
        sample,baseline,fresh = samples[j],baselines[j],job_rows[j]
        assert row['sample_index']==j
        assert all(row[k]==sample[k]==baseline[k]==fresh[k] for k in ('user_id','target','history','evaluation_block'))
        for key,value in baseline.items():
            assert row[key]==value,(j,key)
        for arm in FRESH:
            assert row[arm]==fresh[arm]
        for arm,ranks in rr.items():
            ids = np.asarray(row[arm]['ids']); score = np.asarray(row[arm]['scores'])
            assert len(ids)==len(set(ids))==len(score)==50
            assert ids.min()>=1 and ids.max()<=raw.shape[1] and np.isfinite(score).all()
            at = np.flatnonzero(ids==sample['target'])
            rank = (1+int(((score>score[at[0]])|((score==score[at[0]])&(ids<sample['target']))).sum())) if len(at) else 2**30
            assert rank==ranks[j],(domain,j,arm)
            ordered = ids[np.lexsort((ids,-score))][:10]
            hp[arm].append(float(np.isin(ordered,sample['full_history']).mean()))
        history_ix = np.unique(sample['full_history'])-1
        assert np.array_equal(np.delete(raw[j,:,1],history_ix),np.delete(loo[j],history_ix))
        for arm,value in score_vectors(raw[j],loo[j],sample['full_history']).items():
            t = sample['target']-1
            rank = 1+np.count_nonzero(value>value[t])+np.count_nonzero(value[:t]==value[t])
            assert rank==pure[arm][j]==fresh['standalone_ranks'][arm]==row['standalone_ranks'][arm]
    blocks = np.array([s['evaluation_block'] for s in samples])
    groups = dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024),
                  **{f'block{k+1}':np.flatnonzero(blocks==k) for k in range(5)})
    summary = read(folder/'summary.json')
    root_summary = read(OUT/'summary.json')['domains'][domain]
    assert root_summary==summary
    for group,ix in groups.items():
        saved = summary[group]; assert saved['users']==len(ix)
        for key,values in rr.items():
            for metric,value in metrics(values[ix]).items():
                assert abs(value-saved['metrics'][key][metric])<1e-12
            assert abs(float(np.mean(np.asarray(hp[key])[ix]))-saved['mean_history_fraction_top10'][key])<1e-12
        for key,values in pure.items():
            for metric,value in metrics(values[ix]).items():
                assert abs(value-saved['standalone_metrics'][key][metric])<1e-12
    primary = ('sas_penalty','qwen_loo_penalty','pair_bge_loo_penalty')
    checked = 0
    for group,ix in groups.items():
        for baseline in primary:
            key = 'pair_qwen_loo_penalty_vs_'+baseline
            measured = contrast(rr['pair_qwen_loo_penalty'][ix],rr[baseline][ix])
            assert measured==summary[group]['comparisons'][key],(domain,group,key)
            checked+=1
    switch_pairs = [('pair_qwen_loo','pair_qwen_raw'),('pair_qwen_loo_penalty','pair_qwen_raw_penalty'),
                    ('pair_qwen_raw_penalty','pair_qwen_raw'),('pair_qwen_loo_penalty','pair_qwen_loo')]
    for a,b in switch_pairs:
        assert contrast(rr[a],rr[b])==summary['all1024']['comparisons'][a+'_vs_'+b]
        checked+=1
    overall = summary['all1024']['metrics']
    result = dict(state='PASSED',domain=domain,users=1024,methods=19,
        all_prediction_ranks_verified=1024*19,all_pure_qwen_ranks_verified=1024*8,
        all_8_group_metrics_recomputed=True,output_hashes_verified=True,
        shard_metadata_and_cached_comparisons_exact=True,all_eight_smoke_and_runtime_passed=True,
        all_leases_released=True,descriptive_intervals_recomputed=checked,
        primary_pair_vs_sas=summary['all1024']['comparisons']['pair_qwen_loo_penalty_vs_sas_penalty'],
        relative_N10_vs_sas=overall['pair_qwen_loo_penalty']['N10']/overall['sas_penalty']['N10']-1,
        first_second512_relative_N10_vs_sas={g:summary[g]['metrics']['pair_qwen_loo_penalty']['N10']/summary[g]['metrics']['sas_penalty']['N10']-1 for g in ('first512','second512')},
        blocks_above_sas=sum(summary[g]['metrics']['pair_qwen_loo_penalty']['N10']>summary[g]['metrics']['sas_penalty']['N10'] for g in groups if g.startswith('block')),
        cpu_only=True,training_or_generation_rerun=False,test_read=False)
    write(folder/'independent_review.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    assert read(OUT/'status.json')['state']=='COMPLETED'
    manifest = read(OUT/'manifest.json')
    for rel,digest in manifest['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    protocol = read(REC/'resume_protocol.json')
    for rel,digest in protocol['untouched_completed_files'].items():
        assert sha(ROOT/rel)==digest,rel
    old = (REC/'original_failure/Beauty_1/predictions.jsonl').read_bytes()
    resumed = (REC/'jobs/Beauty_1/predictions.jsonl').read_bytes()
    assert (OUT/'jobs/Beauty_1/predictions.jsonl').read_bytes()==old+resumed
    assert len(old.splitlines())==12 and len(resumed.splitlines())==244
    assert read(REC/'jobs/Beauty_1/prefix_replay.json')['state']=='PASSED'
    assert read(OUT/'jobs/Beauty_1/recovery_smoke.json')['state']=='PASSED'
    assert read(REC/'parallel_tail_20261017_v1/status.json')['state']=='CANCELLED_BY_USER'
    domains = {d:review(d) for d in ('Toys','Beauty')}
    summary = read(OUT/'summary.json'); choice = read(OUT/'text_expert_decision.json')
    candidates = {}
    for text in ('bge','qwen'):
        gains = {}; hr = []; above = {}
        for d in domains:
            mm = summary['domains'][d]['all1024']['metrics']
            pair,sas,content = mm[f'pair_{text}_loo_penalty'],mm['sas_penalty'],mm[f'{text}_loo_penalty']
            gains[d] = pair['N10']/sas['N10']-1
            hr.append(pair['H10']/sas['H10']-1)
            above[d] = pair['N10']>max(sas['N10'],content['N10'])
        candidates[text] = dict(relative_N10_vs_common_sas=gains,worst_domain_relative_N10=min(gains.values()),
            mean_relative_N10=float(np.mean(list(gains.values()))),mean_relative_H10=float(np.mean(hr)),
            pair_above_both_matching_singles_by_mean=above,both_domains_above_both_singles_by_mean=all(above.values()))
    assert candidates==choice['candidates']
    selected = max(candidates,key=lambda t:(round(candidates[t]['worst_domain_relative_N10'],12),
        round(candidates[t]['mean_relative_N10'],12),round(candidates[t]['mean_relative_H10'],12),t=='bge'))
    assert selected==choice['selected_development_text_expert']=='qwen'
    assert choice==summary['selection']
    result = dict(state='PASSED',input_hashes_verified=len(manifest['sha256']),
        recovery_original_12_and_new_244_byte_exact=True,seven_old_shards_hashes_unchanged=True,
        selected_text_expert=selected,selection_rule_independently_verified=True,
        domains=domains,cpu_only=True,test_read=False,
        limitation='Saved numerical verification on reused development users; no independent training-seed or held-out confirmation.')
    write(OUT/'independent_review.json',result)
    print(json.dumps(dict(state='PASSED',selected=selected,input_hashes=len(manifest['sha256'])),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
