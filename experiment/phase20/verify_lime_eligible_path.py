"""Independent saved-output and input-integrity audit, no scoring-policy import."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from experiment.phase20.verify_lime_mechanism import read, rows, digest, calculate, compare

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/lime_eligible_path_20260922_v1'
PRIORS=[ROOT/'artifacts/phase20'/p for p in ('lime_decode_screen_20260921_v1/full_parallel_v4',
        'lime_mechanism_screen_20260921_v1','lime_prefix_expert_20260921_v1')]
METHODS=('hard_filter','eligible_fixed','path_rho05','path_rho10')
DECODED={'GRAM','fixed','GRAM_filter',*METHODS}
NAMES=DECODED|{'fixed_late',*(n+'_late' for n in METHODS),
               'LIME_after_GRAM','LIME_after_GRAM_filter','LIME'}


def main():
    manifest=read(OUT/'manifest.json')['sha256']
    for path,expected in manifest.items():
        assert digest(ROOT/path)==expected,('Changed frozen input',path)
    checked={}
    for domain in ('Toys','Beauty'):
        folder=OUT/domain; samples=read(folder/'samples.json'); evaluation=rows(folder/'evaluation.jsonl')
        prior=[read(p/domain/'samples.json') for p in PRIORS]
        excluded={r['user_id'] for cohort in prior for r in cohort}
        assert len(samples)==515 and len({r['user_id'] for r in samples})==515
        assert samples[:3]==prior[0][:3]
        recovery=ROOT/f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        with np.load(recovery/'evaluation_seed0_cal2048.npz',allow_pickle=False) as f:
            old={k:f[k] for k in ('users','indices','targets','top50','arm_names')}
        expected=sorted([str(u) for u in old['users'] if str(u) not in excluded],
            key=lambda u:hashlib.sha256(f'20260922|lime-eligible-path-v1|{domain}|{u}'.encode()).digest())[:512]
        assert [r['user_id'] for r in samples[3:]]==expected
        with np.load(ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz',allow_pickle=False) as q:
            assert {r['query_index'] for r in samples}.isdisjoint(q['calibration_indices'].tolist())
        with np.load(folder/'signals.npz',allow_pickle=False) as f:
            signals={k:f[k] for k in f.files}
        old_lime=old['arm_names'].tolist().index('LIME')
        for j,sample in enumerate(samples):
            pos=sample['evaluation_cache_index']
            assert str(old['users'][pos])==sample['user_id'] and old['targets'][pos]==sample['target']
            assert old['indices'][pos]==sample['query_index']
            np.testing.assert_array_equal(signals['lime_top50'][j],old['top50'][old_lime,pos])
        cfg=read(folder/'data_config.json'); catalog=read(ROOT/cfg['prepared']/'catalog.json')
        paths_by_item={i:tuple(p) for i,p in enumerate(catalog['decoder_paths'],1)}
        selection=read(folder/'selection.json'); summary=read(folder/'summary.json')
        assert selection['alpha']==summary['alpha']==.3 and selection['rhos']==summary['rhos']==[0.,.5,1.]
        assert selection['evaluation_users_read']==0 and selection['new_tuning_users']==summary['new_tuning_users']==0
        assert digest(folder/'selection.json')==summary['selection_sha256']
        assert summary['model_unchanged'] and not summary['test_read'] and summary['optimizer_steps']==0
        assert summary['completed_decodes']==27+512*7
        smoke=read(folder/'smoke.json')
        assert smoke['users']==3 and smoke['zero_exact'] and smoke['all_filtered_paths_eligible']
        assert len(evaluation)==512 and [r['sample_index'] for r in evaluation]==list(range(3,515))
        predictions=0
        for row in evaluation:
            j=row['sample_index']; sample=samples[j]; history=set(row['history'])
            assert all(row[k]==sample[k] for k in ('user_id','target','history','split'))
            assert set(row['arms'])==NAMES and set(row['paths'])==DECODED
            assert row['filtered_candidates_all_eligible'] and row['shared_path_scores_match']
            for name,arm in row['arms'].items():
                if name=='LIME':
                    ids=np.arange(1,signals['lime'].shape[1]+1); scores=signals['lime'][j]
                else:
                    raw=row['candidate_scores'][name]; ids=np.asarray(raw['ids']); scores=np.asarray(raw['scores'])
                    assert len(ids)==len(set(ids.tolist()))==50 and len(scores)==50
                assert np.isfinite(scores).all()
                keep=np.array([int(i) not in history for i in ids]); ids,scores=ids[keep],scores[keep]
                order=ids[np.lexsort((ids,-scores))][:50].tolist()
                rank=order.index(row['target'])+1 if row['target'] in order else 51
                assert order==arm['items'] and rank==arm['rank'],(domain,j,name)
                if name in {'GRAM_filter',*METHODS,*(n+'_late' for n in METHODS),'LIME_after_GRAM_filter'}:
                    assert len(order)==50
                predictions+=1
            for name in DECODED:
                raw=row['candidate_scores'][name]
                assert [tuple(p) for p in row['paths'][name]]==[paths_by_item[i] for i in raw['ids']]
                assert row['decode_seconds'][name]>0 and row['diagnostics'][name]['decoder_steps']>0
                if name in {'GRAM_filter',*METHODS}:
                    assert set(raw['ids']).isdisjoint(history)
            for name in ('GRAM','fixed'):
                assert row['raw_history_counts'][name]==len(set(row['candidate_scores'][name]['ids'])&history)
            for name in ('fixed',)+METHODS:
                ref='GRAM' if name=='fixed' else 'GRAM_filter'
                late=row['candidate_scores'][name+'_late']; raw=row['candidate_scores'][name]
                assert late['ids']==row['candidate_scores'][ref]['ids']
                lookup=dict(zip(late['ids'],late['scores']))
                for i,s in zip(raw['ids'],raw['scores']):
                    if i in lookup:
                        np.testing.assert_allclose(s,lookup[i],atol=5e-6,rtol=1e-5)
        ranks={n:[r['arms'][n]['rank'] for r in evaluation] for n in NAMES}
        assert set(summary['metrics'])==NAMES
        pairs={n+'_vs_GRAM':(n,'GRAM') for n in NAMES if n!='GRAM'}
        for name in ('fixed',)+METHODS:
            pairs[name+'_vs_late']=(name,name+'_late')
        for a,b in [('GRAM_filter','GRAM'),('hard_filter','fixed'),('eligible_fixed','hard_filter'),
                    ('eligible_fixed','fixed'),('path_rho05','eligible_fixed'),('path_rho10','eligible_fixed'),
                    ('path_rho05','GRAM_filter'),('path_rho10','GRAM_filter')]:
            pairs[a+'_vs_'+b]=(a,b)
        assert set(summary['comparisons'])==set(pairs)
        for name,rr in ranks.items():
            for k,v in calculate(rr).items():
                np.testing.assert_allclose(v,summary['metrics'][name][k],atol=1e-14,rtol=0)
            np.testing.assert_allclose(np.mean([len(r['arms'][name]['items']) for r in evaluation]),
                                       summary['mean_eligible_candidate_count'][name],atol=1e-14,rtol=0)
        for name,(a,b) in pairs.items():
            assert summary['comparison_pairs'][name]==[a,b]
            for k,v in compare(ranks[a],ranks[b]).items():
                np.testing.assert_allclose(v,summary['comparisons'][name][k],atol=1e-12,rtol=0)
        for path,expected_hash in read(folder/'loaded_sources.json').items():
            assert digest(ROOT/path)==expected_hash,path
        checked[domain]=dict(users=512,predictions_checked=predictions,cohort_exclusion_verified=True,
            item_paths_and_filtering_verified=True,metrics_and_bootstrap_recomputed=True,
            summary_sha256=digest(folder/'summary.json'))
    result=dict(state='VERIFIED',verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),domains=checked,
                frozen_inputs_verified=len(manifest),independent_of_runner_and_policy=True,test_read=False)
    temp=OUT/f'verification.{os.getpid()}.tmp'; temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(OUT/'verification.json'); print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
