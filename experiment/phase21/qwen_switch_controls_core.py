"""Matched Qwen4 single/pair scores and a predeclared development choice."""
import numpy as np

ARMS = ('pair_qwen_raw','pair_qwen_loo','pair_qwen_raw_penalty','pair_qwen_loo_penalty',
        'qwen_raw','qwen_loo','qwen_raw_penalty','qwen_loo_penalty')
FRESH = tuple(a for a in ARMS if a not in ('pair_qwen_raw','qwen_raw'))


def scores(raw,loo,history,penalty=.05):
    raw,loo = np.asarray(raw,np.float32),np.asarray(loo,np.float32)
    assert raw.ndim==2 and raw.shape==(len(loo),2)
    result = dict(pair_qwen_raw=np.float32(.5)*raw[:,0]+np.float32(.5)*raw[:,1],
                  pair_qwen_loo=np.float32(.5)*raw[:,0]+np.float32(.5)*loo,
                  qwen_raw=raw[:,1].copy(),qwen_loo=loo.copy())
    ids = np.unique(np.asarray(history,np.int64))-1
    assert len(ids)>0 and ids.min()>=0 and ids.max()<len(loo)
    for name in ('pair_qwen_raw','pair_qwen_loo','qwen_raw','qwen_loo'):
        value = result[name].copy()
        value[ids]-=np.float32(penalty)
        result[name+'_penalty'] = value
    return {a:result[a] for a in ARMS}


def choose(domains):
    """One expert for both domains; prioritize the weakest relative N10 gain.

    All candidates share exactly the same SASRec+penalty reference. Comparing
    against their own content single is an extra success check, not a way to
    reward a weak single-expert baseline. This is a development choice only.
    """
    rows = {}
    for text in ('bge','qwen'):
        gains,hr,success = {},[],{}
        for domain,mm in domains.items():
            pair = mm[f'pair_{text}_loo_penalty']
            sas,content = mm['sas_penalty'],mm[f'{text}_loo_penalty']
            gains[domain] = pair['N10']/sas['N10']-1
            hr.append(pair['H10']/sas['H10']-1)
            success[domain] = pair['N10']>max(sas['N10'],content['N10'])
        rows[text] = dict(relative_N10_vs_common_sas=gains,worst_domain_relative_N10=min(gains.values()),
            mean_relative_N10=float(np.mean(list(gains.values()))),mean_relative_H10=float(np.mean(hr)),
            pair_above_both_matching_singles_by_mean=success,
            both_domains_above_both_singles_by_mean=all(success.values()))
    winner = max(rows,key=lambda t:(round(rows[t]['worst_domain_relative_N10'],12),
        round(rows[t]['mean_relative_N10'],12),round(rows[t]['mean_relative_H10'],12),t=='bge'))
    return dict(selected_development_text_expert=winner,candidates=rows,
        rule='maximize worst-domain relative N10 vs common SASRec+penalty; ties: mean relative N10, mean relative H10, BGE',
        primary_setting='leave-self-out + history penalty .05',independent_confirmation=False,
        statistical_superiority_claim=False)


def tests():
    raw = np.arange(18,dtype=np.float32).reshape(9,2)/20
    loo = raw[:,1].copy();loo[[0,8]]*=.2
    s = scores(raw,loo,[1,9,1])
    for name in ('pair_qwen_raw','pair_qwen_loo','qwen_raw','qwen_loo'):
        base = s[name].copy();base[[0,8]]-=np.float32(.05)
        np.testing.assert_array_equal(s[name+'_penalty'],base)
        np.testing.assert_array_equal(scores(raw,loo,[1,9],0.)[name+'_penalty'],s[name])
    np.testing.assert_array_equal(s['pair_qwen_raw'],.5*raw[:,0]+.5*raw[:,1])
    np.testing.assert_array_equal(s['pair_qwen_loo'],.5*raw[:,0]+.5*loo)
    fake = {}
    for domain,gains in [('Toys',(.10,.20)),('Beauty',(-.02,-.03))]:
        fake[domain] = dict(sas_penalty=dict(N10=.1,H10=.2))
        for text,g in zip(('bge','qwen'),gains):
            fake[domain][f'pair_{text}_loo_penalty']=dict(N10=.1*(1+g),H10=.2)
            fake[domain][f'{text}_loo_penalty']=dict(N10=.08,H10=.2)
    choice = choose(fake)
    assert choice['selected_development_text_expert']=='bge'
    assert not choice['candidates']['bge']['both_domains_above_both_singles_by_mean']
    return dict(state='PASSED',single_pair_formulas=True,full_history_penalty_only=True,
        zero_penalty_exact=True,duplicates_penalized_once=True,common_sas_reference=True,
        weakest_domain_rule=True,negative_results_not_declared_success=True)
