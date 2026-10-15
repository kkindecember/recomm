"""Target-free proposals/evidence and a fixed weighted ridge benefit probe."""
import hashlib
from collections import defaultdict

import numpy as np

from analyze_history_evidence import CatalogEvidence
from analyze_expert_signal_loss import full_ranking

BASE_NAMES=([f'{side}_{e}_score' for side in ('a','b','difference') for e in ('sas','cf','text')]
    +[f'{side}_{e}_reciprocal_rank' for side in ('a','b') for e in ('sas','cf','text')]
    +[f'a_{e}_top10' for e in ('sas','cf','text')]
    +['a_fused','b_fused','fused_difference','a_fused_reciprocal_rank','gap10_11','gap1_10']
    +['weight_sas','weight_cf','weight_text','history_penalty','a_log_frequency','b_log_frequency','log_history_length'])
ITEM_NAMES=['log_brand_support','log_word_support','log_phrase_support','word_idf_support','phrase_idf_support',
    'older_exact_repeat','cf_log_support','cf_max_share','cf_log_effective_support','cf_recent_share','cf_no_edge','cf_log_total',
    'text_max','text_second','text_mean','text_std','text_recent_older_difference','text_last','text_top2_mean']
EVIDENCE_NAMES=[f'{side}_{name}' for side in ('a','b','difference') for name in ITEM_NAMES]
ARMS=('score','evidence','shuffled')


def fold(user,domain):
    return int.from_bytes(hashlib.sha256(f'20260925|selective-evidence|{domain}|{user}'.encode()).digest()[:8],'big')%5


def donors(samples,domain,training):
    groups=defaultdict(list)
    for i,s in enumerate(samples):
        group=fold(s['user_id'],domain) if training else (s['training_holdout_role'],s['previously_evaluated'])
        groups[(group,len(s['full_history']))].append(i)
    result=np.arange(len(samples))
    for ix in groups.values():
        ix=sorted(ix,key=lambda i:samples[i]['user_id'])
        result[ix]=ix[1:]+ix[:1]
    return result


def proposals(experts,fused,history,weights,penalty,counts):
    """All observable inputs; deliberately no target argument."""
    eligible=np.ones(fused.size,bool);eligible[np.asarray(history[-20:],int)-1]=False
    order,frank=full_ranking(fused,eligible)
    eranks=[];suggestions=set()
    for scores in experts:
        eo,rr=full_ranking(scores,eligible);eranks.append(rr);suggestions.update(eo[:10].tolist())
    ids=np.array(sorted(suggestions-set(order[:10])),dtype=np.int32)
    b=int(order[9]);ranks=np.stack(eranks)
    values=[]
    for a in ids:
        av=experts[:,a-1];bv=experts[:,b-1]
        values.append(np.concatenate([av,bv,av-bv,1/ranks[:,a],1/ranks[:,b],(ranks[:,a]<=10).astype(float),
            [fused[a-1],fused[b-1],fused[a-1]-fused[b-1],1/frank[a],fused[b-1]-fused[order[10]-1],
             fused[order[0]-1]-fused[b-1]],weights,[penalty,np.log1p(counts[a]),np.log1p(counts[b]),np.log1p(len(history))]]))
    x=np.asarray(values,dtype=np.float32).reshape(len(ids),len(BASE_NAMES))
    assert len(ids)<=30 and not(set(ids)&set(history[-20:]))
    return ids,b,x,frank,order[:10],ranks


class Evidence:
    def __init__(self,catalog,embedding,cf):
        self.lexical=CatalogEvidence(catalog)
        self.n=len(catalog['items'])-1
        self.emb=embedding.astype(np.float64)
        self.emb/=np.maximum(np.linalg.norm(self.emb,axis=1,keepdims=True),1e-12)
        index={item:i for i,item in enumerate(catalog['items'])}
        assert cf['training_only'] and cf['scoring_history_length']==20
        self.neighbors={index[key]:{index[r['item_id']]:float(np.float32(r['score']))
            for r in rows if r['item_id'] in index} for key,rows in cf['neighbors'].items() if key in index}

    def item_features(self,items,history):
        """Features from supplied history; no target or label."""
        items=np.asarray(items,int)
        # Preserve last occurrence, so repeated events cannot inflate distinct support.
        unique=list(dict.fromkeys(reversed(history)))[::-1]
        context=self.lexical.history(history)
        sim=self.emb[items-1]@self.emb[np.asarray(unique)-1].T
        window=history[-20:];ch=sorted(set(window));hi={h:i for i,h in enumerate(ch)}
        cf=np.zeros((len(items),len(ch)));recent=np.zeros(len(items))
        for pos,h in enumerate(window):
            contribution=np.array([self.neighbors.get(h,{}).get(int(a),0.) for a in items])*(pos+1)/len(window)
            cf[:,hi[h]]+=contribution
            if pos>=len(window)-5:recent+=contribution
        out=[]
        for j,a in enumerate(items):
            e=self.lexical.evidence(int(a),context)
            def strength(tokens,df,source):
                return max((np.log((self.n+1)/(df[t]+1))*np.log1p(len(source.get(t,[]))) for t in tokens),default=0.)
            word=strength(self.lexical.rare_words[a],self.lexical.word_df,context['words'])
            phrase=strength(self.lexical.rare_phrases[a],self.lexical.phrase_df,context['phrases'])
            support=cf[j];total=support.sum();nonzero=int((support>0).sum())
            ss=np.sort(sim[j]);last=min(5,len(unique));older=sim[j,:-last].mean() if len(unique)>last else sim[j,-last:].mean()
            out.append([np.log1p(e['brand_count']),np.log1p(e['word_count']),np.log1p(e['phrase_count']),
                word,phrase,e['older_exact_item_repeat'],np.log1p(nonzero),support.max()/total if total else 0.,
                np.log1p(total**2/(support@support)) if total else 0.,recent[j]/total if total else 0.,float(total==0),np.log1p(total),
                ss[-1],ss[-2] if len(ss)>1 else ss[-1],ss.mean(),ss.std(),sim[j,-last:].mean()-older,
                sim[j,-1],ss[-2:].mean()])
        result=np.asarray(out,np.float32)
        assert result.shape==(len(items),len(ITEM_NAMES)) and np.isfinite(result).all()
        return result

    def paired(self,items,b,history):
        if not len(items):return np.empty((0,len(EVIDENCE_NAMES)),np.float32)
        f=self.item_features(list(items)+[b],history)
        a=f[:-1];bv=np.repeat(f[-1:],len(items),axis=0)
        return np.concatenate([a,bv,a-bv],axis=1)

    def check(self,items,history):
        f=self.item_features(items,history)
        for j,a in enumerate(items):
            w=history[-20:]
            direct=sum((i+1)/len(w)*self.neighbors.get(h,{}).get(int(a),0.) for i,h in enumerate(w))
            assert abs(f[j,11]-np.log1p(direct))<1e-6
            sims=[float(self.emb[int(a)-1]@self.emb[h-1]) for h in set(history)]
            assert abs(f[j,12]-max(sims))<1e-6
        return True


def ridge_fit(x,y,w):
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float);w=w/w.sum()
    mean=w@x;variance=w@((x-mean)**2);std=np.maximum(np.sqrt(variance),1e-5)
    z=(x-mean)/std;intercept=float(w@y)
    coef=np.linalg.solve((z.T*w)@z+np.eye(x.shape[1]),z.T@(w*(y-intercept)))
    return dict(mean=mean,std=std,coef=coef,intercept=intercept)


def predict(model,x):
    return (np.asarray(x,float)-model['mean'])/model['std']@model['coef']+model['intercept']


def model_checks():
    rng=np.random.default_rng(21);x=rng.normal(size=(37,8));y=rng.normal(size=37);w=rng.uniform(size=37)
    model=ridge_fit(x,y,w);w/=w.sum();z=(x-model['mean'])/model['std'];xx=np.column_stack([np.ones(len(x)),z])
    penalty=np.diag([0.]+[1.]*x.shape[1]);direct=np.linalg.solve((xx.T*w)@xx+penalty,xx.T@(w*y))
    np.testing.assert_allclose(predict(model,x),xx@direct,atol=1e-12,rtol=1e-12)
    duplicate=ridge_fit(np.repeat(x,2,0),np.repeat(y,2),np.repeat(w/2,2))
    np.testing.assert_allclose(predict(model,x),predict(duplicate,x),atol=1e-12,rtol=1e-12)
    return dict(ridge_independent_direct_solution=True,duplicate_weight_invariance=True)
