"""Separate feature arithmetic, fold, prediction and decision accounting for 21.13."""
import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from category_relation_core import CategoryCatalog
from evidence_decomposition_core import read_users

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/category_relations_20260925_v1'
PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def independent_features(engine,items,b,history):
    hist=sorted(set(history));result=[]
    for a in items:
        row=[]
        for table in (engine.coarse,engine.fine):
            pair=[]
            for item in (int(a),b):
                label=table[item]
                n=sum(bool(label) and table[h]==label for h in hist)
                df=sum(bool(label) and v==label for v in table[1:])
                pair.append(np.array([bool(label),np.log1p(n),n/max(len(hist),1),np.log1p(df)],np.float32))
            row.extend(np.concatenate([pair[0],pair[1],pair[0]-pair[1],
                [float(bool(table[int(a)]) and table[int(a)]==table[b])]]))
        result.append(row)
    return np.asarray(result,np.float32).reshape(-1,26)


def choice(engine,items,b,history,fused_scores):
    hist=set(history);count_b=sum(bool(engine.fine[b]) and engine.fine[h]==engine.fine[b] for h in hist)
    qualified=[]
    for a,score in zip(items,fused_scores):
        a=int(a);label=engine.fine[a]
        count=sum(bool(label) and engine.fine[h]==label for h in hist)
        if label and engine.fine[b] and count>=2 and count>count_b:
            qualified.append((-count,-float(score),a))
    return min(qualified)[2] if qualified else 0


def check_domain(domain,summary):
    folder=OUT/domain;catalog=json.loads((RAW/domain/'catalog.json').read_text());engine=CategoryCatalog(catalog,domain)
    assert 'target' not in inspect.signature(engine.paired).parameters
    result=dict(manual_feature_pairs=0,manual_predictions=0,policy_user_decisions=0,rule_user_decisions=0,
                guided_rules_independently_reselected=0,source_category_fields_verified=0)
    # Parse field boundaries independently for the catalogue entries, before testing arithmetic.
    for i,passage in enumerate(catalog['passages'][1:],1):
        field={};boundaries=[]
        # Semicolons inside HTML entities / descriptions are values, not field boundaries.
        for boundary,start in [(0,0)]+[(j,j+1) for j,char in enumerate(passage) if char==';']:
            colon=passage.find(':',start)
            if colon<0:continue
            key=passage[start:colon].strip()
            if key in ('title','categories','brand','description','price','salesrank','item','similar items'):
                boundaries.append((boundary,colon+1,key))
        for j,(_,start,key) in enumerate(boundaries):
            end=boundaries[j+1][0] if j+1<len(boundaries) else len(passage)
            field.setdefault(key,passage[start:end].strip())
        assert field.get('categories','')==engine.fields[i].get('categories',''),(domain,i)
        result['source_category_fields_verified']+=1
    fitted=set()
    for kind in ('train','validation'):
        users=read_users(PRIOR/domain/kind/'users.csv')
        source=(RAW if kind=='train' else FULL)/domain
        samples={r['user_id']:r for r in json.loads((source/'samples.json').read_text())}
        if kind=='train':fitted={r['user_id'] for r in users}
        else:assert not fitted&{r['user_id'] for r in users if r['cohort']=='primary'}
        with np.load(PRIOR/domain/kind/'pairs.npz') as z:pairs={k:z[k] for k in z.files}
        cats={k:np.load(folder/kind/(k+'.npy'),mmap_mode='r') for k in ('category','shuffled_category')}
        n=len(pairs['items']);score=np.load(PRIOR/domain/kind/'score.npy',mmap_mode='r')[:n]
        evidence=np.load(PRIOR/domain/kind/'evidence.npy',mmap_mode='r')[:n]
        for x in cats.values():
            assert np.isfinite(x).all()
            np.testing.assert_array_equal(x[:,8:12],x[:,:4]-x[:,4:8])
            np.testing.assert_array_equal(x[:,21:25],x[:,13:17]-x[:,17:21])
        # A deterministic sample of three candidate pairs per 64 evenly spaced users per split.
        for ui in np.unique(np.linspace(0,len(users)-1,64,dtype=int)):
            r=users[ui];lo,hi=r['start'],r['end']
            ix=np.unique(np.linspace(lo,hi-1,min(3,hi-lo),dtype=int)) if hi>lo else []
            for key,who in [('category',r['user_id']),('shuffled_category',r['donor_user'])]:
                got=independent_features(engine,pairs['items'][ix],r['boundary'],samples[who]['full_history'])
                np.testing.assert_array_equal(got,cats[key][ix]);result['manual_feature_pairs']+=len(ix)
        for r in users:
            own=samples[r['user_id']];donor=samples[r['donor_user']]
            assert len(own['full_history'])==len(donor['full_history'])
            if kind=='train':
                # The fold formula is checked independently from the stored records.
                def fold(uid):return int.from_bytes(hashlib.sha256(f'20260925|selective-evidence|{domain}|{uid}'.encode()).digest()[:8],'big')%5
                assert fold(r['user_id'])==r['fold']==fold(r['donor_user'])
            else:
                assert (own['training_holdout_role'],own['previously_evaluated'])==(donor['training_holdout_role'],donor['previously_evaluated'])
        dataset='train_oof' if kind=='train' else 'validation'
        with np.load(folder/(dataset+'_predictions.npz')) as z:preds={k:z[k] for k in z.files}
        for arm,pred in preds.items():
            for k in (range(5) if kind=='train' else (None,)):
                with np.load(folder/(arm+('_fold'+str(k) if k is not None else '_final')+'.npz')) as model:
                    if k is None:assert set(model['fit_users'])==fitted
                    else:assert set(model['fit_users'])=={r['user_id'] for r in users if r['fold']!=k}
                    options=[r for r in users if r['end']>r['start'] and (k is None or r['fold']==k)]
                    for r in (options[0],options[len(options)//2],options[-1]):
                        j=r['start'];x=score[j].tolist()
                        if arm in ('prior_full','prior_full_category'):x+=evidence[j].tolist()
                        if arm=='coarse':x+=cats['category'][j,:13].tolist()
                        if arm in ('category','prior_full_category'):x+=cats['category'][j].tolist()
                        if arm=='shuffled_category':x+=cats['shuffled_category'][j].tolist()
                        value=float(model['intercept'])+sum((v-m)/s*c for v,m,s,c in zip(x,model['mean'],model['std'],model['coef']))
                        assert abs(value-pred[j])<1e-9;result['manual_predictions']+=1
        cohorts=('train_oof',) if kind=='train' else ('primary','previous512','seen_fit_user_later_event','all')
        for cohort in cohorts:
            rows=[r for r in users if cohort in ('train_oof','all') or r['cohort']==cohort]
            with np.load(folder/(cohort+'_decisions.npz')) as z:dec={k:z[k] for k in z.files}
            assert dec['users'].tolist()==[r['user_id'] for r in rows]
            eligible=[j for j,r in enumerate(rows) if r['end']>r['start']]
            for ai,arm in enumerate(dec['arms']):
                best={};values={}
                for j in eligible:
                    r=rows[j];p=min(range(r['start'],r['end']),key=lambda t:(-preds[arm][t],int(pairs['items'][t])))
                    best[j]=int(pairs['items'][p]);values[j]=float(preds[arm][p])
                    assert best[j]==dec['items'][ai,j] and values[j]==dec['scores'][ai,j]
                for pi,policy in enumerate(dec['policies']):
                    if policy=='positive':acted={j for j in eligible if values[j]>0}
                    else:acted=set(sorted(eligible,key=lambda j:(-values[j],rows[j]['user_id']))[:int(int(policy[-2:])/100*len(eligible))])
                    new=[];changes=[]
                    for j,r in enumerate(rows):
                        change=(int(best[j]==r['target'])-int(r['boundary']==r['target'])) if j in acted else 0
                        assert change==dec['delta'][pi,ai,j] and (j in acted)==bool(dec['actions'][pi,ai,j])
                        rank=10 if change==1 else 2**30 if change==-1 else r['old_rank']
                        new.append(1/np.log2(rank+1) if rank<=10 else 0.);changes.append(change)
                        result['policy_user_decisions']+=1
                    m=summary['cohorts'][cohort]['policies'][arm+'_'+policy]
                    assert (len(acted),changes.count(1),changes.count(-1),sum(changes))==(m['actions'],m['gained10'],m['lost10'],m['net_hits'])
                    assert abs(float(np.mean(new))-m['N10'])<1e-12
            # Full independent selection of deterministic category rules for every user.
            for key in ('category','shuffled_category'):
                with np.load(folder/(cohort+'_'+key+'_rule.npz')) as z:rules={k:z[k] for k in z.files}
                for j,r in enumerate(rows):
                    sl=slice(r['start'],r['end']);uid=r['user_id'] if key=='category' else r['donor_user']
                    a=choice(engine,pairs['items'][sl],r['boundary'],samples[uid]['full_history'],score[sl,18])
                    delta=(int(a==r['target'])-int(r['boundary']==r['target'])) if a else 0
                    assert a==rules['items'][j] and delta==rules['delta'][j];result['rule_user_decisions']+=1
                m=summary['direct'][cohort][key]['rule'];d=rules['delta']
                assert (int((rules['items']>0).sum()),int((d==1).sum()),int((d==-1).sum()))==(m['actions'],m['gained10'],m['lost10'])
    # Guided checks: count every saved action; independently regenerate candidates and rule choices
    # for 256 target-independent user indices and all non-neutral real or shuffled outcomes.
    with (folder/'guided_rule.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k in ('target','old_rank','boundary','candidates','category_item','category_delta','shuffled_category_item','shuffled_category_delta'):row[k]=int(row[k])
    valid_users=read_users(PRIOR/domain/'validation/users.csv');vmap={r['user_id']:r for r in valid_users}
    samples=json.loads((FULL/domain/'samples.json').read_text());smap={s['user_id']:s for s in samples}
    ex=np.load(FULL/domain/'experts.npy',mmap_mode='r');fused=np.load(FULL/domain/'fused.npy',mmap_mode='r')
    selected=set(np.linspace(0,len(rows)-1,256,dtype=int).tolist())|{i for i,r in enumerate(rows) if r['category_delta'] or r['shuffled_category_delta']}
    with (FULL/domain/'predictions.jsonl').open() as f:
        for i,line in enumerate(f):
            if i not in selected:continue
            p=json.loads(line);s=samples[i];r=rows[i];assert s['user_id']==p['user_id']==r['user_id']
            excluded=set(s['full_history'][-20:]);g=p['guided']
            order=sorted([(float(v),int(a)) for a,v in zip(g['ids'],g['scores']) if int(a) not in excluded and np.isfinite(v)],key=lambda t:(-t[0],t[1]))
            top=[a for v,a in order[:10]];assert top[9]==r['boundary']
            experts=set();ids=[a for a in range(1,ex.shape[2]+1) if a not in excluded]
            for scores in ex[i]:experts.update(sorted(ids,key=lambda a:(-float(scores[a-1]),a))[:10])
            proposals=sorted(experts-set(top));assert len(proposals)==r['candidates']
            for key,uid in [('category',r['user_id']),('shuffled_category',vmap[r['user_id']]['donor_user'])]:
                a=choice(engine,proposals,r['boundary'],smap[uid]['full_history'],fused[i,np.array(proposals,dtype=int)-1])
                assert a==r[key+'_item'];result['guided_rules_independently_reselected']+=1
    for cohort in ('primary','previous512','seen_fit_user_later_event','all'):
        take=[r for r in rows if cohort=='all' or r['cohort']==cohort]
        for key in ('category','shuffled_category'):
            d=[(int(r[key+'_item']==r['target'])-int(r['boundary']==r['target'])) if r[key+'_item'] else 0 for r in take]
            assert d==[r[key+'_delta'] for r in take]
            m=summary['guided']['cohorts'][cohort][key]
            assert (sum(r[key+'_item']>0 for r in take),d.count(1),d.count(-1),sum(d))==(m['actions'],m['gained10'],m['lost10'],m['net_hits'])
    result.update(all_decision_metrics_exact=True,source_parser_agreement=True,user_fold_and_donor_constraints=True)
    return result


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETED'
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
    previous=json.loads((ROOT/'artifacts/phase21/evidence_decomposition_20260925_v1/manifest.json').read_text())
    overlap=set(previous)&set(manifest)
    assert all(previous[p]==manifest[p] for p in overlap)
    result={d:check_domain(d,summary['domains'][d]) for d in ('Toys','Beauty')}
    result.update(source_hash_checks=len(manifest),prior_manifest_overlap=len(overlap),
                  verification_script_sha256=sha(Path(__file__)))
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
