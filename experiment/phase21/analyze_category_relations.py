"""21.13 fixed offline category-relation audit, with reproducible CPU probes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from analyze_selective_evidence import sha, write, csv_write
from category_relation_core import CategoryCatalog, NAMES, qualifying, rule_choice, top_items
from evidence_decomposition_core import read_users
from selective_evidence_core import donors, predict, ridge_fit

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
RAW = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL = ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT = ROOT/'artifacts/phase21/category_relations_20260925_v1'
PROTOCOL = ROOT/'plan/第二十一阶段/21.13_具体类别关系能否识别专家建议_20260925.md'
ARMS = ('score', 'coarse', 'category', 'shuffled_category', 'prior_full', 'prior_full_category')
CONTRASTS = [('category', 'score'), ('category', 'coarse'), ('category', 'shuffled_category'),
             ('prior_full_category', 'prior_full')]
POLICIES = ('budget05', 'budget10', 'budget20', 'positive')
UNIT = 1/np.log2(11)


def use(path, manifest, expected=None):
    digest = sha(path); manifest[str(path.relative_to(ROOT))] = digest
    if expected is not None: assert digest == expected, str(path)


def load_features(domain, kind, engine, manifest):
    folder = PRIOR/domain/kind; dest = OUT/domain/kind; dest.mkdir()
    for name in ('users.csv', 'pairs.npz', 'score.npy', 'evidence.npy'):
        use(folder/name, manifest)
    users = read_users(folder/'users.csv')
    with np.load(folder/'pairs.npz', allow_pickle=False) as z: data = {k:z[k] for k in z.files}
    n = len(data['items'])
    data.update(users=users, score=np.load(folder/'score.npy', mmap_mode='r')[:n],
                evidence=np.load(folder/'evidence.npy', mmap_mode='r')[:n])
    source = (RAW if kind == 'train' else FULL)/domain
    use(source/'samples.json', manifest)
    all_samples = json.loads((source/'samples.json').read_text())
    by_user = {r['user_id']:r for r in all_samples}
    samples = [by_user[r['user_id']] for r in users]
    ds = donors(samples, domain, kind == 'train')
    arrays = {key:np.lib.format.open_memmap(dest/(key+'.npy'), mode='w+', dtype='float32', shape=(n,26))
              for key in ('category', 'shuffled_category')}
    selected = set(sorted(range(len(users)), key=lambda i:hashlib.sha256(
        ('21.13|'+domain+'|'+users[i]['user_id']).encode()).digest())[:20])
    cards = []
    for i,(r,s) in enumerate(zip(users,samples)):
        sl = slice(r['start'], r['end']); items = data['items'][sl]; b = r['boundary']
        assert r['donor_user'] == samples[int(ds[i])]['user_id']
        arrays['category'][sl] = engine.paired(items, b, s['full_history'])
        arrays['shuffled_category'][sl] = engine.paired(items, b, samples[int(ds[i])]['full_history'])
        if i in selected:
            cards.append(dict(user_id=r['user_id'], history=[engine.card(h) for h in sorted(set(s['full_history']))],
                boundary=engine.card(b, s['full_history']), proposals=[engine.card(int(a), s['full_history']) for a in items]))
        if (i+1) % 4096 == 0: print(domain, kind, 'relations', i+1, '/', len(users), flush=True)
    for array in arrays.values(): array.flush()
    data.update(arrays); data['samples'] = samples
    write(dest/'blind_user_cards.json', cards)
    write(dest/'metadata.json', dict(users=len(users), pairs=n, donor_changed=int((ds != np.arange(len(ds))).sum())))
    return data


def matrix(data, arm):
    base = data['score']
    if arm == 'score': return base
    if arm == 'coarse': return np.concatenate([base, data['category'][:,:13]], axis=1)
    if arm in ('category', 'shuffled_category'): return np.concatenate([base, data[arm]], axis=1)
    if arm == 'prior_full': return np.concatenate([base, data['evidence']], axis=1)
    assert arm == 'prior_full_category'
    return np.concatenate([base, data['evidence'], data['category']], axis=1)


def cohort_indices(data, cohort):
    return [i for i,r in enumerate(data['users']) if cohort in ('all','train_oof') or r['cohort'] == cohort]


def metrics(delta, action, old):
    gains = int((delta == 1).sum()); losses = int((delta == -1).sum())
    return dict(users=len(old), actions=int(action.sum()), gained10=gains, lost10=losses,
        net_hits=gains-losses, neutral_actions=int(action.sum())-gains-losses,
        delta_N10=float(delta.mean())*UNIT,
        N10=float(np.where(old<=10,1/np.log2(old+1),0.).mean())+float(delta.mean())*UNIT,
        H10=float((old<=10).mean())+float(delta.mean()))


def bootstrap(deltas):
    rng = np.random.default_rng(20260925); output = []; n = len(deltas)
    for _ in range(20):
        ix = rng.integers(0,n,size=(100,n)); output.append(deltas[ix].mean(axis=1)*UNIT)
    return np.concatenate(output)


def evaluate(data, preds, cohort, folder):
    ix = cohort_indices(data, cohort); rows = [data['users'][i] for i in ix]; n = len(rows)
    old = np.array([r['old_rank'] for r in rows]); targets = np.array([r['target'] for r in rows])
    boundaries = np.array([r['boundary'] for r in rows]); names = np.array([r['user_id'] for r in rows])
    available = np.array([r['end'] > r['start'] for r in rows])
    items = np.zeros((len(ARMS),n), int); scores = np.full((len(ARMS),n), -np.inf)
    for ai,arm in enumerate(ARMS):
        for j,r in enumerate(rows):
            if not available[j]: continue
            p = r['start']+int(np.argmax(preds[arm][r['start']:r['end']]))
            items[ai,j] = data['items'][p]; scores[ai,j] = preds[arm][p]
    actions = np.zeros((len(POLICIES),len(ARMS),n),bool); delta = np.zeros(actions.shape,np.int8)
    result = dict(users=n, old_N10=float(np.where(old<=10,1/np.log2(old+1),0.).mean()), policies={}, contrasts={})
    for pi,policy in enumerate(POLICIES):
        for ai,arm in enumerate(ARMS):
            if policy == 'positive': act = available & (scores[ai] > 0)
            else:
                count = int(int(policy[-2:])/100*int(available.sum()))
                order = sorted(np.flatnonzero(available),key=lambda j:(-scores[ai,j],names[j]))
                act = np.zeros(n,bool); act[order[:count]] = True
            d = act.astype(np.int8)*((items[ai] == targets).astype(np.int8)-(boundaries == targets).astype(np.int8))
            actions[pi,ai] = act; delta[pi,ai] = d
            result['policies'][arm+'_'+policy] = metrics(d,act,old)
    for policy in POLICIES:
        for a,b in CONTRASTS:
            ma = result['policies'][a+'_'+policy]; mb = result['policies'][b+'_'+policy]
            result['contrasts'][a+'_vs_'+b+'_'+policy] = dict(net_hit_difference=ma['net_hits']-mb['net_hits'],
                delta_N10=ma['delta_N10']-mb['delta_N10'])
    if cohort in ('train_oof','primary'):
        boot = bootstrap(delta[1].T)
        for ai,arm in enumerate(ARMS):
            result['policies'][arm+'_budget10']['ci95'] = np.quantile(boot[:,ai],[.025,.975]).tolist()
        for a,b in CONTRASTS:
            result['contrasts'][a+'_vs_'+b+'_budget10']['ci95'] = np.quantile(
                boot[:,ARMS.index(a)]-boot[:,ARMS.index(b)],[.025,.975]).tolist()
    np.savez_compressed(folder/(cohort+'_decisions.npz'),users=names,target=targets,boundary=boundaries,old_rank=old,
        arms=np.array(ARMS),policies=np.array(POLICIES),items=items,scores=scores,actions=actions,delta=delta)
    return result


def direct_audit(data, cohort, folder):
    ix = cohort_indices(data,cohort); result = {}; details = []
    for key in ('category','shuffled_category'):
        groups = {s:{y:dict(users=0, pair_weight=0., qualifying_weight=0., qualifying_pairs=0,pairs=0)
                    for y in ('rescuable_user','boundary_correct_user','neither_user')}
                  for s in ('all','same_coarse_different_fine')}
        choices = []; delta = []
        for i in ix:
            r = data['users'][i]; sl = slice(r['start'],r['end']); ids = data['items'][sl]; x = data[key][sl]
            qual = qualifying(x); label = data['labels'][sl]; weights = data['weight'][sl]
            group = 'rescuable_user' if (label>0).any() else 'boundary_correct_user' if (label<0).any() else 'neither_user'
            for name,mask in [('all',np.ones(len(ids),bool)),('same_coarse_different_fine',
                    (x[:,12]>0)&(x[:,13]>0)&(x[:,17]>0)&(x[:,25]==0))]:
                cell = groups[name][group]; cell['users'] += int(mask.any())
                cell['pair_weight'] += float(weights[mask].sum()); cell['qualifying_weight'] += float(weights[mask&qual].sum())
                cell['pairs'] += int(mask.sum()); cell['qualifying_pairs'] += int((mask&qual).sum())
            a = rule_choice(ids,x,data['score'][sl,18])
            change = int(bool(a))*((int(a)==r['target'])-(r['boundary']==r['target']))
            choices.append(a); delta.append(change)
            if key == 'category': details.append(dict(user_id=r['user_id'],item=a,hit_change=change))
        # For positive users also report whether the *actual* helpful candidate has evidence;
        # membership is outcome-defined only for this descriptive table, never for actions.
        target_table = {}
        for label_name,which in [('helpful',1),('harmful',-1),('neutral',0)]:
            all_n = yes = 0; mass = selected_mass = 0.; user_count = 0
            for i in ix:
                r=data['users'][i];sl=slice(r['start'],r['end']);mask=data['labels'][sl]==which
                q=qualifying(data[key][sl]);w=data['weight'][sl]
                all_n+=int(mask.sum());yes+=int((mask&q).sum());user_count+=int(mask.any())
                mass+=float(w[mask].sum());selected_mass+=float(w[mask&q].sum())
            target_table[label_name]=dict(pairs=all_n,qualifying_pairs=yes,users=user_count,
                user_weighted_qualifying_rate=selected_mass/mass if mass else None)
        delta=np.array(delta,np.int8);choices=np.array(choices,int);old=np.array([data['users'][i]['old_rank'] for i in ix])
        result[key]=dict(user_groups=groups,label_groups=target_table,rule=metrics(delta,choices>0,old))
        np.savez_compressed(folder/(cohort+'_'+key+'_rule.npz'),users=np.array([data['users'][i]['user_id'] for i in ix]),
            items=choices,delta=delta)
    return result,details


def guided_rule(domain,engine,manifest):
    folder=FULL/domain;users=read_users(PRIOR/domain/'validation/users.csv');by_user={r['user_id']:r for r in users}
    samples=json.loads((folder/'samples.json').read_text());ds=donors(samples,domain,False)
    expected=json.loads((folder/'score_cache_manifest.json').read_text())['sha256']
    for name in ('experts.npy','fused.npy'): use(folder/name,manifest,expected[name])
    for name in ('predictions.jsonl','ranks.npz'): use(folder/name,manifest)
    ex=np.load(folder/'experts.npy',mmap_mode='r');fused=np.load(folder/'fused.npy',mmap_mode='r')
    with np.load(folder/'ranks.npz') as z: reference=z['guided']
    rows=[]; case_pool={1:[],-1:[]}; checked=0
    with (folder/'predictions.jsonl').open() as f:
        for i,line in enumerate(f):
            pred=json.loads(line);s=samples[i];assert pred['user_id']==s['user_id']
            h=s['full_history']; ids=np.array(pred['guided']['ids']);scores=np.array(pred['guided']['scores'])
            valid=np.isfinite(scores)&~np.isin(ids,h[-20:]); order=ids[valid][np.lexsort((ids[valid],-scores[valid]))]
            assert len(order)>=10 and len(set(order))==len(order)
            pos=np.flatnonzero(order==s['target']);rank=int(pos[0]+1) if len(pos) else 2**30
            assert rank==reference[i];checked+=1
            suggestions=set()
            for e in ex[i]: suggestions.update(top_items(e,h[-20:]).tolist())
            a=np.array(sorted(suggestions-set(order[:10])),np.int32);b=int(order[9])
            row=dict(user_id=s['user_id'],cohort=by_user[s['user_id']]['cohort'],target=s['target'],old_rank=rank,boundary=b,
                     candidates=len(a))
            for key,history in [('category',h),('shuffled_category',samples[int(ds[i])]['full_history'])]:
                features=engine.paired(a,b,history);choice=rule_choice(a,features,fused[i,a-1])
                delta=int(bool(choice))*((choice==s['target'])-(b==s['target']))
                row[key+'_item']=choice;row[key+'_delta']=delta
                if key=='category' and delta in case_pool:
                    case_pool[delta].append((hashlib.sha256(s['user_id'].encode()).hexdigest(),s,choice,b))
            rows.append(row)
            if (i+1)%4096==0: print(domain,'guided_rule',i+1,'/',len(samples),flush=True)
    assert checked==len(samples)
    csv_write(OUT/domain/'guided_rule.csv',rows)
    summary={}
    for cohort in ('primary','previous512','seen_fit_user_later_event','all'):
        selected=[r for r in rows if cohort=='all' or r['cohort']==cohort]
        old=np.array([r['old_rank'] for r in selected]);deltas=[];summary[cohort]={}
        for key in ('category','shuffled_category'):
            d=np.array([r[key+'_delta'] for r in selected]);act=np.array([r[key+'_item']>0 for r in selected])
            summary[cohort][key]=metrics(d,act,old);deltas.append(d)
        if cohort=='primary':
            boot=bootstrap(np.column_stack(deltas))
            for j,key in enumerate(('category','shuffled_category')):
                summary[cohort][key]['ci95']=np.quantile(boot[:,j],[.025,.975]).tolist()
            summary[cohort]['real_minus_shuffled']=dict(delta_N10=float(np.mean(deltas[0]-deltas[1]))*UNIT,
                ci95=np.quantile(boot[:,0]-boot[:,1],[.025,.975]).tolist())
    examples=[]
    for delta,pool in case_pool.items():
        for _,s,a,b in sorted(pool,key=lambda r:r[0])[:3]:
            examples.append(dict(selection='posthoc_hash_within_gain_loss',hit_change=delta,user_id=s['user_id'],target=s['target'],
                proposal=engine.card(a,s['full_history']),boundary=engine.card(b,s['full_history']),
                history=[engine.card(h) for h in sorted(set(s['full_history']))]))
    write(OUT/domain/'guided_rule_examples.json',examples)
    return dict(cohorts=summary,original_guided_ranks_exact=checked)


def run_domain(domain,manifest):
    folder=OUT/domain;folder.mkdir();catalog_path=RAW/domain/'catalog.json'
    use(catalog_path,manifest,json.loads((PRIOR/'manifest.json').read_text())[str(catalog_path.relative_to(ROOT))])
    engine=CategoryCatalog(json.loads(catalog_path.read_text()),domain);write(folder/'catalog_audit.json',engine.audit())
    train=load_features(domain,'train',engine,manifest);valid=load_features(domain,'validation',engine,manifest)
    primary_users={r['user_id'] for r in valid['users'] if r['cohort']=='primary'}
    assert not primary_users&{r['user_id'] for r in train['users']}
    assert len(primary_users)==(15406 if domain=='Toys' else 17766)
    folds=np.array([train['users'][i]['fold'] for i in train['row_user']]);tp={};vp={};old_error=0.
    for arm in ARMS:
        x=matrix(train,arm);v=matrix(valid,arm);oof=np.full(len(x),np.nan)
        for k in range(5):
            fit=folds!=k;held=~fit;m=ridge_fit(x[fit],train['labels'][fit],train['weight'][fit])
            np.savez(folder/(arm+'_fold'+str(k)+'.npz'),**m,fit_users=np.array([r['user_id'] for r in train['users'] if r['fold']!=k]))
            oof[held]=predict(m,x[held])
        m=ridge_fit(x,train['labels'],train['weight']);np.savez(folder/(arm+'_final.npz'),**m,
            fit_users=np.array([r['user_id'] for r in train['users']]))
        p=np.concatenate([predict(m,v[j:j+10000]) for j in range(0,len(v),10000)])
        assert np.isfinite(oof).all() and np.isfinite(p).all();tp[arm]=oof;vp[arm]=p
        if arm in ('score','prior_full'):
            oldkey='score' if arm=='score' else 'evidence'
            for name,new in [('train_oof',oof),('validation',p)]:
                with np.load(PRIOR/domain/(name+'_predictions.npz')) as z:old=z[oldkey]
                old_error=max(old_error,float(np.max(np.abs(new-old))));np.testing.assert_allclose(new,old,atol=1e-10,rtol=0)
        print(domain,arm,'fitted',flush=True)
        write(OUT/'status.json',dict(state='RUNNING',stage='fit',domain=domain,arm=arm,pid=os.getpid()))
        del x,v
    np.savez(folder/'train_oof_predictions.npz',**tp);np.savez(folder/'validation_predictions.npz',**vp)
    cohorts={};direct={}
    for name,data,pred in [('train_oof',train,tp)]+[(c,valid,vp) for c in ('primary','previous512','seen_fit_user_later_event','all')]:
        cohorts[name]=evaluate(data,pred,name,folder);direct[name],_=direct_audit(data,name,folder)
        print(domain,'statistics',name,flush=True)
    previous=json.loads((PRIOR/domain/'summary.json').read_text())['cohorts']
    for c in cohorts:
        for arm,old in [('score','score'),('prior_full','evidence')]:
            for policy in POLICIES:
                a=cohorts[c]['policies'][arm+'_'+policy];b=previous[c]['policies'][old+'_'+policy]
                assert all(a[k]==b[k] for k in ('actions','gained10','lost10','net_hits'))
    guided=guided_rule(domain,engine,manifest)
    result=dict(cohorts=cohorts,direct=direct,guided=guided,max_original_prediction_error=old_error,
                original_two_arms_decisions_exact=True,primary_users_disjoint=True)
    write(folder/'summary.json',result);return result


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    scripts=[Path(__file__),Path(__file__).with_name('category_relation_core.py'),
        Path(__file__).with_name('selective_evidence_core.py'),Path(__file__).with_name('analyze_history_evidence.py')]
    manifest={}
    for p in scripts+[PROTOCOL]:use(p,manifest)
    for p in scripts:(OUT/p.name).write_text(p.read_text())
    (OUT/'protocol.md').write_text(PROTOCOL.read_text());write(OUT/'features.json',dict(category=NAMES,arms=ARMS,contrasts=CONTRASTS,L2=1))
    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),stage='start'))
    try:
        domains={d:run_domain(d,manifest) for d in ('Toys','Beauty')}
        summary=dict(state='COMPLETED',seconds=time.monotonic()-started,domains=domains,statistical_ridge_fits=72,
            GRAM_optimizer_steps=0,expert_optimizer_steps=0,gpu_inferences=0,test_read=False,exploratory=True,
            independent_confirmation=False,automatic_gpu_launch=False,new_independent_interactions=0)
        write(OUT/'summary.json',summary);write(OUT/'manifest.json',manifest)
        write(OUT/'status.json',{k:v for k,v in summary.items() if k!='domains'})
        print(json.dumps({k:v for k,v in summary.items() if k!='domains'}),flush=True)
    except BaseException as e:
        write(OUT/'status.json',dict(state='FAILED',pid=os.getpid(),error=repr(e)))
        (OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
