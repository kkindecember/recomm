"""Target-free title/history evidence; CPU-only cache diagnostics, no fitting."""
import csv
import hashlib
import html
import json
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
PRIOR = ROOT / 'artifacts/phase21/expert_signal_loss_20260924_v1'
OUT = ROOT / 'artifacts/phase21/history_evidence_20260924_v1'
PROTOCOL = ROOT / 'plan/第二十一阶段/21.8_历史证据能否识别文本异议_20260924.md'
ABSENT = 2**30
STOP = set('a an the and or for of to in on at by with from into as is are be this that these those '
           'it its your you our their new pack packs set sets piece pieces count counts size sizes '
           'ounce ounces oz fl ml liter litre liters litres gram grams inch inches cm mm lb lbs '
           'may vary colors colours color colour assorted one two three four six twelve per each '
           'amp quot reg trade nbsp'.split())
FEATURES = {'brand_any': ('brand_count', 1), 'brand_repeat': ('brand_count', 2),
            'word_repeat': ('word_count', 2), 'phrase_any': ('phrase_count', 1),
            'phrase_repeat': ('phrase_count', 2)}
RULE_FEATURES = {'brand_repeat': 'brand_count', 'word_repeat': 'word_count', 'phrase_repeat': 'phrase_count'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def fields(passage):
    matches = list(re.finditer(r'(?:^|;\s*)(item|similar items|title|categories|brand|description|price|salesrank)\s*:', passage))
    result = {}
    for j, m in enumerate(matches):
        result.setdefault(m.group(1), passage[m.end():matches[j+1].start() if j+1 < len(matches) else len(passage)].strip())
    return result


def tokens(text):
    text = html.unescape(text.lower())
    text = re.sub(r'&?(?:amp|quot|reg|trade|nbsp);', ' ', text)
    return re.findall(r'[a-z0-9]+', text)


def content(word):
    return len(word) >= 3 and word.isalpha() and word not in STOP


class CatalogEvidence:
    def __init__(self, catalog):
        self.catalog = catalog
        self.parsed = [fields(p) for p in catalog['passages']]
        self.brands, self.words, self.phrases = [], [], []
        self.brand_words = []
        for f in self.parsed:
            raw = f.get('brand', '').strip().lower()
            brand = '' if raw in ('','na','n/a','none','nan','unknown') else ' '.join(tokens(raw))
            self.brands.append(brand)
            self.brand_words.append(set(tokens(raw)) if brand else set())
            ts = tokens(f.get('title',''))
            self.words.append({w for w in ts if content(w)})
            self.phrases.append({(a,b) for a,b in zip(ts,ts[1:]) if content(a) and content(b)})
        self.word_df = Counter(w for words in self.words[1:] for w in words)
        self.phrase_df = Counter(p for phrases in self.phrases[1:] for p in phrases)
        limit = .01*(len(catalog['items'])-1)
        self.rare_words = [{w for w in ws if 2 <= self.word_df[w] <= limit and w not in bw}
                           for ws,bw in zip(self.words,self.brand_words)]
        self.rare_phrases = [{p for p in ps if 2 <= self.phrase_df[p] <= limit and not(set(p)&bw)}
                             for ps,bw in zip(self.phrases,self.brand_words)]

    def history(self, item_ids):
        unique = sorted(set(item_ids))
        brands, words, phrases = defaultdict(list), defaultdict(list), defaultdict(list)
        for item in unique:
            if self.brands[item]:
                brands[self.brands[item]].append(item)
            for word in self.words[item]:
                words[word].append(item)
            for phrase in self.phrases[item]:
                phrases[phrase].append(item)
        return dict(items=set(unique), brands=brands, words=words, phrases=phrases)

    def evidence(self, item, history):
        # No target, user outcome or recommendation label enters this method.
        brand = self.brands[item]
        bitems = history['brands'].get(brand, []) if brand else []
        def best(features, source):
            supported = [(feature,source.get(feature,[])) for feature in features if source.get(feature)]
            if not supported:
                return '', []
            feature, items = min(supported, key=lambda x: (-len(x[1]), x[0]))
            return ' '.join(feature) if isinstance(feature,tuple) else feature, items
        word,witems = best(self.rare_words[item],history['words'])
        phrase,pitems = best(self.rare_phrases[item],history['phrases'])
        return dict(brand_count=len(bitems), word_count=len(witems), phrase_count=len(pitems),
                    brand=brand, word=word, phrase=phrase, brand_support_items=bitems,
                    word_support_items=witems, phrase_support_items=pitems,
                    older_exact_item_repeat=int(item in history['items']))


def donor_mapping(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[(s['role'],len(s['full_history']))].append(s['user_id'])
    donors = {}
    for users in groups.values():
        users.sort()
        donors.update(zip(users,users[1:]+users[:1]))
    return donors


def choose(candidates, evidence, boundary_evidence, feature):
    # Candidate objects supplied here exclude target labels.
    qualified = [c for c in candidates if evidence[c['item']][feature] >= 2
                 and evidence[c['item']][feature] > boundary_evidence[feature]]
    if not qualified:
        return None
    return min(qualified, key=lambda c: (-evidence[c['item']][feature],c['semantic_rank'],c['item']))['item']


def utility(rank):
    rank = np.asarray(rank)
    return np.where(rank <= 10, 1 / np.log2(rank + 1), 0.)


def compare(new, old):
    new,old = np.asarray(new),np.asarray(old)
    delta = utility(new)-utility(old)
    rng = np.random.default_rng(20260924); boot=[]
    for start in range(0,2000,100):
        ix = rng.integers(0,len(delta),size=(100,len(delta)))
        boot.extend(delta[ix].mean(1).tolist())
    return dict(users=len(new), gained10=int(((new<=10)&(old>10)).sum()), lost10=int(((old<=10)&(new>10)).sum()),
                new_N10=float(utility(new).mean()), old_N10=float(utility(old).mean()),
                delta_N10=float(delta.mean()), delta_N10_ci95=np.quantile(boot,[.025,.975]).tolist())


def standardized_difference(counts):
    # counts[..., stratum, evidence_present, all_candidates_or_targets]
    n0,n1 = counts[...,0,0],counts[...,1,0]
    y0,y1 = counts[...,0,1],counts[...,1,1]
    valid = (n0>0)&(n1>0)
    weight = np.where(valid,n0+n1,0.)
    total = weight.sum(-1)
    difference = y1/np.maximum(n1,1)-y0/np.maximum(n0,1)
    effect = (weight*difference).sum(-1)/np.maximum(total,1)
    return effect,total


def associations(records, users, edges):
    ix = {u:i for i,u in enumerate(users)}
    output = {}
    for evidence_type in ('real','shuffled'):
        output[evidence_type] = {}
        for name,(feature,threshold) in FEATURES.items():
            cells = np.zeros((len(users),15,2,2),np.float64)
            for r in records:
                rb = 0 if r['semantic_rank']==1 else 1 if r['semantic_rank']<=3 else 2
                sb = int(np.searchsorted(edges,r['semantic_score'],side='right'))
                present = int(r[evidence_type][feature]>=threshold)
                cells[ix[r['user_id']],rb*5+sb,present,0] += 1
                cells[ix[r['user_id']],rb*5+sb,present,1] += r['target_hit']
            totals = cells.sum(0)
            effect,covered = standardized_difference(totals)
            out = dict(with_evidence=dict(candidates=int(totals[:,1,0].sum()),targets=int(totals[:,1,1].sum())),
                       without_evidence=dict(candidates=int(totals[:,0,0].sum()),targets=int(totals[:,0,1].sum())))
            for group in ('with_evidence','without_evidence'):
                a=out[group];a['target_rate']=a['targets']/a['candidates'] if a['candidates'] else None
            out.update(score_matched_rate_difference=float(effect) if covered else None,
                       matched_candidate_coverage=float(covered/max(totals[...,0].sum(),1)))
            boot=[]; rng=np.random.default_rng(20260924)
            if covered:
                flat = cells.reshape(len(users),-1)
                for start in range(0,1000,100):
                    # Multinomial counts implement resampling whole users.
                    multipliers = rng.multinomial(len(users),np.full(len(users),1/len(users)),size=100)
                    b = (multipliers@flat).reshape(100,15,2,2)
                    values,den = standardized_difference(b)
                    boot.extend(values[den>0].tolist())
            out['bootstrap_valid_replicates']=len(boot)
            out['score_matched_ci95']=np.quantile(boot,[.025,.975]).tolist() if boot else None
            out['strata']=[dict(rank_group=k//5,score_bin=k%5,
                without_candidates=int(totals[k,0,0]),without_targets=int(totals[k,0,1]),
                with_candidates=int(totals[k,1,0]),with_targets=int(totals[k,1,1])) for k in range(15)]
            output[evidence_type][name]=out
    return output


def summarize(users, records):
    training_scores=[r['semantic_score'] for r in records if r['role']=='ranker_fit']
    edges=np.quantile(training_scores,[.2,.4,.6,.8])
    out={'text_score_edges_from_ranker_fit':edges.tolist()}
    for role in ('ranker_fit','development'):
        rows=[r for r in users if r['role']==role]
        candidates=[r for r in records if r['role']==role]
        names=list(rows[0]['ranks'])
        ranks={n:np.asarray([r['ranks'][n] for r in rows]) for n in names}
        out[role]=dict(users=len(rows), dissent_candidates=len(candidates),
            dissent_targets=sum(r['target_hit'] for r in candidates),
            users_with_dissent=sum(r['dissent_count']>0 for r in rows),
            changed_history_users=sum(r['donor_user']!=r['user_id'] for r in rows),
            older_exact_item_repeat_candidates=sum(r['real']['older_exact_item_repeat'] for r in candidates),
            rules={}, associations=associations(candidates,[r['user_id'] for r in rows],edges))
        for name in names:
            out[role]['rules'][name]=dict(versus_original=compare(ranks[name],ranks['original']),
                versus_text_first=compare(ranks[name],ranks['text_first']),
                actions=sum(r['choices'][name]!=r['original10'] for r in rows))
    return out


def run_domain(domain, manifest):
    folder=SOURCE/domain
    paths=[folder/'catalog.json',folder/'samples.json',PRIOR/(domain+'_candidates.csv'),PRIOR/'per_user.csv']
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    manifest.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    catalog=json.loads((folder/'catalog.json').read_text())
    all_samples=json.loads((folder/'samples.json').read_text())
    samples=[s for s in all_samples if s['role'] in ('ranker_fit','development')]
    by_user={s['user_id']:s for s in samples};assert len(by_user)==len(samples)
    donors=donor_mapping(samples)
    engine=CatalogEvidence(catalog)
    with (PRIOR/'per_user.csv').open() as f:
        prior={r['user_id']:r for r in csv.DictReader(f) if r['domain']==domain}
    grouped=defaultdict(list)
    with (PRIOR/(domain+'_candidates.csv')).open() as f:
        for r in csv.DictReader(f):
            r['item']=int(r['item']);r['target_hit']=int(r['target_hit'])
            for key in ('semantic_rank','LIME_rank','SASRec_rank','ItemCF_rank'):
                r[key]=int(r[key])
            for key in ('semantic_score','LIME_score','SASRec_score','ItemCF_score'):
                r[key]=float(r[key])
            grouped[r['user_id']].append(r)
    assert set(grouped)==set(by_user)==set(prior)
    user_rows,records,cases=[],[],[]
    for pos,user in enumerate(sorted(by_user)):
        sample=by_user[user]
        target=sample['target'];assert int(prior[user]['target'])==target and prior[user]['role']==sample['role']
        cs=grouped[user]
        top=sorted([r for r in cs if r['LIME_rank']<=10],key=lambda r:r['LIME_rank'])
        assert [r['LIME_rank'] for r in top]==list(range(1,11))
        top_ids=[r['item'] for r in top]; boundary=top_ids[-1]
        dissent=[r for r in cs if r['semantic_rank']<=10 and r['LIME_rank']>10]
        history=engine.history(sample['full_history']);other=engine.history(by_user[donors[user]]['full_history'])
        assert len(sample['full_history'])==len(by_user[donors[user]]['full_history'])
        assert sample['role']==by_user[donors[user]]['role']
        eval_items=[r['item'] for r in dissent]+[boundary]
        real={i:engine.evidence(i,history) for i in eval_items}
        shuffled={i:engine.evidence(i,other) for i in eval_items}
        choices={'original':boundary,
                 'text_first':min(dissent,key=lambda r:(r['semantic_rank'],r['item']))['item'] if dissent else boundary}
        # Deliberately pass only inference-visible identity and rank to selector.
        label_free=[dict(item=r['item'],semantic_rank=r['semantic_rank']) for r in dissent]
        for name,feature in RULE_FEATURES.items():
            choices[name]=choose(label_free,real,real[boundary],feature) or boundary
            choices['shuffled_'+name]=choose(label_free,shuffled,shuffled[boundary],feature) or boundary
        ranks={}
        for name,item in choices.items():
            selected=top_ids[:9]+[item]
            assert len(set(selected))==10 and not(set(selected)&set(sample['history'][-20:]))
            ranks[name]=selected.index(target)+1 if target in selected else ABSENT
        assert ranks['original']==(int(prior[user]['LIME']) if int(prior[user]['LIME'])<=10 else ABSENT)
        if dissent:
            # Existing 21.7 reserve rule agrees whenever a dissent remains in text top10.
            assert choices['text_first']==int(prior[user]['reserve_semantic_item'])
        ur=dict(domain=domain,user_id=user,role=sample['role'],target=target,original10=boundary,
                history_length=len(sample['full_history']),donor_user=donors[user],dissent_count=len(dissent),
                choices=choices,ranks=ranks)
        user_rows.append(ur)
        for r in dissent:
            assert r['target_hit']==int(r['item']==target)
            records.append(dict(r,real=real[r['item']],shuffled=shuffled[r['item']],
                                boundary_real=real[boundary],boundary_shuffled=shuffled[boundary],
                                donor_user=donors[user]))
        if (pos+1)%512==0:
            write(OUT/'status.json',dict(state='RUNNING',domain=domain,stage='extract',completed=pos+1,total=len(samples),pid=os.getpid()))
            print(domain,'evidence',pos+1,'/',len(samples),flush=True)
        # Both the known illustration and all changed development decisions are retained.
        if user=='A18G3K8RVJD28' or (sample['role']=='development' and any(c!=boundary for n,c in choices.items() if n not in ('original','text_first'))):
            item_details={str(i):dict(item=i,title=engine.parsed[i].get('title'),real=real[i],shuffled=shuffled[i]) for i in eval_items}
            cases.append(dict(ur,history=[dict(item=h,title=engine.parsed[h].get('title'),brand=engine.brands[h]) for h in sample['full_history']],items=item_details))
    write(OUT/(domain+'_users.json'),user_rows)
    write(OUT/(domain+'_cases.json'),cases)
    flattened=[]
    for r in records:
        row={k:v for k,v in r.items() if not isinstance(v,dict)}
        for prefix in ('real','shuffled','boundary_real','boundary_shuffled'):
            for k,v in r[prefix].items():
                row[prefix+'_'+k]=json.dumps(v) if isinstance(v,list) else v
        flattened.append(row)
    with (OUT/(domain+'_evidence.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    write(OUT/'status.json',dict(state='RUNNING',domain=domain,stage='clustered_bootstrap',pid=os.getpid()))
    result=summarize(user_rows,records)
    result['metadata']=dict(catalog_items=len(catalog['items'])-1,
        missing_brand_items=sum(not b for b in engine.brands[1:]),
        missing_title_items=sum(not p.get('title') or p.get('title') in ('na','unknown') for p in engine.parsed[1:]),
        rarity_min_document_frequency=2,rarity_max_document_frequency=.01*(len(catalog['items'])-1))
    for p in paths:
        assert (p.stat().st_size,p.stat().st_mtime_ns)==before[str(p)]
    result['verification']=dict(all_users_retained=True,all_lists_exactly_10=True,no_recent_history_recommended=True,
         first9_unchanged=True,original_ranks_match=True,text_first_matches_previous_when_dissent=True,
         donor_role_and_history_length_match=True,sources_unmodified=True,
         target_free_evidence_and_selection=True)
    write(OUT/(domain+'_summary.json'),result)
    return result


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    (OUT/'analysis_source.py').write_text(Path(__file__).read_text())
    write(OUT/'status.json',dict(state='RUNNING',stage='start',pid=os.getpid()))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in (PROTOCOL,Path(__file__))}
    try:
        domains={}
        for domain in ('Toys','Beauty'):
            domains[domain]=run_domain(domain,manifest)
            print(domain,'complete',flush=True)
        summary=dict(state='COMPLETED',domains=domains,seconds=time.monotonic()-start,
            gpu_inferences=0,optimizer_steps=0,statistical_model_fits=0,test_read=False,
            exploratory=True,independent_confirmation=False)
        write(OUT/'summary.json',summary);write(OUT/'manifest.json',manifest)
        write(OUT/'status.json',{k:v for k,v in summary.items() if k!='domains'})
        print(json.dumps({k:v for k,v in summary.items() if k!='domains'}),flush=True)
    except BaseException as error:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        write(OUT/'status.json',dict(state='FAILED',error=repr(error),pid=os.getpid()))
        raise


if __name__=='__main__':
    main()
