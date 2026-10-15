"""21.16 title identity versus mention, with explicit version fields and abstention."""
from collections import Counter, defaultdict
import hashlib
from analyze_history_evidence import fields, tokens
from entity_relation_core import normalize, contains, UNITS

BOUNDARIES=set('for with fits compatible works replacement refill compare versus vs designed suitable including'.split())
SCOPES={'tv','comic','movie'}
SEASONS=set('spring summer fall autumn winter holiday'.split())
NUMBER_WORDS={w:str(i) for i,pair in enumerate(zip(
    'one two three four five six seven eight nine ten'.split(),
    'first second third fourth fifth sixth seventh eighth ninth tenth'.split()),1) for w in pair}
DIMENSIONS=('scope','batch','year')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def find_family(ts,names):
    matches=sorted(name for name in names if contains(ts,name.split()))
    if not matches:return '',matches
    longest=sorted(matches,key=lambda n:(-len(n.split()),n))[0]
    return (longest if all(contains(longest.split(),n.split()) for n in matches) else ''),matches


def version_fields(ts):
    evidence={k:[] for k in DIMENSIONS};sizes=[]
    for i,t in enumerate(ts):
        if t in SCOPES:evidence['scope'].append(dict(value=t,span=t,start=i,end=i+1))
        if t=='series' and i+1<len(ts) and (ts[i+1].isdigit() or ts[i+1] in NUMBER_WORDS):
            end=i+2
            while end<len(ts) and ts[end].isdigit():end+=1
            if end<len(ts) and ts[end] in UNITS:sizes.append(' '.join(ts[i:end+1]))
            else:evidence['batch'].append(dict(value=NUMBER_WORDS.get(ts[i+1],str(int(ts[i+1])) if ts[i+1].isdigit() else ''),
                span=' '.join(ts[i:i+2]),start=i,end=i+2))
        if t.isdigit() and len(t)==4 and 1900<=int(t)<=2099:
            adjacent=[j for j in (i-1,i+1) if 0<=j<len(ts) and ts[j] in SEASONS|{'collection'}]
            for j in adjacent:
                lo,hi=min(i,j),max(i,j)+1
                evidence['year'].append(dict(value=t,span=' '.join(ts[lo:hi]),start=lo,end=hi))
    result={}
    for k in DIMENSIONS:
        values=sorted({r['value'] for r in evidence[k]})
        result[k]=dict(value=values[0] if len(values)==1 else '',values=values,
            status='known' if len(values)==1 else 'absent' if not values else 'ambiguous',evidence=evidence[k])
    return result,sizes


class TypedRelations:
    def __init__(self,catalog,previous,domain):
        self.domain=domain;self.names=sorted(previous['seeds']);self.parsed=[fields(p) for p in catalog['passages']]
        self.items={};self.groups=defaultdict(list)
        for item,f in enumerate(self.parsed[1:],1):
            title=f.get('title','');ts=tokens(title);cut=next((i for i,t in enumerate(ts) if t in BOUNDARIES),len(ts))
            subject,tail=ts[:cut],ts[cut:];family,matches=find_family(normalize(subject),self.names)
            _,mentions=find_family(normalize(tail),self.names);versions,sizes=version_fields(subject)
            before=previous['items'][item-1];assert before['item']==item
            r=dict(item=item,title=title,own_family=family,own_matches=matches,mentioned_families=mentions,
                identity_status='known' if family else 'ambiguous' if matches else 'mention_only' if mentions else 'unmatched',
                subject_tokens=subject,boundary_token=ts[cut] if cut<len(ts) else '',boundary_index=cut,
                versions=versions,size_exclusions=sizes,previous_family=before['family'])
            self.items[item]=r
            if family:self.groups[family].append(item)
        self.requirements={family:[k for k in DIMENSIONS if any(self.items[i]['versions'][k]['values'] for i in ids)]
                           for family,ids in self.groups.items()}

    def export(self):
        own=[r for r in self.items.values() if r['own_family']]
        fine=[r for r in own if any(r['versions'][k]['status']=='known' for k in DIMENSIONS)]
        def sample(rows,label):
            cards=[dict(audit_id=digest(f'{self.domain}|{label}|{r["item"]}')[:16],kind=label,record=r) for r in rows]
            return sorted(cards,key=lambda c:c['audit_id'])[:32]
        return dict(domain=self.domain,names=self.names,items=list(self.items.values()),families=dict(self.groups),requirements=self.requirements,
            identity_audit=sample(own,'identity'),version_audit=sample(fine,'version'),
            role_changes=[r for r in self.items.values() if r['previous_family']!=r['own_family']],
            all_version_items=fine,coverage=dict(catalog_items=len(self.items),known_identity=len(own),version_items=len(fine),
                states=dict(Counter(r['identity_status'] for r in self.items.values())),
                explicit_dimensions={k:sum(r['versions'][k]['status']=='known' for r in own) for k in DIMENSIONS},
                role_changes=sum(r['previous_family']!=r['own_family'] for r in self.items.values())))


def comparable_support(records,requirements,a,b,history,mode):
    """No target argument. Unknown is uncomparable, never negative evidence."""
    ra,rb=records[a],records[b];fa,fb=ra['own_family'],rb['own_family']
    if not fa or not fb:return None
    if mode=='parent' or fa!=fb:
        if mode=='fine_only':return None
        ca=sum(h!=a and records[h]['own_family']==fa for h in set(history))
        cb=sum(h!=b and records[h]['own_family']==fb for h in set(history))
        return dict(a_support=ca,b_support=cb,level='parent',dimensions=[])
    dims=requirements[fa]
    if not dims or any(ra['versions'][k]['status']!='known' or rb['versions'][k]['status']!='known' for k in dims):return None
    va=tuple(ra['versions'][k]['value'] for k in dims);vb=tuple(rb['versions'][k]['value'] for k in dims)
    if va==vb:return None
    def count(item,key):
        return sum(h!=item and records[h]['own_family']==fa and
            all(records[h]['versions'][k]['status']=='known' for k in dims) and
            tuple(records[h]['versions'][k]['value'] for k in dims)==key for h in set(history))
    return dict(a_support=count(a,va),b_support=count(b,vb),level='version',dimensions=dims)
