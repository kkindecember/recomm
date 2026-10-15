"""Catalogue-only, traceable syntactic proposals for named-series and product relations."""
from collections import Counter, defaultdict
import hashlib
import re

from analyze_history_evidence import STOP, fields, tokens

GENERIC = STOP | set('series collection professional products product item items use used all quality best '
    'special deluxe original edition system formula beauty hair skin care toy toys game games'.split())
ORDINALS = set('one two three four five six seven eight nine ten first second third fourth fifth sixth seventh eighth ninth tenth'.split())
CUES = ('compatible with','designed for use with','works with','for use with','requires the',
        'expansion for','expansion to','refill for','replacement for')
NEGATION = {'not','no','never','incompatible','except'}


def informative(ts):
    return {t for t in ts if len(t)>=3 and not t.isdigit() and t not in GENERIC}


def series_key(title):
    ts=tokens(title)
    for i,t in enumerate(ts):
        if t not in ('series','collection'):continue
        key=ts[:i+1]
        if t=='series' and i+1<len(ts) and (ts[i+1].isdigit() or ts[i+1] in ORDINALS):key.append(ts[i+1])
        if len(key)<=12 and len(informative(ts[:i]))>=2:return ' '.join(key)
        return ''
    return ''


def hash_key(value):return hashlib.sha256(value.encode()).hexdigest()


class ExplicitRelations:
    def __init__(self,catalog,domain):
        self.catalog=catalog;self.domain=domain;self.parsed=[fields(p) for p in catalog['passages']]
        titles=defaultdict(list);groups=defaultdict(list)
        for i,f in enumerate(self.parsed[1:],1):
            ts=tuple(tokens(f.get('title','')))
            if 2<=len(ts)<=12 and len(informative(ts))>=2:titles[ts].append(i)
            key=series_key(f.get('title',''))
            if key:groups[key].append(i)
        self.ambiguous_titles={k:v for k,v in titles.items() if len(v)!=1}
        self.title_index={k:v[0] for k,v in titles.items() if len(v)==1}
        by_first=defaultdict(list)
        for title,item in self.title_index.items():by_first[title[0]].append((title,item))
        for values in by_first.values():values.sort(key=lambda p:(-len(p[0]),p[1]))
        self.groups={k:v for k,v in sorted(groups.items()) if len(v)>=2}
        self.series_neighbors=defaultdict(set)
        for key,items in self.groups.items():
            for i in items:self.series_neighbors[i].update(set(items)-{i})
        evidence=defaultdict(list);cue_counts=Counter();rejected_negative=0
        pattern=re.compile(r'\b(?:'+ '|'.join(re.escape(c) for c in CUES)+r')\b',re.I)
        for a,f in enumerate(self.parsed[1:],1):
            for field in ('title','description'):
                raw=f.get(field,'')
                for sentence in re.split(r'[.!?;\n]',raw):
                    for match in pattern.finditer(sentence):
                        cue=match.group().lower();cue_counts[cue]+=1
                        before=tokens(sentence[:match.start()])[-6:]
                        if set(before)&NEGATION:rejected_negative+=1;continue
                        tail=tokens(sentence[match.end():])[:40];found=[]
                        for offset,token in enumerate(tail):
                            if set(tail[:offset])&NEGATION:continue
                            for target,b in by_first.get(token,[]):
                                if a!=b and tuple(tail[offset:offset+len(target)])==target:
                                    found.append((offset,offset+len(target),b,target))
                        for start,end,b,target in found:
                            if any(s<=start and end<=e and (s<start or end<e) for s,e,_,_ in found):continue
                            card=dict(method='explicit_full_title',field=field,cue=cue,source_sentence=sentence.strip(),
                                      matched_title=' '.join(target),tail_start=start,tail_end=end)
                            if card not in evidence[(a,b)]:evidence[(a,b)].append(card)
            ts=tokens(f.get('title',''))
            if 'expansion' in ts:
                prefix=ts[:ts.index('expansion')]
                while prefix and (prefix[-1] in ORDINALS or prefix[-1].isdigit()):prefix.pop()
                b=self.title_index.get(tuple(prefix))
                if b and b!=a:
                    evidence[(a,b)].append(dict(method='title_expansion',field='title',cue='expansion',
                        source_sentence=f['title'],matched_title=' '.join(prefix)))
        self.edges=[dict(source=a,target=b,evidence=e) for (a,b),e in sorted(evidence.items())]
        self.forward=defaultdict(set);self.reverse=defaultdict(set)
        for edge in self.edges:
            a,b=edge['source'],edge['target'];self.forward[a].add(b);self.reverse[b].add(a)
        self.audit_counts=dict(cue_occurrences=dict(cue_counts),negated_cues_rejected=rejected_negative,
            unique_short_title_items=len(self.title_index),ambiguous_short_title_strings=len(self.ambiguous_titles))

    def card(self,item):
        f=self.parsed[item]
        return dict(item=item,asin=self.catalog['items'][item],title=f.get('title',''),
                    category=f.get('categories',''),brand=f.get('brand',''))

    def support(self,item,history,kind):
        mapping={'series':self.series_neighbors,'explicit':self.forward,'reverse':self.reverse}[kind]
        return sorted(set(history)&mapping.get(int(item),set()))

    def export(self):
        series=[]
        for key,items in self.groups.items():
            uid=self.domain+'|series|'+key
            chosen=sorted(items,key=lambda i:hash_key(uid+'|'+str(i)))[:5]
            series.append(dict(audit_id=hash_key(uid)[:16],kind='series',domain=self.domain,label=key,
                               group_size=len(items),items=[self.card(i) for i in chosen]))
        explicit=[]
        for edge in self.edges:
            uid=f'{self.domain}|explicit|{edge["source"]}|{edge["target"]}'
            explicit.append(dict(audit_id=hash_key(uid)[:16],kind='explicit',domain=self.domain,
                source=self.card(edge['source']),target=self.card(edge['target']),evidence=edge['evidence']))
        sample=sorted(series,key=lambda r:r['audit_id'])[:32]+sorted(explicit,key=lambda r:r['audit_id'])[:32]
        return dict(domain=self.domain,catalog_items=len(self.parsed)-1,series_groups=self.groups,explicit_edges=self.edges,
            coverage=dict(series_groups=len(self.groups),series_items=len(self.series_neighbors),explicit_edges=len(self.edges),
                explicit_source_items=len(self.forward),explicit_target_items=len(self.reverse)),
            extraction_audit=self.audit_counts,blind_audit_cards=sample)
