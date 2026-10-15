"""21.15 catalogue-only, conservative name normalization and alias lookup."""
from collections import Counter, defaultdict
import re

from analyze_history_evidence import fields, tokens
from explicit_relation_core import CUES, GENERIC, NEGATION, ORDINALS, hash_key, informative

PREFIXES = ('mcfarlane toys', 'dc collectibles', 'dc direct', 'diamond select toys', 'neca')
EXCLUDED = {'the new york doll collection', 'melissa doug sticker collection',
            'kleancolor nail lacquer mini collection', 'john frieda collection'}
UNITS = set('inch inches in cm mm oz ounce ounces ml g gram grams lb lbs pound pounds ft foot feet'.split())
SCOPES = {'tv', 'comic', 'movie'}


def metadata(title):
    ts = tokens(title)
    batches = []
    specifications = []
    for i, t in enumerate(ts[:-1]):
        if t == 'series' and (ts[i+1].isdigit() or ts[i+1] in ORDINALS):
            # Decimal tokenization can yield 1 5 inch: scan the numeric run.
            end = i + 2
            while end < len(ts) and ts[end].isdigit():
                end += 1
            if end < len(ts) and ts[end] in UNITS:
                specifications.append(' '.join(ts[i+1:end+1]))
            else:
                batches.append(ts[i+1])
    return dict(scope=sorted(set(ts) & SCOPES), batch=batches, size_after_series=specifications)


def normalize(ts):
    ts = list(ts)
    for prefix in PREFIXES:
        p = prefix.split()
        if ts[:len(p)] == p:
            ts = ts[len(p):]
            break
    for phrase in (['nail', 'polish'], ['nail', 'lacquer']):
        i = 0
        while i < len(ts)-1:
            if ts[i:i+2] == phrase:
                del ts[i:i+2]
            else:
                i += 1
    return [t for t in ts if t not in {'the', 'amp', 'series', 'collection'} | SCOPES]


def contains(sequence, part):
    return any(sequence[i:i+len(part)] == part for i in range(len(sequence)-len(part)+1))


class EntityRelations:
    def __init__(self, catalog, old, domain):
        self.catalog = catalog
        self.domain = domain
        self.parsed = [fields(p) for p in catalog['passages']]
        self.seeds = defaultdict(list)
        self.rejected = []
        for label, ids in old['series_groups'].items():
            ts = tokens(label)
            stop = next(i for i, t in enumerate(ts) if t in ('series', 'collection'))
            name = normalize(ts[:stop])
            reason = ('prior_source_uncertain' if label in EXCLUDED else
                      'insufficient_distinctive_tokens' if len(name) < 2 or len(informative(name)) < 2 else '')
            if reason:
                self.rejected.append(dict(label=label, items=ids, reason=reason))
            else:
                self.seeds[' '.join(name)].append(dict(label=label, items=ids))
        self.items = {}
        self.groups = defaultdict(list)
        names = {k: k.split() for k in self.seeds}
        for item, f in enumerate(self.parsed[1:], 1):
            title = f.get('title', '')
            ts = normalize(tokens(title))
            matches = sorted(k for k, p in names.items() if contains(ts, p))
            chosen = ''
            if matches:
                longest = sorted(matches, key=lambda k: (-len(names[k]), k))[0]
                if all(contains(names[longest], names[k]) for k in matches):
                    chosen = longest
            is_seed = chosen and any(item in s['items'] for s in self.seeds[chosen])
            status = ('seed' if is_seed else 'alias_added') if chosen else ('ambiguous' if matches else 'unmatched')
            self.items[item] = dict(item=item, family=chosen, status=status, matches=matches,
                                    **metadata(title))
            if chosen:
                self.groups[chosen].append(item)
        index = defaultdict(set)
        for item, f in enumerate(self.parsed[1:], 1):
            ts = tokens(f.get('title', ''))
            for i in range(len(ts)):
                for n in range(1, min(6, len(ts)-i)+1):
                    part = ts[i:i+n]
                    if informative(part) and (n > 1 or len(part[0]) >= 5):
                        index[''.join(part)].add(item)
        self.lookups = []
        pattern = re.compile(r'\b(?:'+'|'.join(re.escape(c) for c in CUES)+r')\b', re.I)
        for item, f in enumerate(self.parsed[1:], 1):
            for field in ('title', 'description'):
                for sentence in re.split(r'[.!?;\n]', f.get(field, '')):
                    for match in pattern.finditer(sentence):
                        tail = tokens(sentence[match.end():])
                        while tail and tail[0] in {'the', 'all', 'any'}:
                            tail.pop(0)
                        tail = tail[:8]
                        negated = bool(set(tokens(sentence[:match.start()])[-6:]) & NEGATION)
                        status, alias, ids = ('negated' if negated else 'unmatched'), '', []
                        if not negated:
                            for n in range(len(tail), 0, -1):
                                part = tail[:n]
                                if set(part) & NEGATION or not informative(part) or (n == 1 and len(part[0]) < 5):
                                    continue
                                possible = sorted(index.get(''.join(part), set()) - {item})
                                if possible:
                                    alias, ids = ' '.join(part), possible
                                    status = ('too_broad' if len(ids) > .02*(len(self.parsed)-1) else
                                              'single_catalog_match' if len(ids) == 1 else 'multiple_catalog_matches')
                                    break
                        uid = f'{domain}|{item}|{field}|{sentence}|{match.start()}'
                        self.lookups.append(dict(audit_id=hash_key(uid)[:16], source=item, field=field,
                            cue=match.group().lower(), source_span=sentence.strip(), object_prefix=' '.join(tail),
                            alias=alias, possible_items=ids, status=status))

    def card(self, item):
        f = self.parsed[item]
        return dict(item=item, title=f.get('title', ''), brand=f.get('brand', ''),
                    **{k: v for k, v in self.items[item].items() if k != 'item'})

    def export(self):
        audits = []
        for item, r in self.items.items():
            if r['status'] == 'alias_added':
                audits.append(dict(audit_id=hash_key(f'{self.domain}|family|{item}')[:16], kind='family_alias',
                    domain=self.domain, item=self.card(item), seeds=[dict(label=s['label'],
                    titles=[self.card(i) for i in s['items']]) for s in self.seeds[r['family']]]))
        family_audit = sorted(audits, key=lambda r: r['audit_id'])[:32]
        # Round-robin SHA samples cover every available status, max 16 in total.
        pools = {s: sorted([r for r in self.lookups if r['status'] == s], key=lambda r: r['audit_id'])
                 for s in sorted(set(r['status'] for r in self.lookups))}
        compatibility_audit = []
        for i in range(16):
            for pool in pools.values():
                if i < len(pool) and len(compatibility_audit) < 16:
                    r = dict(pool[i]);r['source_title'] = self.parsed[r['source']].get('title', '')
                    r['matched_titles'] = [dict(item=j, title=self.parsed[j].get('title', '')) for j in r['possible_items']]
                    compatibility_audit.append(r)
        return dict(domain=self.domain, seeds=dict(self.seeds), rejected_seeds=self.rejected,
            items=list(self.items.values()), families=dict(self.groups), compatibility_lookups=self.lookups,
            coverage=dict(catalog_items=len(self.items), families=len(self.groups),
                statuses=dict(Counter(r['status'] for r in self.items.values())),
                compatibility_statuses=dict(Counter(r['status'] for r in self.lookups))),
            family_audit=family_audit, compatibility_audit=compatibility_audit)
