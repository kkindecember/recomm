"""Target-independent packing of the existing catalog; matched description control."""
import hashlib
import re
import numpy as np

ARMS = ('original', 'retained', 'shuffled')
EMPTY = {'', 'na', 'n/a', 'none', 'nan', 'unknown'}


def parse_fields(passage):
    matches = list(re.finditer(r'(?:^|;\s*)(item|similar items|title|categories|brand|description|price|salesrank)\s*:', passage))
    result = {}
    for j, match in enumerate(matches):
        stop = matches[j+1].start() if j+1 < len(matches) else len(passage)
        result.setdefault(match.group(1), passage[match.end():stop].strip())
    return result


def token_ids(tokenizer, text):
    return [i for i in tokenizer(text, add_special_tokens=False, truncation=True, max_length=999)['input_ids']
            if i not in (0, 1, 1820, 9175)]


def native_tokens(tokenizer, passage):
    values = [i for i in tokenizer(passage, truncation=True, max_length=999)['input_ids'] if i not in (1820, 9175)][:128]
    if 1 not in values:
        values[-1] = 1
    return values


def pack_catalog(catalog, tokenizer, domain):
    size = len(catalog['items'])
    banks = {arm: np.zeros((size, 128), dtype=np.int64) for arm in ARMS}
    lengths = {arm: np.zeros(size, dtype=np.int64) for arm in ARMS}
    slices = {}; groups = {}; decoded = []
    original_marker = 0; usable_descriptions = 0
    for i in range(1, size):
        passage = catalog['passages'][i]
        original = native_tokens(tokenizer, passage)
        banks['original'][i, :len(original)] = original
        lengths['original'][i] = len(original)
        fields = parse_fields(passage)
        assert fields['item'] == catalog['lexical_ids'][i]
        neighbors = fields['similar items'].split(', ')
        assert len(neighbors) == {'Toys': 5, 'Beauty': 10}[domain]
        prefix = token_ids(tokenizer, f"item: {fields['item']}; similar items: {', '.join(neighbors[:2])};")
        for name, quota in (('title', 24), ('categories', 12), ('brand', 8)):
            prefix += token_ids(tokenizer, f" {name}: {fields.get(name, 'na')};")[:quota]
        prefix += token_ids(tokenizer, ' description:')
        capacity = 127-len(prefix)
        assert capacity >= 16, (domain, i, len(prefix))
        text = fields.get('description', 'na')
        content = token_ids(tokenizer, ' '+text)[:capacity]
        if text.strip().lower() not in EMPTY and content:
            groups.setdefault(len(content), []).append(i)
            usable_descriptions += 1
        seq = prefix + content + [1]
        assert len(seq) <= 128
        banks['retained'][i, :len(seq)] = seq
        lengths['retained'][i] = len(seq)
        slices[i] = (len(prefix), len(prefix)+len(content))
        original_marker += int('description:' in tokenizer.decode(original, skip_special_tokens=True).lower())
        if i <= 3:
            decoded.append(dict(item_id=i, original=tokenizer.decode(original, skip_special_tokens=True),
                                retained=tokenizer.decode(seq, skip_special_tokens=True)))
    banks['shuffled'][:] = banks['retained']
    lengths['shuffled'][:] = lengths['retained']
    donors = np.arange(size, dtype=np.int64)
    for width, items in groups.items():
        items.sort(key=lambda i: hashlib.sha256(f'20260923|description-placebo|{domain}|{catalog["items"][i]}'.encode()).digest())
        if len(items) < 2:
            continue
        for receiver, donor in zip(items, items[1:]+items[:1]):
            a, b = slices[receiver]; c, d = slices[donor]
            assert b-a == d-c == width
            banks['shuffled'][receiver, a:b] = banks['retained'][donor, c:d]
            donors[receiver] = donor
    changed = 0
    for i in range(1, size):
        a, b = slices[i]
        assert np.array_equal(banks['retained'][i, :a], banks['shuffled'][i, :a])
        assert np.array_equal(banks['retained'][i, b:], banks['shuffled'][i, b:])
        changed += int(not np.array_equal(banks['retained'][i], banks['shuffled'][i]))
    assert np.array_equal(lengths['retained'], lengths['shuffled'])
    audit = dict(items=size-1, original_description_markers=original_marker,
                 usable_descriptions=usable_descriptions, placebo_changed_items=changed,
                 minimum_description_capacity=16, maximum_passage_tokens=128,
                 mean_tokens={a: float(lengths[a][1:].mean()) for a in ARMS}, examples=decoded,
                 description_offsets={str(i): list(v) for i, v in slices.items()})
    return banks, lengths, donors, audit


def install_bank(original_collator, catalog, bank, lengths):
    import torch
    from experiment.phase20.lime_parallel_core import CachedCollator
    cached = CachedCollator(original_collator)
    for i, passage in enumerate(catalog['passages'][1:], 1):
        n = int(lengths[i])
        cached.passages[passage] = (torch.from_numpy(bank[i, :n].copy()), torch.ones(n, dtype=torch.bool))
    return cached
