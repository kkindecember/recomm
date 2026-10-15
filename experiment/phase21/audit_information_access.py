"""Target-free inventory of the existing GRAM inputs; no model inference."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT / 'artifacts/phase21/capability_inventory_20260922'
TOKENIZER = ROOT / '.cache/huggingface/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / 'GRAM/src'))
    from processor import CollatorGRAM
    torch.set_num_threads(2)
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'input_audit.json'
    assert not destination.exists(), 'Keep the original inventory receipt'
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    collator = CollatorGRAM(tokenizer, SimpleNamespace(item_prompt_max_len=128,
        target_max_len=32, max_his=20, item_id_type='split',
        hierarchical_id_type='unused_for_input_encoding'))
    source_files = [Path(__file__), ROOT / 'GRAM/src/processor/Collator.py',
        ROOT / 'experiment/phase20/lime_clean_upstream_core.py',
        ROOT / 'experiment/phase20/lime_comparison_core.py',
        ROOT / 'experiment/phase20/lime_clean_benefit.py']
    source_files += [TOKENIZER / name for name in ('tokenizer.json', 'tokenizer_config.json', 'spiece.model')]
    results = {}
    for domain in ('Toys', 'Beauty'):
        folder = SOURCE / domain
        source_files += [folder / 'samples.json', folder / 'catalog.json', folder / 'data_config.json']
        samples = json.loads((folder / 'samples.json').read_text())
        catalog = json.loads((folder / 'catalog.json').read_text())
        histories = {}
        for role in ('ranker_fit', 'development'):
            lengths = np.array([len(row['full_history']) for row in samples if row['role'] == role])
            histories[role] = dict(users=len(lengths), median=float(np.median(lengths)),
                p90=float(np.quantile(lengths, .9)), over20=int((lengths > 20).sum()),
                over50=int((lengths > 50).sum()))
        ids = sorted(range(1, len(catalog['items'])), key=lambda i: hashlib.sha256(
            f'20260922|stage21-input-audit|{domain}|{i}'.encode()).digest())[:128]
        counts = Counter()
        for item in ids:
            passage = catalog['passages'][item]
            prompt = 'What would user purchase after ' + catalog['lexical_ids'][item] + ' ?'
            actual, mask = collator.encode_texts_split([[prompt, passage]], tokenizer)
            actual = actual[0, 1][mask[0, 1]].tolist()
            raw = tokenizer(passage, max_length=999, truncation=True)['input_ids']
            filtered = [token for token in raw if token not in (1820, 9175)]
            expected = filtered[:128]
            if 1 not in expected:
                expected[-1] = 1
            assert expected == actual, (domain, item, 'collator parity')
            decoded = tokenizer.decode(actual, skip_special_tokens=True).lower()
            counts['truncated_after_separator_filter'] += int(len(filtered) > 128)
            for marker in ('title:', 'categories:', 'brand:', 'description:', 'similar items:'):
                counts[f'raw_has_{marker}'] += int(marker in passage.lower())
                counts[f'encoded_has_{marker}'] += int(marker in decoded)
            if time.monotonic() - started > 300:
                raise TimeoutError('Five-minute read-only input audit budget')
        results[domain] = dict(histories=histories,
            catalog_items=len(catalog['items'])-1,
            raw_passages_with_collaborative_hint=sum('similar items:' in p for p in catalog['passages'][1:]),
            prompt_sample=dict(items=len(ids), item_ids=ids, counts=dict(counts),
                exact_collator_parity=True,
                scope='Target-free catalog sample; marker presence is not field completeness or information utility.'))
        print(json.dumps(dict(domain=domain, histories=histories, prompt_counts=dict(counts))), flush=True)
    result = dict(state='COMPLETED', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        domains=results, seconds=time.monotonic()-started, gpu_used=False,
        new_training=False, new_generation=False, target_values_accessed=False,
        test_files_read=False, note='Existing sample JSON contains target fields; this audit never accesses their values.',
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in source_files})
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
