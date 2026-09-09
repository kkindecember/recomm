"""Toys input contract; reuse the unchanged, already running Beauty model core."""
import json

import torch
from torch.nn import functional as F

from experiment.phase18.core.etegrec_adapter import (
    ROOT, CatalogTrie, build_models, collate, digest, generate_items, isolated_rng,
    load_bundle, losses, make_codes, rng_state, save_bundle, scheduler, seed_all,
    set_phase, write_json,
)


def validate_population(records, names, cohort):
    if len(records['train']) != 109361 or len(records['validation']) != 19412 or len(names) != 11925:
        raise ValueError('Toys population changed')
    by_user = {r['user_id']: r for r in records['validation']}
    if len(by_user) != len(records['validation']):
        raise ValueError('Duplicate validation users')
    if len(cohort) != 2000 or len(set(cohort)) != len(cohort) or not set(cohort) <= set(by_user):
        raise ValueError('Toys trend cohort mismatch')
    if names[1:] != sorted(set(names[1:])):
        raise ValueError('Catalog must follow the CF checkpoint sorted raw-item mapping')
    for split in ('train', 'validation'):
        for row in records[split]:
            if row['user_id'] not in by_user or not 1 <= len(row['history']) <= 20:
                raise ValueError('Invalid history or user')
            if any(not isinstance(i, int) or not 0 < i < len(names)
                   for i in row['history'] + [row['target']]):
                raise ValueError('Invalid catalog item')
    return by_user


def load_inputs(config):
    review = json.loads((ROOT / config['input_review']).read_text())
    if config['dataset'] != 'Toys' or review['dataset'] != 'Toys':
        raise ValueError('Toys runner requires Toys inputs')
    required = {config['prepared'] + '/' + name for name in ('train.jsonl', 'validation.jsonl', 'catalog.json')}
    required.update((config['cohort'], config['references']))
    if not required <= review['files'].keys():
        raise ValueError('Unfrozen input paths')
    for path, sha in review['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'Frozen input changed: {path}')
    checkpoint = ROOT / review['collaborative_checkpoint']['path']
    if digest(checkpoint) != review['collaborative_checkpoint']['sha256']:
        raise ValueError('Collaborative checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    raw = saved['model_state_dict']['item_embedding.weight'].float()
    if not torch.isfinite(raw).all() or (raw[1:].norm(dim=-1) == 0).any():
        raise ValueError('Nonfinite or zero collaborative embeddings')
    embeddings = F.normalize(raw, dim=-1)
    embeddings[0].zero_()
    directory = ROOT / config['prepared']
    records = {split: [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines()]
               for split in ('train', 'validation')}
    names = json.loads((directory / 'catalog.json').read_text())['items']
    if embeddings.shape != (len(names), config['model']['semantic_hidden_size']):
        raise ValueError('Embedding/catalog shape mismatch')
    cohort = json.loads((ROOT / config['cohort']).read_text())['user_ids']
    by_user = validate_population(records, names, cohort)
    train_items = sorted({i for r in records['train'] for i in r['history'] + [r['target']]})
    if len(train_items) != 11876:
        raise ValueError('Training-only RQ catalog changed')
    records['trend'] = [by_user[u] for u in cohort]
    return embeddings, records, train_items, names
