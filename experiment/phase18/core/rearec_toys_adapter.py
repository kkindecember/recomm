"""Toys data contract with the frozen ReaRec Beauty model and checkpoint code."""
import json

from experiment.phase18.core.rearec_adapter import (
    ROOT, ReaRec, collate, digest, isolated_rng, load_checkpoint, restore_rng,
    rng_state, save_checkpoint, seed_all, source_layers, write_json,
)


def load_inputs(config):
    review = json.loads((ROOT / config['input_review']).read_text())
    if config['dataset'] != 'Toys' or review['dataset'] != 'Toys':
        raise ValueError('This screen requires Toys inputs')
    if config['max_history'] != 20:
        raise ValueError('The frozen Toys history contract requires max_history=20')
    required = {config['prepared'] + '/' + name for name in ('train.jsonl', 'validation.jsonl', 'catalog.json')}
    required.update((config['cohort'], config['references'], config['reference_predictions']))
    if not required <= review['files'].keys():
        raise ValueError('Input paths must be frozen')
    for path, sha in review['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'Frozen input changed: {path}')
    directory = ROOT / config['prepared']
    rows = {split: [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines()]
            for split in ('train', 'validation')}
    names = json.loads((directory / 'catalog.json').read_text())['items']
    by_user = {r['user_id']: r for r in rows['validation']}
    if (len(rows['train']), len(rows['validation']), len(by_user), len(names)) != (109361, 19412, 19412, 11925):
        raise ValueError('Toys population changed')
    if names[1:] != sorted(set(names[1:])):
        raise ValueError('Catalog identity mismatch')
    for split in rows.values():
        for row in split:
            if row['user_id'] not in by_user or not 1 <= len(row['history']) <= config['max_history']:
                raise ValueError('Invalid history or user')
            if any(type(i) is not int or not 0 < i < len(names) for i in row['history'] + [row['target']]):
                raise ValueError('Invalid item ID')
    cohort = json.loads((ROOT / config['cohort']).read_text())['user_ids']
    if len(cohort) != 2000 or len(set(cohort)) != 2000 or not set(cohort) <= by_user.keys():
        raise ValueError('Trend cohort mismatch')
    rows['trend'] = [by_user[u] for u in cohort]
    return rows, names
