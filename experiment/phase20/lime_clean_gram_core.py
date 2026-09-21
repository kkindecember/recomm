"""Training-only inputs, deterministic update windows and resumable state."""
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now
from experiment.phase20.lime_clean_upstream_core import (
    CleanRows, validate_rows, item_inputs, new_gram, runtime_args,
)

T5 = ROOT / '.cache/huggingface/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4'
SPECS = {
    'Toys': dict(rows=105867, users=19412, items=11924, held=3494, neighbors=5,
                 lexical='hierarchy_v1_c32_l5_len32768_split',
                 counts='lime_clean_upstream_toys_20260920_v1/prepare/frequencies.npy'),
    'Beauty': dict(rows=127328, users=22363, items=12101, held=4085, neighbors=10,
                   lexical='hierarchy_v1_c128_l7_len32768_split',
                   counts='lime_clean_beauty_preflight_20260920_v1/frequencies.npy'),
}


def json_rows(path):
    import json
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def line_map(path):
    with Path(path).open() as f:
        return dict(line.rstrip('\n').split(' ', 1) for line in f)


def load_inputs(domain):
    """Check the heldout boundary; return only allowed rows to the trainer."""
    spec = SPECS[domain]
    manifest = {}

    def use(path, expected=None):
        path = Path(path)
        relative = str(path.relative_to(ROOT))
        assert path.name != 'user_sequence.txt' and 'validation.json' not in relative
        assert '/test/' not in relative and 'test.json' not in relative
        digest = sha(path)
        if expected is not None:
            assert digest == expected, f'Changed input: {relative}'
        manifest[relative] = digest
        return path

    design = ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1'
    receipt = read(use(design / 'summary.json'))
    paths = {}
    for name in ('upstream_train_without_terminal.jsonl', 'upstream_sequences_without_terminal.jsonl', 'terminal_holdout.jsonl'):
        p = design / domain / name
        paths[name] = use(p, receipt['output_sha256'][str(p.relative_to(ROOT))])
    rows = json_rows(paths['upstream_train_without_terminal.jsonl'])
    sequences = {r['user_id']: r['history'] for r in json_rows(paths['upstream_sequences_without_terminal.jsonl'])}
    held = json_rows(paths['terminal_holdout.jsonl'])
    assert validate_rows(rows, sequences) == spec['rows']
    assert len(sequences) == spec['users'] and len(held) == spec['held']
    assert len({r['user_id'] for r in held}) == len(held)
    assert all(r['history'] == sequences[r['user_id']][-20:] for r in held)
    # A repeated item is legal; exclusion refers to the last event position.
    by_user = {u: len(s)-1 for u, s in sequences.items()}
    assert sum(by_user.values()) == len(rows)
    items = json_rows(use(ROOT / f'artifacts/phase20/nearest_comparison_20260918/{domain}.items.jsonl'))
    ids = [r['source_item_id'] for r in items]
    assert [r['item_id'] for r in items] == list(range(1, spec['items']+1))
    assert len(set(ids)) == len(ids)
    sas_root = ROOT / f'artifacts/phase20/lime_clean_sasrec_{domain.lower()}_20260920_v1'
    verification = read(use(ROOT / f'artifacts/phase20/lime_clean_sasrec_verification_20260920_v1/{domain}.json'))
    assert verification['state'] == 'VERIFIED' and verification['epochs'] == 200
    for name, digest in verification['input_sha256'].items():
        use(ROOT / name, digest)
    for name, digest in verification['outputs_sha256'].items():
        use(sas_root / name, digest)
    near = np.load(sas_root / 'neighbor_ids.npy', allow_pickle=False)
    assert near.shape == (spec['items'], spec['neighbors'])
    assert near.min() >= 1 and near.max() <= spec['items']
    assert all(len(set(ns)) == len(ns) and i+1 not in ns for i, ns in enumerate(near))
    hint_map = line_map(sas_root / 'similar_item_sasrec.txt')
    assert hint_map == {item: ' '.join(ids[j-1] for j in ns) for item, ns in zip(ids, near)}
    directory = ROOT / 'GRAM/rec_datasets' / domain
    texts = line_map(use(directory / 'item_plain_text.txt'))
    lexical = line_map(use(directory / f'item_generative_indexing_{spec["lexical"]}.txt'))
    inputs = item_inputs(ids, texts, lexical, near)
    counts = np.bincount(np.concatenate(list(sequences.values())), minlength=len(ids)+1)
    np.testing.assert_array_equal(counts, np.load(use(ROOT / 'artifacts/phase20' / spec['counts']), allow_pickle=False))
    users = set(use(ROOT / f'artifacts/phase4/gcdh_p0_splits/{domain}/train_users.txt').read_text().splitlines())
    assert users <= set(sequences)
    c1_rows = [r for r in rows if r['user_id'] in users]
    assert c1_rows
    for name in ('config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'spiece.model'):
        use(T5 / name)
    dataset = CleanRows(rows, ids, inputs, lexical)
    c1_dataset = CleanRows(c1_rows, ids, inputs, lexical)
    summary = dict(domain=domain, training_rows=len(rows), training_users=len(sequences),
                   c1_rows=len(c1_rows), c1_users=len(users), heldout_events=len(held),
                   items=len(ids), neighbors=spec['neighbors'], source_checks=len(manifest),
                   training_interactions=int(counts.sum()), development_labels_read=False, test_read=False)
    return dict(dataset=dataset, c1_dataset=c1_dataset, ids=ids, lexical=lexical,
                counts=counts, manifest=manifest, summary=summary)


def epoch_order(n, seed, epoch):
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order


def update_windows(order, batch_size, accumulation):
    width = batch_size * accumulation
    for start in range(0, len(order), width):
        indices = order[start:start+width]
        yield [indices[j:j+batch_size] for j in range(0, len(indices), batch_size)]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return obj


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        assert torch.cuda.is_available()
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def pack_state(model, optimizer, scheduler, position, manifest, stage, config, **extra):
    assert all(p.grad is None for p in model.parameters()), 'Save only at completed optimizer boundaries'
    return dict(state_dict=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                scheduler=scheduler.state_dict(), rng=rng_state(), position=position.copy(),
                input_sha256=manifest.copy(), stage=stage, training_config=config.copy(),
                created_at=now(), **extra)


def load_state(checkpoint, model, optimizer, scheduler, manifest, stage, config):
    assert checkpoint['input_sha256'] == manifest, 'Checkpoint input/source identity changed'
    assert checkpoint['stage'] == stage and checkpoint['training_config'] == config
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(checkpoint['rng'])
    return checkpoint['position'].copy()


def build_model(data, stage, config, device, parent=None):
    sys.path.insert(0, str(ROOT / 'GRAM/src'))
    model, tokenizer = new_gram(T5)
    model.set_checkpoint(config['encoder_checkpointing'])
    if stage == 'c1':
        assert parent is not None
        model.load_state_dict(parent, strict=True)
        from experiment.phase4.gcdh_p0 import CatalogDualHead
        model = CatalogDualHead(model, tokenizer, data['ids'], data['lexical'],
                                torch.tensor(np.log(data['counts'][1:]+1), dtype=torch.float32))
    from processor import CollatorGRAM
    args = runtime_args()
    args.hierarchical_id_type = SPECS[data['summary']['domain']]['lexical']
    collator = CollatorGRAM(tokenizer, args=args, mode='train')
    return model.to(device), collator


def make_optimizer(model, stage, config, total_updates):
    from transformers import AdamW, get_linear_schedule_with_warmup
    if stage == 'gram':
        groups = [dict(params=[p for n, p in model.named_parameters() if not any(s in n for s in ('bias', 'LayerNorm.weight'))], weight_decay=config['weight_decay']),
                  dict(params=[p for n, p in model.named_parameters() if any(s in n for s in ('bias', 'LayerNorm.weight'))], weight_decay=0.)]
        optimizer = AdamW(groups, lr=config['gram_learning_rate'], eps=1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['c1_learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates*config['warmup_fraction']), total_updates)
    optimizer.zero_grad(set_to_none=True)
    return optimizer, scheduler


def optimizer_update(model, optimizer, scheduler, collator, dataset, batches, stage, config, device, stress=False):
    model.train()
    total = 0.
    shapes = []
    for indices in batches:
        samples = [dataset[i] for i in indices]
        batch = collator(samples)
        for key in ('item_text_ids', 'item_text_masks', 'target_ids'):
            batch[key] = batch[key].to(device)
        if stress:
            import torch.nn.functional as F
            b, p, w = batch['item_text_ids'].shape
            assert b == config['batch_size'] and p <= 21 and w <= 128
            batch['item_text_ids'] = F.pad(batch['item_text_ids'], (0, 128-w, 0, 21-p))
            batch['item_text_masks'] = F.pad(batch['item_text_masks'], (0, 128-w, 0, 21-p))
            batch['target_ids'] = F.pad(batch['target_ids'], (0, 32-batch['target_ids'].shape[1]), value=-100)
        if stage == 'c1':
            targets = torch.tensor([s['target_item_id']-1 for s in samples], device=device)
            loss, parts, logits = model.forward_loss(batch, targets, config['lambda_item'])
            assert torch.isfinite(logits).all()
            del parts, logits
        else:
            loss = model(input_ids=batch['item_text_ids'], attention_mask=batch['item_text_masks'], labels=batch['target_ids']).loss
        assert torch.isfinite(loss), 'Nonfinite training loss'
        (loss / len(batches)).backward()
        total += float(loss.detach()) / len(batches)
        shapes.append(list(batch['item_text_ids'].shape))
        del loss, batch
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    if stage == 'c1':
        assert float(model.catalog_head.weight.grad.norm()) > 0, 'Missing catalog-head learning signal'
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
    assert torch.isfinite(norm), 'Nonfinite gradient'
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=total, gradient_norm=float(norm), rows=sum(map(len, batches)), shapes=shapes)


def total_updates(data, stage, config):
    dataset = data['dataset' if stage == 'gram' else 'c1_dataset']
    return math.ceil(len(dataset)/(config['batch_size']*config['gradient_accumulation'])) * config[f'{stage}_epochs']
