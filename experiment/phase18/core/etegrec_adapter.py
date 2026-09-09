"""Pinned ETEGRec model/RQVAE with local data, optimizer and item interfaces."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def source_classes(config):
    manifest = json.loads((ROOT / config['source_manifest']).read_text())
    if manifest['commit'] != config['source_commit']:
        raise ValueError('Source commit mismatch')
    source = ROOT / manifest['source']
    for name, sha in manifest['files'].items():
        if digest(source / name) != sha:
            raise ValueError(f'Author source changed: {name}')
    # Author files use absolute imports; isolate those names from project modules.
    prior = {name: sys.modules.get(name) for name in ('layers', 'vq')}
    loaded = {}
    try:
        for name in ('layers', 'vq', 'model'):
            spec = importlib.util.spec_from_file_location('_s18_eteg_' + name, source / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
            if name in prior:
                sys.modules[name] = module
    finally:
        for name, value in prior.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return loaded['model'].Model, loaded['vq'].RQVAE


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_models(config, embeddings, device):
    Model, RQVAE = source_classes(config)
    m = config['model']
    t5 = T5ForConditionalGeneration(T5Config(
        num_layers=m['encoder_layers'], num_decoder_layers=m['decoder_layers'],
        d_model=m['d_model'], d_ff=m['d_ff'], num_heads=m['num_heads'], d_kv=m['d_kv'],
        dropout_rate=m['dropout_rate'], vocab_size=1, pad_token_id=0, eos_token_id=300,
        decoder_start_token_id=0, feed_forward_proj='relu', n_positions=4 * config['max_history'] + 10,
        use_cache=False,
    ))
    # Compatibility for the author's GenerationMixin wrapper on transformers 4.56.
    if not hasattr(t5, '_supports_cache_class'):
        t5._supports_cache_class = False
    rec = Model(m, t5, n_items=len(embeddings), code_length=4, code_number=m['code_num']).to(device)
    rec.device = torch.device(device)
    with torch.no_grad():
        rec.semantic_embedding.weight.copy_(embeddings.to(device))
    rec.semantic_embedding.requires_grad_(False)
    rq = RQVAE(m, in_dim=embeddings.shape[1]).to(device)
    return rec, rq


def load_inputs(config):
    review = json.loads((ROOT / config['input_review']).read_text())
    for path, sha in review['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'Frozen input changed: {path}')
    checkpoint = ROOT / review['collaborative_checkpoint']['path']
    if digest(checkpoint) != review['collaborative_checkpoint']['sha256']:
        raise ValueError('Collaborative checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    embeddings = F.normalize(saved['model_state_dict']['item_embedding.weight'].float(), dim=-1)
    embeddings[0].zero_()
    directory = ROOT / config['prepared']
    records = {split: [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines()]
               for split in ('train', 'validation')}
    names = json.loads((directory / 'catalog.json').read_text())['items']
    if embeddings.shape != (len(names), config['model']['semantic_hidden_size']):
        raise ValueError('Embedding/catalog shape mismatch')
    if not torch.isfinite(embeddings).all():
        raise ValueError('Nonfinite embeddings')
    train_items = sorted({i for r in records['train'] for i in r['history'] + [r['target']]})
    cohort = json.loads((ROOT / config['cohort']).read_text())['user_ids']
    by_user = {r['user_id']: r for r in records['validation']}
    if len(by_user) != 22363 or len(records['train']) != 131413 or len(names) != 12102:
        raise ValueError('Local population changed')
    records['trend'] = [by_user[u] for u in cohort]
    return embeddings, records, train_items, names


def collate(rows):
    width = max(len(r['history']) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    for n, r in enumerate(rows):
        ids[n, :len(r['history'])] = torch.tensor(r['history'])
    return {'ids': ids, 'targets': torch.tensor([r['target'] for r in rows]),
            'users': [r['user_id'] for r in rows]}


def set_phase(rec, rq, phase):
    if phase not in ('id', 'rec', 'finetune'):
        raise ValueError(phase)
    for p in rec.parameters():
        p.requires_grad_(phase != 'id')
        p.grad = None
    rec.semantic_embedding.requires_grad_(False)
    for p in rq.parameters():
        p.requires_grad_(phase == 'id')
        p.grad = None
    rec.train(phase != 'id')
    rq.train(phase == 'id')


def symmetric_kl(a, b):
    # Preserve author's distance-logit convention and batchmean normalization.
    a, b = F.log_softmax(a.reshape(-1, a.shape[-1]), -1), F.log_softmax(b.reshape(-1, b.shape[-1]), -1)
    return (F.kl_div(a, b, reduction='batchmean', log_target=True)
            + F.kl_div(b, a, reduction='batchmean', log_target=True))


def symmetric_contrastive(a, b, temperature=0.07):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)


def losses(rec, rq, codes, batch, phase, alignment=True):
    device = rec.semantic_embedding.weight.device
    ids, targets = batch['ids'].to(device), batch['targets'].to(device)
    inputs = codes[ids].flatten(1).clone()
    labels = codes[targets].clone()
    out = rec(input_ids=inputs, attention_mask=inputs.ne(-1), labels=labels)
    ce = F.cross_entropy(out.logits.reshape(-1, rec.code_number), labels.reshape(-1))
    values = {'ce': ce}
    if phase == 'finetune':
        return ce, values
    semantic = rec.semantic_embedding(targets)
    recon, _, _, _, target_logits = rq(semantic)
    _, unique_index = np.unique(targets.detach().cpu().numpy(), return_index=True)
    unique_index = torch.as_tensor(unique_index, device=device)
    _, _, _, _, seq_logits = rq.rq(out.seq_project_latents)
    kl = symmetric_kl(seq_logits[unique_index], target_logits[unique_index])
    cl = symmetric_contrastive(recon[unique_index], out.dec_latents[unique_index])
    values.update(kl=kl, contrastive=cl)
    if phase == 'id':
        unique_semantic = semantic[unique_index]
        unique_recon, commitment, _, _, _ = rq(unique_semantic)
        vq = F.mse_loss(unique_recon, unique_semantic) + rq.quant_loss_weight * commitment
        values['vq'] = vq
        total = vq
    else:
        total = ce
    if alignment:
        total = total + 0.0001 * kl + 0.0003 * cl
    return total, values


@torch.no_grad()
def make_codes(rec, rq, batch_size=1024):
    rec.eval()
    rq.eval()
    prefixes = torch.cat([rq.get_indices(x) for x in rec.semantic_embedding.weight[1:].split(batch_size)]).cpu().tolist()
    counts, rows = {}, [[-1] * 4]
    for prefix in prefixes:
        key = tuple(prefix)
        suffix = counts.get(key, 0)
        if suffix >= rec.code_number:
            raise ValueError(f'Code collision overflow: {suffix + 1}')
        rows.append(prefix + [suffix])
        counts[key] = suffix + 1
    return torch.tensor(rows, dtype=torch.long, device=rec.device), {
        'distinct_prefixes': len(counts), 'max_collision': max(counts.values()),
        'catalog_items': len(prefixes)}


class CatalogTrie:
    def __init__(self, codes, code_num):
        children, items = [[-1] * code_num], [0]
        for item, code in enumerate(codes[1:].cpu().tolist(), start=1):
            node = 0
            for token in code:
                if children[node][token] == -1:
                    children[node][token] = len(children)
                    children.append([-1] * code_num)
                    items.append(0)
                node = children[node][token]
            if items[node]:
                raise ValueError('Duplicate complete item code')
            items[node] = item
        self.children = torch.tensor(children, dtype=torch.long, device=codes.device)
        self.items = torch.tensor(items, dtype=torch.long, device=codes.device)


@torch.no_grad()
def generate_items(rec, codes, trie, batch, beam=50):
    from transformers.modeling_outputs import BaseModelOutput
    rec.eval()
    inputs = codes[batch['ids'].to(rec.device)].flatten(1).clone()
    mask = inputs.ne(-1)
    embeds = rec.get_input_embeddings(inputs, mask)
    enc = rec.get_encoder()(inputs_embeds=embeds, attention_mask=mask, return_dict=True).last_hidden_state
    b = len(inputs)
    sequences = torch.zeros(b, 1, 1, dtype=torch.long, device=rec.device)
    scores = torch.zeros(b, 1, device=rec.device)
    nodes = torch.zeros(b, 1, dtype=torch.long, device=rec.device)
    for depth in range(4):
        width = sequences.shape[1]
        out = rec(encoder_outputs=BaseModelOutput(last_hidden_state=enc.repeat_interleave(width, 0)),
                  attention_mask=mask.repeat_interleave(width, 0),
                  decoder_input_ids=sequences.reshape(b * width, -1))
        logits = out.logits[:, -1].float().log_softmax(-1).reshape(b, width, -1)
        child = trie.children[nodes.clamp_min(0)]
        candidates = (scores[..., None] + logits).masked_fill((child < 0) | (nodes[..., None] < 0), -torch.inf)
        k = min(beam, candidates.shape[1] * candidates.shape[2])
        scores, selected = candidates.flatten(1).topk(k, dim=1, sorted=True)
        parent, token = selected // rec.code_number, selected % rec.code_number
        sequences = torch.cat([sequences.gather(1, parent[..., None].expand(-1, -1, sequences.shape[-1])), token[..., None]], -1)
        nodes = child.flatten(1).gather(1, selected)
    item_ids = trie.items[nodes.clamp_min(0)]
    if not torch.isfinite(scores).all() or (item_ids == 0).any():
        raise ValueError('Beam failed to produce requested legal items')
    # Deterministic tie order by item ID, without altering sequence probabilities.
    item_order = item_ids.argsort(dim=1, stable=True)
    item_ids, scores = item_ids.gather(1, item_order), scores.gather(1, item_order)
    rank_order = scores.argsort(dim=1, descending=True, stable=True)
    return item_ids.gather(1, rank_order).cpu(), (scores.gather(1, rank_order) / 4).cpu()


def scheduler(optimizer, total_steps, warmup=0, kind='cosine'):
    if not 0 <= warmup < total_steps:
        raise ValueError('Invalid optimizer schedule')
    def factor(step):
        if warmup and step < warmup:
            return step / warmup
        progress = min(1.0, max(0.0, (step - warmup) / (total_steps - warmup)))
        return max(0.0, 1.0 - progress) if kind == 'linear' else (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def save_bundle(path, rec, rq, codes, optimizers, schedulers, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save({'rec': rec.state_dict(), 'rq': rq.state_dict(), 'codes': codes,
                'rq_initted': [layer.initted for layer in rq.rq.vq_layers],
                'optimizers': {k: o.state_dict() for k, o in optimizers.items()},
                'schedulers': {k: s.state_dict() for k, s in schedulers.items()},
                'progress': progress, 'rng': rng_state()}, temporary)
    temporary.replace(path)


def load_bundle(path, rec, rq, optimizers=None, schedulers=None, restore_random=True):
    bundle = torch.load(path, map_location=rec.device, weights_only=False)
    rec.load_state_dict(bundle['rec'], strict=True)
    rq.load_state_dict(bundle['rq'], strict=True)
    for layer, initted in zip(rq.rq.vq_layers, bundle['rq_initted']):
        layer.initted = initted
    # Schedulers are constructed before optimizer state is restored.
    if schedulers is not None:
        for k, s in schedulers.items():
            s.load_state_dict(bundle['schedulers'][k])
    if optimizers is not None:
        for k, o in optimizers.items():
            o.load_state_dict(bundle['optimizers'][k])
    if restore_random:
        restore_rng(bundle['rng'])
    return bundle['codes'], bundle['progress']
