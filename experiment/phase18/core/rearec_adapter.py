"""Pinned ReaRec layers with local Beauty data, valid-item scores and PRL loss."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in value['cuda']])


@contextmanager
def isolated_rng():
    saved = rng_state()
    try:
        yield
    finally:
        restore_rng(saved)


def source_layers(config):
    manifest = json.loads((ROOT / config['source_manifest']).read_text())
    if manifest['commit'] != config['source_commit']:
        raise ValueError('Author commit mismatch')
    source = ROOT / manifest['source']
    for path, sha in manifest['files'].items():
        if digest(source / path) != sha:
            raise ValueError(f'Author source changed: {path}')
    previous = {key: sys.modules.get(key) for key in ('utils', 'utils.constants')}
    try:
        package = types.ModuleType('utils')
        package.__path__ = []
        spec = importlib.util.spec_from_file_location('_s18_rearec_constants', source / 'src/utils/constants.py')
        constants = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(constants)
        # Constants live in this isolated import only; author files are unchanged.
        constants.MAX_ITEM_SEQ_LEN = config['max_history']
        package.constants = constants
        sys.modules['utils'] = package
        sys.modules['utils.constants'] = constants
        spec = importlib.util.spec_from_file_location('_s18_rearec_layers', source / 'src/utils/layers.py')
        layers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layers)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return layers


def load_inputs(config):
    review = json.loads((ROOT / config['input_review']).read_text())
    if config['dataset'] != 'Beauty' or review['dataset'] != 'Beauty':
        raise ValueError('This screen requires Beauty inputs')
    required = {config['prepared'] + '/' + name for name in ('train.jsonl', 'validation.jsonl', 'catalog.json')}
    required.update((config['cohort'], config['references'], config['reference_predictions']))
    if not required <= review['files'].keys():
        raise ValueError('Input paths must be frozen')
    for path, sha in review['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'Frozen input changed: {path}')
    directory = ROOT / config['prepared']
    rows = {s: [json.loads(l) for l in (directory / f'{s}.jsonl').read_text().splitlines()]
            for s in ('train', 'validation')}
    names = json.loads((directory / 'catalog.json').read_text())['items']
    by_user = {r['user_id']: r for r in rows['validation']}
    if (len(rows['train']), len(rows['validation']), len(by_user), len(names)) != (131413, 22363, 22363, 12102):
        raise ValueError('Beauty population changed')
    if names[1:] != sorted(set(names[1:])):
        raise ValueError('Catalog identity mismatch')
    for split in rows.values():
        for row in split:
            if row['user_id'] not in by_user or not 1 <= len(row['history']) <= config['max_history']:
                raise ValueError('Invalid history or user')
            if any(not isinstance(i, int) or not 0 < i < len(names) for i in row['history'] + [row['target']]):
                raise ValueError('Invalid item ID')
    cohort = json.loads((ROOT / config['cohort']).read_text())['user_ids']
    if len(cohort) != 2000 or len(set(cohort)) != 2000 or not set(cohort) <= by_user.keys():
        raise ValueError('Trend cohort mismatch')
    rows['trend'] = [by_user[u] for u in cohort]
    return rows, names


def collate(rows, max_history=20):
    ids = torch.zeros(len(rows), max_history, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, -len(row['history']):] = torch.tensor(row['history'])
    return dict(ids=ids, lengths=torch.tensor([len(r['history']) for r in rows]),
                targets=torch.tensor([r['target'] for r in rows]), users=[r['user_id'] for r in rows])


class ReaRec(nn.Module):
    def __init__(self, config, n_items):
        super().__init__()
        m = config['model']
        self.config = config
        self.max_history = config['max_history']
        self.temperature = m['temperature']
        self.temp_scale = m['temp_scale']
        self.noise_factor = m['noise_factor']
        self.warmup_epoch = m['warmup_epochs']
        self.reason_steps = m['reason_steps']
        self.pl_weight, self.cl_weight = m['pl_weight'], m['cl_weight']
        layers = source_layers(config)
        self.item_emb = nn.Embedding(n_items, m['emb_size'], padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_history + 1, m['emb_size'], padding_idx=self.max_history)
        self.trm_encoder = layers.TransformerEncoder(
            n_layers=m['num_layers'], n_heads=m['num_heads'], hidden_size=m['emb_size'],
            inner_size=m['inner_size'], hidden_dropout_prob=m['dropout'], attn_dropout_prob=m['dropout'],
            hidden_act='gelu', layer_norm_eps=m['layer_norm_eps'])
        self.model = layers.ReaRecAutoRegressiveWrapper(self.trm_encoder, m['emb_size'], self.reason_steps)
        self.apply(self.init_weights)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[self.max_history].zero_()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config['model']['initializer_range'])
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def input_embeddings(self, ids):
        valid = ids.ne(0)
        positions = torch.where(valid, valid.long().cumsum(1) - 1, self.max_history)
        return self.item_emb(ids) + self.pos_emb(positions)

    def forward(self, batch, epoch=0, reason_steps=None, noise_override=None):
        device = self.item_emb.weight.device
        ids, lengths = batch['ids'].to(device), batch['lengths'].to(device)
        noise = self.noise_factor if self.training and epoch > self.warmup_epoch else 0.0
        if noise_override is not None:
            noise = noise_override
        states = self.model(self.input_embeddings(ids), lengths, noise_factor=noise, reason_step=reason_steps)
        # Exclude PAD from the softmax denominator, both final and intermediate.
        weights = self.item_emb.weight[1:]
        scores = states[:len(ids), -1] @ weights.T / self.temperature
        return dict(prediction=scores, model_output=states, test_item_embs=weights,
                    labels=batch['targets'].to(device) - 1)

    def loss(self, out):
        labels, scores, states, weights = (out[k] for k in ('labels', 'prediction', 'model_output', 'test_item_embs'))
        b = len(labels)
        k = states.shape[1] - 1
        if k < 1:
            raise ValueError('Zero-step mode is inference-only; not a PRL training baseline')
        ce = F.cross_entropy(scores, labels)
        logits = torch.einsum('btd,nd->btn', states[:b, :-1], weights)
        temperatures = self.temperature * self.temp_scale ** torch.arange(k, 0, -1, device=scores.device)
        progressive = F.cross_entropy((logits / temperatures[None, :, None]).reshape(-1, len(weights)),
                                      labels.repeat_interleave(k))
        # The author's PRL loss leaves cl_loss unbound during its no-noise warmup.
        # Initialize to zero; otherwise retain the source's unnormalized formula.
        contrastive = ce.new_zeros(())
        if len(states) == 2 * b and self.cl_weight > 0:
            similarities = torch.einsum('btd,ktd->btk', states[b:, 1:], states[:b, 1:]) / self.temperature
            targets = torch.arange(b, device=scores.device)[:, None].expand(-1, k)
            contrastive = F.cross_entropy(similarities.permute(0, 2, 1), targets)
        parts = dict(ce=ce, progressive=progressive, contrastive=contrastive)
        return ce + self.pl_weight * progressive + self.cl_weight * contrastive, parts


def save_checkpoint(path, model, optimizer, progress):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), progress=progress,
                    rng=rng_state()), tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, restore_random=True):
    saved = torch.load(path, map_location=model.item_emb.weight.device, weights_only=False)
    model.load_state_dict(saved['model'])
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer'])
    if restore_random:
        restore_rng(saved['rng'])
    return saved['progress']
