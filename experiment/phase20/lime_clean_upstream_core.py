"""Explicit train-row adapters. Never calls GRAM's full user_sequence reader."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def validate_rows(rows, sequences):
    rebuilt = {}; counts = {}
    for row in rows:
        u = row['user_id']; h = row['history']; y = row['target']
        if u not in rebuilt:
            assert len(h) == 1
            rebuilt[u] = list(h)
        assert rebuilt[u][-20:] == h
        rebuilt[u].append(y); counts[u] = counts.get(u, 0)+1
    for u,s in sequences.items():
        assert rebuilt.get(u,s[:1]) == s
        assert counts.get(u,0) == len(s)-1
    assert set(rebuilt) <= set(sequences)
    return sum(counts.values())


def cosine_neighbors(embeddings, k=5, block=128):
    """Exclude self; ties resolve by ascending 1-based numeric catalog ID."""
    e = np.asarray(embeddings,dtype=np.float32)
    e = e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    assert np.isfinite(e).all()
    ids = np.arange(1,len(e)+1)
    out = []
    for start in range(0,len(e),block):
        scores = e[start:start+block] @ e.T
        scores[np.arange(len(scores)),np.arange(start,start+len(scores))] = -np.inf
        order = np.lexsort((np.broadcast_to(ids,scores.shape),-scores),axis=1)[:,:k]
        out.append(ids[order])
    result = np.concatenate(out)
    assert not (result == ids[:,None]).any()
    return result


def item_inputs(source_ids, texts, lexical, neighbors):
    assert len(source_ids) == len(neighbors) and set(source_ids) == set(lexical) == set(texts)
    result = {}
    for item, near in zip(source_ids,neighbors):
        assert item not in [source_ids[j-1] for j in near]
        similar = ', '.join(lexical[source_ids[j-1]] for j in near)
        result[item] = f'item: {lexical[item]}; similar items: {similar}; {texts[item]}'
    return result


def gram_sample(row, source_ids, inputs, lexical):
    history = row['history'][-20:][::-1]
    source_history = [source_ids[i-1] for i in history]
    target = source_ids[row['target']-1]
    prompt = ' ; '.join(lexical[i] for i in source_history)
    return dict(user_id=row['user_id'],
        input=[f'What would user purchase after {prompt} ?']+[inputs[i] for i in source_history],
        output=lexical[target],history_item_ids=history,target_item_id=row['target'])


def runtime_args():
    return SimpleNamespace(item_prompt_max_len=128,target_max_len=32,max_his=20,
        item_id_type='split',hierarchical_id_type='hierarchy_v1_c32_l5_len32768_split')


def new_gram(backbone_path):
    """Only external T5 initialization; accepts no trained recommendation checkpoint."""
    from transformers import T5Config,AutoModelForSeq2SeqLM,AutoTokenizer
    from model import create_model
    p = str(Path(backbone_path))
    config = T5Config.from_pretrained(p,local_files_only=True)
    config.max_seq_len=128; config.max_item_num=20; config.use_position_embedding=1; config.sample_num='1'
    config.cf0_enabled=False; config.hi_gram_enabled=False; config.s17_modules=''
    tokenizer = AutoTokenizer.from_pretrained(p,local_files_only=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(p,config=config,local_files_only=True)
    model = create_model('gram',config=config); model.load_t5(base.state_dict())
    return model,tokenizer


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in state.items(): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class CleanRows(torch.utils.data.Dataset):
    def __init__(self, rows, source_ids, inputs, lexical):
        self.rows,self.source_ids,self.inputs,self.lexical=rows,source_ids,inputs,lexical
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return gram_sample(self.rows[i],self.source_ids,self.inputs,self.lexical)
