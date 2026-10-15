"""Bounded two-domain experiment: expert opinions inside GRAM before scoring."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20.catalog_scope_core import metrics, contrast, ranks
from experiment.phase21.expert_feedback_core import (
    ARMS, CardBuilder, shuffle_evidence, pack_cards, encode_context, generate_from_context,
)
from experiment.phase21.night_resume_core import process_identity, identity_alive

ROOT = base.ROOT
OUT = ROOT / 'artifacts/phase21/expert_feedback_20260924_v1'
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
REFERENCE = ROOT / 'artifacts/phase21/text_contribution_20260923_v1/real'
PLAN = ROOT / 'plan/第二十一阶段/21.6_专家意见前置与语义反馈_20260924.md'
REPORT = ROOT / 'report/第二十一阶段/专家意见前置与语义反馈_20260924.md'
MODULE = 'experiment.phase21.expert_feedback'
DOMAINS = ('Toys', 'Beauty')
read, write, sha, stamp = base.read, base.write, base.sha, base.stamp


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def status(folder, state, **values):
    write(folder / 'status.json', dict(state=state, updated_at=stamp(), pid=os.getpid(), test_read=False, **values))


def prepare():
    from transformers import AutoTokenizer
    from experiment.phase20.lime_clean_gram_core import T5
    OUT.mkdir(exist_ok=False)
    status(OUT, 'PREPARING')
    cfg = dict(domains=list(DOMAINS), arms=list(ARMS), seed=2023, epochs=2, learning_rate=3e-5,
               weight_decay=.01, batch_size=32, microbatch=4, warmup_fraction=.05,
               per_expert=3, card_width=128, beam_size=50, development_users=512,
               expected_workload_peak_mib=16384, total_lease_mib=30720,
               allocation_limit_gib=15, minimum_free_mib=32768, max_parallel=2,
               worker_timeout_seconds=12*3600, supervisor_timeout_seconds=24*3600,
               gpus=dict(Toys='GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160',
                         Beauty='GPU-7952ad6e-2db7-fb2e-581b-34be56287640'),
               fallback_gpus=['GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09',
                              'GPU-07374397-b213-09c1-2988-718acb6ed231',
                              'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2',
                              'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253'],
               bootstrap_repeats=2000, bootstrap_seed=20260924, test_read=False,
               output_score_correction=False, development_labels_used_for_fit=False,
               encoder_frozen=True, shared_embedding_frozen=True, lm_head_frozen=True)
    write(OUT / 'config.json', cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(T5), local_files_only=True)
    manifest = {}

    def use(path, expected=None):
        path = Path(path)
        assert path.name not in ('test.jsonl', 'test.json', 'user_sequence.txt')
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        manifest[str(path.relative_to(ROOT))] = digest
        return path

    for domain in DOMAINS:
        p, src = OUT / domain, SOURCE / domain
        p.mkdir()
        status(p, 'PREPARING')
        data_cfg = read(use(src / 'data_config.json'))
        use(ROOT / data_cfg['parent_checkpoint'], data_cfg['parent_sha256'])
        catalog = read(use(src / 'catalog.json'))
        original = read(use(src / 'samples.json'))
        samples = [dict(s, source_sample_index=i) for i, s in enumerate(original)
                   if s['role'] in ('ranker_fit', 'development')]
        assert len({s['user_id'] for s in samples}) == len(samples)
        assert sum(s['role'] == 'ranker_fit' for s in samples) == dict(Toys=2773, Beauty=3310)[domain]
        assert sum(s['role'] == 'development' for s in samples) == 512
        design = ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1'
        receipt = read(use(design / 'summary.json'))['output_sha256']
        path = design / domain / 'terminal_holdout.jsonl'
        held = {s['user_id']: s for s in rows(use(path, receipt[str(path.relative_to(ROOT))]))}
        path = design / domain / 'upstream_sequences_without_terminal.jsonl'
        histories = {s['user_id']: s['history'] for s in rows(use(path, receipt[str(path.relative_to(ROOT))]))}
        for s in samples:
            assert s['full_history'] == histories[s['user_id']]
            assert s['history'] == s['full_history'][-20:]
            if s['role'] == 'ranker_fit':
                assert all(s[k] == held[s['user_id']][k] for k in ('history', 'target', 'role'))
            else:
                assert s['user_id'] not in held
        write(p / 'samples.json', samples)
        write(p / 'data_config.json', data_cfg)
        scores = np.load(use(src / 'experts.npy'), mmap_mode='r')
        with np.load(use(ROOT / data_cfg['embedding']), allow_pickle=False) as data:
            assert data['item_ids'].tolist() == catalog['items'][1:]
            embedding = data['embeddings'].copy()
        builder = CardBuilder(catalog, tokenizer, embedding, read(use(ROOT / data_cfg['itemcf'])),
                              cfg['per_expert'], cfg['card_width'])
        prefixes, tails, users, roles, experts, all_facts = [], [], [], [], [], []
        for j, sample in enumerate(samples):
            # Deliberately pass only past history and frozen expert scores.
            a, b, facts = builder.build(sample['full_history'], scores[sample['source_sample_index']])
            prefixes.extend(a); tails.extend(b); users.extend([sample['user_id']] * len(a))
            roles.extend([sample['role']] * len(a)); experts.extend([f['expert'] for f in facts])
            all_facts.append(dict(user_id=sample['user_id'], facts=facts))
            if j % 128 == 0:
                status(p, 'PREPARING', stage='target_free_cards', completed=j, total=len(samples))
        shuffled, donors = shuffle_evidence(prefixes, tails, users, roles, experts)
        numeric = [a + [1] for a in prefixes]
        semantic = [a + b + [1] for a, b in zip(prefixes, tails)]
        banks = {arm: pack_cards(cards, len(samples), cfg['per_expert']*3, cfg['card_width'])
                 for arm, cards in (('numeric', numeric), ('semantic', semantic), ('shuffled', shuffled))}
        np.testing.assert_array_equal((banks['semantic'] != 0).sum(-1), (banks['shuffled'] != 0).sum(-1))
        changed = np.any(banks['semantic'] != banks['shuffled'], axis=-1)
        assert changed.mean() > .95, ('Insufficient placebo mismatch', changed.mean())
        # Replacing all target labels cannot change the card builder's input.
        target_check = []
        for j in (0, len(samples)//2, len(samples)-1):
            sample = dict(samples[j], target=(samples[j]['target'] % (len(catalog['items'])-1)) + 1)
            a, b, facts = builder.build(sample['full_history'], scores[sample['source_sample_index']])
            assert [x+y+[1] for x,y in zip(a,b)] == semantic[j*9:(j+1)*9]
            assert facts == all_facts[j]['facts']
            target_check.append(sample['user_id'])
        np.savez_compressed(p / 'cards.npz', **banks, donors=donors,
                            prefix_lengths=np.asarray(list(map(len, prefixes))).reshape(len(samples), 9))
        write(p / 'card_evidence.json', all_facts)
        examples = []
        for j in (0, len(samples)-1):
            examples.append(dict(user_id=samples[j]['user_id'], role=samples[j]['role'],
                numeric=[tokenizer.decode(x, skip_special_tokens=True) for x in banks['numeric'][j]],
                semantic=[tokenizer.decode(x, skip_special_tokens=True) for x in banks['semantic'][j]],
                shuffled=[tokenizer.decode(x, skip_special_tokens=True) for x in banks['shuffled'][j]]))
        write(p / 'card_examples.json', examples)
        audit = dict(training_rows=sum(s['role']=='ranker_fit' for s in samples), development_rows=512,
            input_labels_ignored_checks=target_check, target_not_argument_to_builder=True,
            disjoint_fit_gate_development_users=True, no_test_read=True,
            cards_per_user=9, card_width=128, evidence_mismatch_fraction=float(changed.mean()),
            evidence_same_lengths=True, candidate_and_score_prefix_unchanged=True,
            mean_feedback_tokens={a: float((bank != 0).sum((1,2)).mean()) for a,bank in banks.items()},
            candidate_coverage={role: float(np.mean([s['target'] in [f['item_id'] for f in all_facts[j]['facts']]
                for j,s in enumerate(samples) if s['role']==role])) for role in ('ranker_fit','development')})
        write(p / 'input_audit.json', audit)
        reference = rows(use(REFERENCE / domain / 'predictions.jsonl'))
        assert {r['user_id'] for r in reference} == {s['user_id'] for s in samples if s['role']=='development'}
        use(REFERENCE / domain / 'summary.json')
        use(src / 'cache_summary.json')
        for name in ('samples.json', 'data_config.json', 'cards.npz', 'card_evidence.json', 'input_audit.json'):
            use(p / name)
        status(p, 'QUEUED', training_rows=audit['training_rows'], development_users=512)
        del scores, builder, embedding, banks
        gc.collect()
    sources = [PLAN, Path(__file__), ROOT/'experiment/phase21/expert_feedback_core.py',
        ROOT/'experiment/phase21/test_expert_feedback.py', ROOT/'experiment/phase21/run_expert_feedback.sh',
        ROOT/'experiment/GPU_MEMORY_LEASE_POLICY.md', ROOT/'experiment/gpu_memory_lease.py',
        ROOT/'experiment/phase21/attribute_retention_core.py', ROOT/'experiment/phase21/night_resume_core.py',
        ROOT/'experiment/phase18/core/diff_data.py']
    sources += [ROOT/'experiment/phase20'/name for name in ('lime_decode_screen.py','lime_decode_screen_core.py',
        'lime_clean_gram_core.py','lime_clean_upstream_core.py','lime_parallel_core.py','catalog_scope_core.py')]
    sources += list((ROOT/'GRAM/src').rglob('*.py'))
    for path in sources:
        use(path)
        dest = OUT/'sources'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for name in ('config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','spiece.model'):
        use(T5/name)
    use(OUT/'config.json')
    shutil.copy2(PLAN, OUT/'protocol.md')
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    status(OUT, 'PREPARED')
    write_report()
    print(json.dumps(dict(state='PREPARED', frozen_files=len(manifest))), flush=True)


def verify_manifest():
    for rel, digest in read(OUT/'manifest.json')['sha256'].items():
        assert sha(ROOT/rel) == digest, f'Frozen input changed: {rel}'


class Worker:
    def __init__(self, domain, lease):
        import torch
        from experiment.phase20.lime_clean_gram_core import T5, SPECS
        from experiment.phase20.lime_clean_upstream_core import new_gram, runtime_args
        from experiment.phase20.lime_parallel_core import CachedCollator
        from experiment.phase18.core.diff_data import DiffDataset
        sys.path.insert(0, str(ROOT/'GRAM/src'))
        from processor import CollatorGRAM
        from utils import generation_trie
        self.domain, self.folder, self.lease = domain, OUT/domain, lease
        self.started = time.monotonic()
        self.cfg = read(OUT/'config.json')
        self.data_cfg = read(self.folder/'data_config.json')
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(self.cfg['allocation_limit_gib']*2**30 /
                                                   torch.cuda.get_device_properties(0).total_memory)
        self.device = torch.device('cuda')
        self.samples = read(self.folder/'samples.json')
        self.fit = [j for j,s in enumerate(self.samples) if s['role']=='ranker_fit']
        self.dev = [j for j,s in enumerate(self.samples) if s['role']=='development']
        self.catalog = read(SOURCE/domain/'catalog.json')
        self.dataset = DiffDataset(self.samples, self.catalog)
        with np.load(self.folder/'cards.npz') as data:
            self.cards = {a: data[a].copy() for a in ARMS if a != 'control'}
        self.model, self.tokenizer = new_gram(T5)
        self.model.set_checkpoint(False)
        saved = torch.load(ROOT/self.data_cfg['parent_checkpoint'], map_location='cpu')
        self.parent = saved['state_dict']; del saved
        self.model.load_state_dict(self.parent, strict=True)
        self.model.to(self.device)
        args = runtime_args(); args.hierarchical_id_type = SPECS[domain]['lexical']
        self.raw_collator = CollatorGRAM(self.tokenizer, args=args)
        self.collator = CachedCollator(self.raw_collator)
        self.path_to_id = {tuple(path): i for i,path in enumerate(self.catalog['decoder_paths'], 1)}
        self.max_length = max(map(len, self.catalog['decoder_paths']))
        self.prefix = generation_trie.prefix_allowed_tokens_fn(generation_trie.Trie(self.catalog['decoder_paths']))
        self.reference = {s['user_id']: s for s in rows(REFERENCE/domain/'predictions.jsonl')}
        self.reset()

    def phase(self, stage, **kw):
        import torch
        assert self.lease.poll() is None, 'GPU lease sidecar exited'
        status(self.folder, 'RUNNING', stage=stage, elapsed_seconds=time.monotonic()-self.started,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, **kw)

    def reset(self):
        import torch
        from experiment.phase20.lime_clean_gram_core import seed_all
        seed_all(self.cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.model.load_state_dict(self.parent, strict=True)
        self.model.requires_grad_(False).eval()
        self.model.decoder.block.requires_grad_(True)
        self.model.decoder.final_layer_norm.requires_grad_(True)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable = [n for n,p in self.model.named_parameters() if p.requires_grad]
        assert self.trainable and all(n.startswith(('decoder.block.', 'decoder.final_layer_norm.')) for n in self.trainable)
        assert not self.model.shared.weight.requires_grad and not self.model.lm_head.weight.requires_grad
        torch.cuda.empty_cache()

    def batch(self, indices, inference=False):
        examples = [self.dataset[j] for j in indices]
        if inference:
            examples = [dict(x, output=self.catalog['lexical_ids'][1], target_item_id=1) for x in examples]
        return self.collator(examples)

    def context(self, indices, arm, batch):
        import torch
        extra = None if arm == 'control' else torch.from_numpy(self.cards[arm][indices]).to(self.device)
        return encode_context(self.model, batch['item_text_ids'].to(self.device),
                              batch['item_text_masks'].to(self.device), extra)

    def rank(self, ids, scores, sample):
        allowed = [(i,float(s)) for i,s in zip(ids,scores) if i not in sample['history']]
        ordered = [i for i,s in sorted(allowed, key=lambda pair:(-pair[1],pair[0]))]
        result = ordered.index(sample['target'])+1 if sample['target'] in ordered else 2**30
        check_scores = np.asarray([scores], dtype=np.float64)
        check_scores[0,np.isin(ids,sample['history'])] = -np.inf
        assert result == int(ranks(check_scores, np.asarray([ids]), [sample['target']])[0])
        return result

    def generate(self, index, arm, native=False):
        import torch
        self.model.eval()
        batch = self.batch([index], inference=True)
        kwargs = dict(max_length=self.max_length, prefix_allowed_tokens_fn=self.prefix,
            num_beams=50, num_return_sequences=50, return_dict_in_generate=True,
            output_scores=True, length_penalty=1.)
        with torch.no_grad():
            if native:
                output = self.model.generate(input_ids=batch['item_text_ids'].to(self.device),
                    attention_mask=batch['item_text_masks'].to(self.device), **kwargs)
            else:
                output = generate_from_context(self.model, self.context([index], arm, batch), **kwargs)
        paths = [p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids = [self.path_to_id[tuple(p)] for p in paths]
        scores = output.sequences_scores.cpu().double().tolist()
        assert len(set(ids)) == 50 and np.isfinite(scores).all()
        sample = self.samples[index]
        return dict(user_id=sample['user_id'], target=sample['target'], sample_index=index,
                    ids=ids, scores=scores, rank=self.rank(ids,scores,sample))

    def windows(self):
        from experiment.phase20.lime_clean_gram_core import epoch_order, update_windows
        result = []
        for epoch in range(1, self.cfg['epochs']+1):
            order = [self.fit[j] for j in epoch_order(len(self.fit), self.cfg['seed'], epoch)]
            result += list(update_windows(order, self.cfg['microbatch'],
                                          self.cfg['batch_size']//self.cfg['microbatch']))
        return result

    def optimizer(self):
        import torch
        from transformers import get_linear_schedule_with_warmup
        opt = torch.optim.AdamW(self.parameters, lr=self.cfg['learning_rate'], weight_decay=self.cfg['weight_decay'])
        count = len(self.windows())
        sched = get_linear_schedule_with_warmup(opt, int(count*self.cfg['warmup_fraction']), count)
        return opt,sched

    def update(self, window, arm, opt, sched):
        import torch
        self.model.eval(); self.model.decoder.train()
        opt.zero_grad(set_to_none=True)
        n = sum(map(len,window)); total=0.
        for indices in window:
            batch = self.batch(indices)
            _, mask, encoded = self.context(indices,arm,batch)
            loss = self.model(encoder_outputs=encoded, attention_mask=mask,
                              labels=batch['target_ids'].to(self.device), use_cache=False).loss
            assert torch.isfinite(loss)
            (loss*len(indices)/n).backward()
            total += float(loss.detach())*len(indices)/n
            del loss,batch,encoded,mask
        assert all(p.grad is None for p in self.model.encoder.parameters())
        assert self.model.lm_head.weight.grad is None
        assert all(p.grad is not None for p in self.parameters)
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.)
        assert torch.isfinite(norm)
        opt.step();sched.step();opt.zero_grad(set_to_none=True)
        return dict(loss=total, gradient_norm=float(norm))

    def profile(self):
        import torch
        self.phase('native_parity_and_profile')
        checks=[]
        # Include the longest history in the evaluation cohort.
        chosen=list(dict.fromkeys(self.dev[:2]+[max(self.dev,key=lambda j:len(self.samples[j]['history']))]))
        for j in chosen:
            native=self.generate(j,'control',native=True)
            pre=self.generate(j,'control')
            assert native['ids']==pre['ids']
            np.testing.assert_array_equal(native['scores'],pre['scores'])
            old=self.reference[self.samples[j]['user_id']]['native']
            assert native['ids']==old['ids']
            np.testing.assert_array_equal(native['scores'],old['scores'])
            checks.append(dict(user_id=native['user_id'],native_entry_exact=True, frozen_reference_exact=True))
        for j in self.fit[:2]+self.dev[:2]:
            a=self.batch([j]);b=self.raw_collator([self.dataset[j]])
            assert torch.equal(a['item_text_ids'],b['item_text_ids'])
            assert torch.equal(a['item_text_masks'],b['item_text_masks'])
        # Two actual optimizer updates, including longest histories, discarded.
        from experiment.phase20.lime_clean_gram_core import update_windows
        longest=sorted(self.fit,key=lambda j:len(self.samples[j]['history']),reverse=True)[:32]
        stress=list(update_windows(longest,4,8))[0]
        opt,sched=self.optimizer(); timings=[]
        for window in (stress,self.windows()[0]):
            started=time.monotonic(); result=self.update(window,'semantic',opt,sched)
            torch.cuda.synchronize();timings.append(dict(seconds=time.monotonic()-started,**result))
        del opt,sched
        self.reset()
        j=max(self.dev,key=lambda j:len(self.samples[j]['history']))
        output=self.generate(j,'semantic')
        assert output['ids']
        audit=dict(state='PASSED', native_parity=checks, original_history_tokens_unchanged=True,
            frozen_encoder_and_output_head=True, pre_decoder_context=True, output_score_correction=False,
            trainable_parameters=sum(p.numel() for p in self.parameters), trainable_names=self.trainable,
            stress_updates=timings, trial_updates_discarded=2,
            projected_training_seconds=float(np.mean([r['seconds'] for r in timings]))*len(self.windows())*4,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write(self.folder/'preflight_checks.json',audit)

    def train(self, arm):
        import torch
        from experiment.phase20.lime_clean_gram_core import atomic_save, cpu_tree, rng_state, restore_rng
        p=self.folder/arm;p.mkdir(exist_ok=True)
        self.reset()
        if (p/'final.pt').exists():
            saved=torch.load(p/'final.pt',map_location='cpu')
            assert saved['manifest_sha256']==sha(OUT/'manifest.json') and saved['arm']==arm
            self.model.load_state_dict(saved['decoder'],strict=False)
            return
        opt,sched=self.optimizer();windows=self.windows();done=0
        if (p/'latest.pt').exists():
            saved=torch.load(p/'latest.pt',map_location='cpu')
            assert saved['manifest_sha256']==sha(OUT/'manifest.json') and saved['arm']==arm
            self.model.load_state_dict(saved['decoder'],strict=False)
            opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng']);done=saved['completed_updates'];del saved
        with (p/'updates.jsonl').open('a') as log:
            for update in range(done,len(windows)):
                started=time.monotonic(); values=self.update(windows[update],arm,opt,sched)
                log.write(json.dumps(dict(update=update+1,seconds=time.monotonic()-started,**values))+'\n')
                log.flush()
                if (update+1)%25==0 or update+1==len(windows):
                    atomic_save(p/'latest.pt',dict(arm=arm,
                        decoder={n:cpu_tree(v) for n,v in self.model.state_dict().items() if n in self.trainable},
                        optimizer=cpu_tree(opt.state_dict()),scheduler=sched.state_dict(),rng=rng_state(),
                        completed_updates=update+1,manifest_sha256=sha(OUT/'manifest.json')))
                if update==done or (update+1)%5==0 or update+1==len(windows):
                    self.phase('train',arm=arm,completed=update+1,total=len(windows),**values)
        atomic_save(p/'final.pt',dict(arm=arm,
            decoder={n:cpu_tree(v) for n,v in self.model.state_dict().items() if n in self.trainable},
            completed_updates=len(windows),manifest_sha256=sha(OUT/'manifest.json')))
        # Ensure no encoder / shared embedding / output-head change occurred.
        for n,v in self.model.state_dict().items():
            if n not in self.trainable:
                assert torch.equal(v.detach().cpu(),self.parent[n]), n
        write(p/'training_summary.json',dict(state='COMPLETED',updates=len(windows),epochs=self.cfg['epochs'],
            training_rows=len(self.fit),non_decoder_parameters_unchanged=True,checkpoint_sha256=sha(p/'final.pt')))
        del opt,sched;gc.collect();torch.cuda.empty_cache()

    def evaluate(self, name, input_arm):
        path=self.folder/f'predictions_{name}.jsonl'
        saved=rows(path) if path.exists() else []
        assert len(saved)<=len(self.dev)
        for record,j in zip(saved,self.dev):
            assert record['user_id']==self.samples[j]['user_id'] and record['sample_index']==j
            assert record['rank']==self.rank(record['ids'],record['scores'],self.samples[j])
        buffer=[]
        with path.open('a') as log:
            for j in self.dev[len(saved):]:
                record=self.generate(j,input_arm);saved.append(record);buffer.append(record)
                if len(buffer)==64 or len(saved)==len(self.dev):
                    log.writelines(json.dumps(r)+'\n' for r in buffer);log.flush();os.fsync(log.fileno());buffer=[]
                if len(saved)%32==0 or len(saved)==1:
                    self.phase('evaluate',arm=name,completed=len(saved),total=len(self.dev))
        rr=np.array([r['rank'] for r in saved])
        write(self.folder/f'metrics_{name}.json',dict(users=len(saved),metrics=metrics(rr),
              prediction_sha256=sha(path),input_arm=input_arm,output_score_correction=False))
        return rr

    def run(self):
        self.profile()
        for arm in ARMS:
            self.train(arm)
            self.evaluate(arm,arm)
            if arm=='semantic':
                self.evaluate('semantic_without_feedback','control')
                self.evaluate('semantic_wrong_evidence','shuffled')
        values={a:np.array([r['rank'] for r in rows(self.folder/f'predictions_{a}.jsonl')])
                for a in (*ARMS,'semantic_without_feedback','semantic_wrong_evidence')}
        for name,key in (('GRAM','native'),('output_guided','guided')):
            values[name]=np.array([self.rank(self.reference[self.samples[j]['user_id']][key]['ids'],
                self.reference[self.samples[j]['user_id']][key]['scores'],self.samples[j]) for j in self.dev])
        pairs=[('semantic',a) for a in ('GRAM','control','numeric','shuffled','output_guided',
                                      'semantic_without_feedback','semantic_wrong_evidence')]
        pairs += [('numeric','control'),('control','GRAM')]
        result=dict(state='COMPLETED',users=512,metrics={a:metrics(rr) for a,rr in values.items()},
            comparisons={a+'_minus_'+b:contrast(values[a],values[b],self.cfg['bootstrap_repeats'],
                                                self.cfg['bootstrap_seed']) for a,b in pairs},
            single_upstream_seed=True,validation_only=True,test_read=False,output_score_correction=False,
            frozen_reference_comparison_not_equal_information_or_training=True,
            seconds=time.monotonic()-self.started)
        np.savez_compressed(self.folder/'ranks.npz',**values,users=np.asarray([self.samples[j]['user_id'] for j in self.dev]))
        write(self.folder/'summary.json',result)
        status(self.folder,'COMPLETED',users=512)


def worker(domain):
    p=OUT/domain;lease=None;log=None;stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            write(p/'heartbeat.json',dict(pid=os.getpid(),updated_at=stamp(),liveness_only=True))
            stop.wait(30)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    def interrupted(*_):
        raise InterruptedError('Worker terminated')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        status(p,'RUNNING',stage='verify_inputs');verify_manifest();cfg=read(OUT/'config.json')
        lease_path=p/f'gpu_lease_{os.getpid()}.json';log=(p/f'gpu_lease_{os.getpid()}.log').open('x')
        lease=subprocess.Popen([sys.executable,'experiment/gpu_memory_lease.py','--gpu','0',
            '--total-lease-mib',str(cfg['total_lease_mib']),'--expected-workload-peak-mib',str(cfg['expected_workload_peak_mib']),
            '--status-path',str(lease_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(p/'lease_launch.json',dict(identity=process_identity(lease.pid),status_path=str(lease_path),
            gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        deadline=time.monotonic()+180
        while True:
            assert lease.poll() is None,'Lease sidecar failed'
            if lease_path.exists() and read(lease_path).get('state')=='holding':break
            assert time.monotonic()<deadline,'Lease admission timed out'
            time.sleep(1)
        Worker(domain,lease).run()
    except BaseException as error:
        (p/f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        status(p,'INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=repr(error))
        raise
    finally:
        if lease is not None and lease.poll() is None:
            lease.terminate()
            try:lease.wait(timeout=10)
            except subprocess.TimeoutExpired:lease.kill();lease.wait()
        if log is not None:log.close()
        stop.set();thread.join(timeout=2)


def complete(domain):
    return (OUT/domain/'summary.json').exists() and read(OUT/domain/'status.json')['state']=='COMPLETED'


def write_report():
    state=read(OUT/'status.json')
    lines=['# 专家意见前置与语义反馈','','Origin Skill：academic-research-suite / experiment-agent。','',
        f"状态：{state['state']}；更新时间：{stamp()}。",'',
        '原历史输入保留；专家卡片先经冻结文本编码器，再由GRAM解码器读取并自行打分。新分支不修改输出分数。',
        '四组同等解码器微调：无意见control、候选分数numeric、语义证据semantic、错配证据shuffled。每域512人，单种子开发筛查；无test。','',
        '| 域 | 状态 | 阶段 | 组 | 完成 | 总量 |','|---|---|---|---|---:|---:|']
    for d in DOMAINS:
        s=read(OUT/d/'status.json')
        lines.append(f"| {d} | {s['state']} | {s.get('stage','等待')} | {s.get('arm','—')} | {s.get('completed',0)} | {s.get('total',0)} |")
        if (OUT/d/'summary.json').exists():
            result=read(OUT/d/'summary.json')
            lines += ['',f'{d} 完整结果：','','| 方法 | NDCG@10 | H10 | H50 |','|---|---:|---:|---:|']
            for name,m in result['metrics'].items():
                lines.append(f"| {name} | {m['N10']:.6f} | {m['H10']:.6f} | {m['H50']:.6f} |")
            for other in ('control','numeric','shuffled','output_guided'):
                c=result['comparisons']['semantic_minus_'+other]
                lines += ['',f"semantic相对{other}：N10绝对差{c['delta_N10']:+.6f}；描述性95%区间{c['descriptive_95pct_N10']}；前十新增/丢失{c['top10_gained']}/{c['top10_lost']}。"]
    lines += ['','原GRAM/旧输出引导复用已核验的同512人预测；新四组及semantic关意见/错意见干预均真实生成。',
        'numeric与semantic同时改变信息量和长度；与旧输出引导比较同时改变训练与专家信息压缩，不能归因于单一格式或单一接入位置。',
        '没有显著提升不能否定所有语义反馈设计；512人单种子区间不覆盖训练随机性或研究选择。','',
        '[执行协议](../../plan/第二十一阶段/21.6_专家意见前置与语义反馈_20260924.md)；'
        '[实时状态](../../artifacts/phase21/expert_feedback_20260924_v1/status.json)。']
    temp=REPORT.with_name(REPORT.name+f'.{os.getpid()}.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(REPORT)


def supervise():
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status(OUT,'VERIFYING_INPUTS');verify_manifest();cfg=read(OUT/'config.json')
    active={};children={};blocked={};started=time.monotonic()
    for path in OUT.glob('launch_*.json'):
        rec=read(path)
        if identity_alive(rec['identity']):
            assert rec['domain'] not in active
            active[rec['domain']]=rec
    for d in DOMAINS:
        s=read(OUT/d/'status.json')
        if d not in active and not complete(d) and s['state'] in ('FAILED','RESOURCE_TIME_LIMIT'):
            blocked[d]=s
    while True:
        for d,rec in list(active.items()):
            if d in children:children[d].poll()
            if identity_alive(rec['identity']) and time.time()-rec['started_wall']>cfg['worker_timeout_seconds']:
                os.killpg(rec['identity']['pid'],signal.SIGKILL if rec.get('terminating') else signal.SIGTERM)
                rec['terminating']=True;blocked[d]=dict(state='RESOURCE_TIME_LIMIT',reason='Actual 12h worker limit')
            if not identity_alive(rec['identity']):
                del active[d]
                if not complete(d):blocked.setdefault(d,read(OUT/d/'status.json'))
        if time.monotonic()-started>cfg['supervisor_timeout_seconds']:
            for rec in active.values():
                if identity_alive(rec['identity']):os.killpg(rec['identity']['pid'],signal.SIGTERM)
            status(OUT,'RESOURCE_TIME_LIMIT',reason='Actual 24h supervisor limit');write_report();return
        pending=[d for d in DOMAINS if d not in active and d not in blocked and not complete(d)]
        error=None
        if pending:
            try:
                output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=30)
                free={row.split(',')[0].strip():int(row.split(',')[1]) for row in output.splitlines()}
            except (subprocess.SubprocessError,ValueError) as exc:
                free={};error=repr(exc)
            for d in pending:
                occupied={r['gpu'] for r in active.values()}
                candidates=[cfg['gpus'][d]]+cfg['fallback_gpus']
                candidates=[g for g in candidates if g not in occupied and free.get(g,0)>=cfg['minimum_free_mib']]
                if not candidates or len(active)>=cfg['max_parallel']:continue
                gpu=candidates[0]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
                path=OUT/d/f'worker_{int(time.time())}.log'
                with path.open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',d],cwd=ROOT,
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rec=dict(domain=d,gpu=gpu,identity=process_identity(child.pid),started_wall=time.time(),log=str(path.relative_to(ROOT)))
                assert rec['identity'];write(OUT/f'launch_{d}_{child.pid}.json',rec)
                active[d]=rec;children[d]=child
        pending=[d for d in DOMAINS if d not in active and d not in blocked and not complete(d)]
        if all(complete(d) for d in DOMAINS):
            state='COMPLETED';write(OUT/'summary.json',dict(state=state,domains={d:read(OUT/d/'summary.json') for d in DOMAINS}))
        elif not active and not pending:state='FAILED'
        else:state='RUNNING' if active else 'WAITING_FOR_RESOURCES'
        status(OUT,state,active=active,pending=pending,blocked=blocked,resource_query_error=error,
               domains={d:read(OUT/d/'status.json') for d in DOMAINS},auto_start_after_reboot=False)
        write_report()
        if state in ('COMPLETED','FAILED'):return
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','supervise','resume','worker'))
    parser.add_argument('--domain',choices=DOMAINS)
    args=parser.parse_args()
    if args.action=='prepare':
        try:prepare()
        except Exception as error:
            if OUT.exists():
                status(OUT,'PREPARATION_FAILED',error=repr(error))
                (OUT/'preparation_failure.txt').write_text(traceback.format_exc())
            raise
    elif args.action=='worker':
        assert args.domain;worker(args.domain)
    else:
        try:supervise()
        except Exception as error:
            status(OUT,'FAILED',error=repr(error));(OUT/f'controller_failure_{os.getpid()}.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    main()
