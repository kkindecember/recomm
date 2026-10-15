"""One-epoch matched input adaptation, two domains, bounded overnight run."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
from experiment.phase21.attribute_retention_core import ARMS, pack_catalog, install_bank
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / os.environ.get('RECOMM_ATTRIBUTE_OUT', 'artifacts/phase21/attribute_retention_20260923_v1')
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
DESIGN = ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.1B_属性保留过夜匹配实验_20260923.md'
REPORT = ROOT / os.environ.get('RECOMM_ATTRIBUTE_REPORT', 'report/第二十一阶段/属性保留过夜实验_20260923.md')
MODULE = 'experiment.phase21.attribute_retention'
GPUS = {'Toys': 'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2',
        'Beauty': 'GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253'}
if os.environ.get('RECOMM_ATTRIBUTE_DOMAINS'):
    selected_domains = os.environ['RECOMM_ATTRIBUTE_DOMAINS'].split(',')
    assert selected_domains and set(selected_domains) <= set(GPUS)
    GPUS = {d: GPUS[d] for d in selected_domains}


def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read(p): return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f: return [json.loads(line) for line in f if line.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(p, value):
    p = Path(p); tmp = p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); tmp.replace(p)


def prepare():
    import torch
    from transformers import AutoTokenizer
    from experiment.phase20.lime_clean_gram_core import T5
    from experiment.phase20.lime_clean_upstream_core import runtime_args, validate_rows
    from experiment.phase18.core.diff_data import DiffDataset
    sys.path.insert(0, str(ROOT/'GRAM/src'))
    from processor import CollatorGRAM
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    sources = [PLAN, Path(__file__), ROOT/'experiment/phase21/attribute_retention_core.py',
               ROOT/'experiment/phase21/test_attribute_retention.py',
               ROOT/'experiment/phase21/run_attribute_retention.sh',
               ROOT/'experiment/phase20/lime_clean_gram_core.py',
               ROOT/'experiment/phase20/lime_clean_upstream_core.py',
               ROOT/'experiment/phase20/lime_parallel_core.py',
               ROOT/'experiment/phase18/core/diff_data.py',
               ROOT/'experiment/phase20/catalog_scope_core.py']
    sources += list((ROOT/'GRAM/src').rglob('*.py'))
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for p in sources:
        dest = OUT/'sources'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    shutil.copy2(PLAN, OUT/'protocol.md')
    tokenizer = AutoTokenizer.from_pretrained(str(T5), local_files_only=True)
    native = CollatorGRAM(tokenizer, runtime_args())
    design_receipt = read(DESIGN/'summary.json')
    old_manifest = read(SOURCE/'manifest.json')['sha256']
    details = {}
    for domain in GPUS:
        folder = OUT/domain; folder.mkdir()
        write(folder/'status.json', dict(state='PREPARING', domain=domain, updated_at=stamp()))
        used = {}

        def use(p, expected=None):
            relative = str(p.relative_to(ROOT)); assert 'test.json' not in relative and '/test/' not in relative
            value = sha(p)
            expected = expected or old_manifest.get(relative)
            if expected: assert value == expected, relative
            used[relative] = value
            return p

        catalog = read(use(SOURCE/domain/'catalog.json'))
        cfg = read(use(SOURCE/domain/'data_config.json'))
        use(ROOT/cfg['parent_checkpoint'], cfg['parent_sha256'])
        training_summary = read(use((ROOT/cfg['parent_checkpoint']).parent/'summary.json'))
        assert training_summary['state'] == 'COMPLETED' and training_summary['epochs'] == 30
        for name in ('upstream_train_without_terminal.jsonl', 'upstream_sequences_without_terminal.jsonl', 'terminal_holdout.jsonl'):
            path = DESIGN/domain/name
            use(path, design_receipt['output_sha256'][str(path.relative_to(ROOT))])
        train = rows(DESIGN/domain/'upstream_train_without_terminal.jsonl')
        sequences = {r['user_id']: r['history'] for r in rows(DESIGN/domain/'upstream_sequences_without_terminal.jsonl')}
        assert validate_rows(train, sequences) == {'Toys':105867, 'Beauty':127328}[domain]
        held = rows(DESIGN/domain/'terminal_holdout.jsonl')
        assert all(r['history'] == sequences[r['user_id']][-20:] for r in held)
        old = read(use(SOURCE/domain/'samples.json'))
        known = [dict(s, cohort='existing512') for s in old if s['role'] == 'development']
        query = np.load(use(ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz'))
        eligible = set(map(str, query['users'][query['evaluation_indices']]))
        eligible -= {r['user_id'] for r in held} | {r['user_id'] for r in known}
        chosen = sorted(eligible, key=lambda u: hashlib.sha256(f'20260923|attribute-overnight|{domain}|{u}'.encode()).digest())[:512]
        old_cfg = read(use(ROOT/f'experiment/phase18/config/s18_diff_gram_{domain.lower()}.json'))
        validation = {r['user_id']: r for r in rows(use(ROOT/old_cfg['prepared']/'validation.jsonl'))}
        selected = known+[dict(validation[u], cohort='additional512') for u in chosen]
        assert len(selected) == len({r['user_id'] for r in selected}) == 1024
        assert not {r['user_id'] for r in held} & {r['user_id'] for r in selected}
        assert all(r['history'] == sequences[r['user_id']][-20:] for r in selected)
        write(folder/'samples.json', selected); write(folder/'data_config.json', cfg)
        banks, lengths, donors, audit = pack_catalog(catalog, tokenizer, domain)
        assert audit['usable_descriptions'] >= 1000 and audit['placebo_changed_items'] >= 1000
        for arm in ARMS:
            np.savez_compressed(folder/f'tokens_{arm}.npz', tokens=banks[arm], lengths=lengths[arm])
        np.save(folder/'description_donors.npy', donors)
        cached = install_bank(native, catalog, banks['original'], lengths['original'])
        dataset = DiffDataset(train, catalog)
        # Include longest and deterministic scattered training rows; no evaluation outcomes.
        check_ids = [0, len(train)//2, len(train)-1, max(range(len(train)), key=lambda i:len(train[i]['history']))]
        for i in check_ids:
            a, b = native([dataset[i]]), cached([dataset[i]])
            for key in ('item_text_ids', 'item_text_masks', 'target_ids', 'history_item_ids'):
                assert torch.equal(a[key], b[key]), (domain, i, key)
        sample_items = sorted(range(1,len(catalog['items'])), key=lambda i:hashlib.sha256(f'attribute-parity|{domain}|{i}'.encode()).digest())[:128]
        for i in sample_items:
            actual, mask = native.encode_texts_split([[catalog['passages'][i]]], tokenizer)
            n = int(mask[0,0].sum())
            np.testing.assert_array_equal(actual[0,0,:n].numpy(), banks['original'][i,:lengths['original'][i]])
        audit['native_collator_exact'] = True; audit['parity_training_rows'] = check_ids; audit['parity_items'] = sample_items
        write(folder/'packing_audit.json', audit)
        for p in list(folder.glob('tokens_*.npz'))+[folder/'samples.json', folder/'data_config.json', folder/'description_donors.npy']:
            used[str(p.relative_to(ROOT))] = sha(p)
        write(folder/'input_manifest.json', dict(sha256=used, test_read=False, development_labels_used_for_fit=False))
        details[domain] = dict(training_rows=len(train), updates_per_arm=math.ceil(len(train)/128), evaluation_users=1024,
                               usable_descriptions=audit['usable_descriptions'], placebo_changed_items=audit['placebo_changed_items'])
        write(folder/'status.json', dict(state='PREPARED', updated_at=stamp(), **details[domain]))
        print(json.dumps(dict(domain=domain, **details[domain])), flush=True)
    write(OUT/'source_manifest.json', dict(sha256=manifest))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), domains=details, test_read=False))


class Worker:
    def __init__(self, domain):
        import torch
        from experiment.phase20.lime_clean_gram_core import seed_all, T5
        from experiment.phase20.lime_clean_upstream_core import new_gram, runtime_args
        from experiment.phase18.core.diff_data import DiffDataset
        sys.path.insert(0, str(ROOT/'GRAM/src'))
        from processor import CollatorGRAM
        from utils import generation_trie
        self.domain = domain; self.folder = OUT/domain; self.started = time.monotonic(); self.stage = 'load'
        self.arm = None; self.progress = 0; self.total = 0; self.status()
        for receipt in (OUT/'source_manifest.json', self.folder/'input_manifest.json'):
            for relative, digest in read(receipt)['sha256'].items(): assert sha(ROOT/relative) == digest, relative
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(25600/(torch.cuda.get_device_properties(0).total_memory/2**20))
        self.device = torch.device('cuda')
        seed_all(2023)
        self.catalog = read(SOURCE/domain/'catalog.json')
        self.cfg = read(self.folder/'data_config.json')
        self.samples = read(self.folder/'samples.json')
        self.train_rows = rows(DESIGN/domain/'upstream_train_without_terminal.jsonl')
        self.dataset = DiffDataset(self.train_rows, self.catalog)
        self.model, self.tokenizer = new_gram(T5)
        self.model.set_checkpoint(False)
        saved = torch.load(ROOT/self.cfg['parent_checkpoint'], map_location='cpu')
        assert saved['position']['epoch'] == 31
        self.parent = saved['state_dict']; del saved
        self.model.load_state_dict(self.parent, strict=True); self.model.to(self.device)
        self.native = CollatorGRAM(self.tokenizer, runtime_args())
        self.banks = {arm: dict(np.load(self.folder/f'tokens_{arm}.npz')) for arm in ARMS}
        self.path_to_id = {tuple(path):i for i,path in enumerate(self.catalog['decoder_paths'],1)}
        trie = generation_trie.Trie(self.catalog['decoder_paths'])
        self.prefix_fn = generation_trie.prefix_allowed_tokens_fn(trie)
        self.max_length = max(map(len,self.catalog['decoder_paths']))
        self.config = dict(batch_size=16, gradient_accumulation=8, encoder_checkpointing=False,
                           gram_learning_rate=.0001, weight_decay=.01, warmup_fraction=.05, gradient_clip_norm=1.)
        from experiment.phase20.lime_clean_gram_core import epoch_order, update_windows
        self.windows = list(update_windows(epoch_order(len(self.dataset),2023,1),16,8))
        self.total = len(self.windows)

    def status(self, state='RUNNING', **kw):
        write(self.folder/'status.json', dict(state=state, domain=self.domain, stage=self.stage, arm=self.arm,
              completed=self.progress, total=self.total, pid=os.getpid(), updated_at=stamp(),
              elapsed_seconds=time.monotonic()-self.started, test_read=False, **kw))

    def reset(self, arm):
        import torch
        from experiment.phase20.lime_clean_gram_core import seed_all
        self.model.load_state_dict(self.parent, strict=True)
        self.model.requires_grad_(True); self.model.zero_grad(set_to_none=True)
        seed_all(2023)
        bank = self.banks[arm]
        self.collator = install_bank(self.native,self.catalog,bank['tokens'],bank['lengths'])
        torch.cuda.empty_cache()

    def profile(self):
        import torch
        from experiment.phase20.lime_clean_gram_core import make_optimizer, update_windows
        from experiment.phase20.lime_parallel_core import parallel_update
        self.stage = 'gpu_profile'; self.arm = 'original'; self.progress = 0; self.status()
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        self.reset('original')
        opt, sch = make_optimizer(self.model,'gram',self.config,len(self.windows))
        longest=sorted(range(len(self.dataset)),key=lambda i:len(self.train_rows[i]['history']),reverse=True)[:128]
        stress=list(update_windows(longest,16,8))[0]
        record=[]
        for i,batches in enumerate([stress]+self.windows[:3]):
            started=time.monotonic()
            r=parallel_update(self.model,opt,sch,self.collator,self.dataset,batches,'gram',self.config,self.device,stress=i==0)
            torch.cuda.synchronize(); r.update(seconds=time.monotonic()-started,stress=i==0)
            record.append(r); self.progress=i+1; self.status(profile_updates=i+1)
        estimate=float(np.median([r['seconds'] for r in record if not r['stress']]))*len(self.windows)*3+3600
        result=dict(records=record, projected_seconds=estimate,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, optimizer_updates_discarded=4,
                    model_reset_before_formal=True)
        write(self.folder/'resource_profile.json',result)
        del opt,sch; self.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if estimate>6.5*3600: raise RuntimeError(f'RESOURCE_BUDGET_STOP: projection {estimate/3600:.2f} h exceeds 6.5 h')
        self.reset('original')

    def evaluate(self, arm):
        import torch
        from experiment.phase18.core.diff_data import DiffDataset
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.model.eval().requires_grad_(False)
        self.stage='generate'; self.arm=arm; self.progress=0; self.total=len(self.samples); self.status()
        data=DiffDataset(self.samples,self.catalog)
        records=[]
        old=rows(SOURCE/self.domain/'candidates.jsonl') if arm=='parent' else []
        old={r['user_id']:r for r in old if r['role']=='development'}
        with (self.folder/f'predictions_{arm}.jsonl').open('x') as stream:
            for i,row in enumerate(self.samples):
                item=dict(data[i],output=self.catalog['lexical_ids'][1])
                batch=self.collator([item])
                with torch.no_grad():
                    output=self.model.generate(input_ids=batch['item_text_ids'].to(self.device),
                        attention_mask=batch['item_text_masks'].to(self.device),max_length=self.max_length,
                        prefix_allowed_tokens_fn=self.prefix_fn,num_beams=50,num_return_sequences=50,
                        return_dict_in_generate=True,output_scores=True,length_penalty=1.)
                paths=[p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
                ids=[self.path_to_id[tuple(p)] for p in paths]
                scores=output.sequences_scores.detach().cpu().double().numpy()
                assert len(set(ids))==50 and np.isfinite(scores).all()
                if arm=='parent' and i<3:
                    assert ids==old[row['user_id']]['ids']
                    np.testing.assert_allclose(scores,old[row['user_id']]['native'],atol=2e-6,rtol=1e-5)
                history=set(row['history'])
                ordered=[int(ids[k]) for k in np.lexsort((ids,-scores)) if ids[k] not in history]
                rank=ordered.index(row['target'])+1 if row['target'] in ordered else 2**30
                record=dict(user_id=row['user_id'],cohort=row['cohort'],target=row['target'],ids=ids,
                            scores=scores.tolist(),rank=rank)
                stream.write(json.dumps(record)+'\n'); stream.flush(); records.append(record)
                del output,batch
                self.progress=i+1
                if i%25==0 or self.progress==len(data): self.status()
        write(self.folder/f'metrics_{arm}.json',metrics(np.array([r['rank'] for r in records])))

    def train(self, arm):
        import torch
        from experiment.phase20.lime_clean_gram_core import make_optimizer, atomic_save, cpu_tree, rng_state
        from experiment.phase20.lime_parallel_core import parallel_update
        self.reset(arm); self.stage='train'; self.arm=arm; self.progress=0; self.total=len(self.windows); self.status()
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        opt,sch=make_optimizer(self.model,'gram',self.config,len(self.windows))
        folder=self.folder/arm; folder.mkdir()
        def save():
            atomic_save(folder/'latest.pt',dict(state_dict=cpu_tree(self.model.state_dict()),optimizer=cpu_tree(opt.state_dict()),
                scheduler=sch.state_dict(),rng=rng_state(),completed_updates=self.progress,domain=self.domain,arm=arm,
                parent_sha256=self.cfg['parent_sha256'],training_config=self.config,
                source_manifest_sha256=sha(OUT/'source_manifest.json'),input_manifest_sha256=sha(self.folder/'input_manifest.json')))
        with (folder/'updates.jsonl').open('x') as log:
            for i,batches in enumerate(self.windows):
                begin=time.monotonic()
                r=parallel_update(self.model,opt,sch,self.collator,self.dataset,batches,'gram',self.config,self.device)
                torch.cuda.synchronize(); self.progress=i+1
                r.update(update=i+1,seconds=time.monotonic()-begin,lr=opt.param_groups[0]['lr'])
                log.write(json.dumps(r)+'\n'); log.flush()
                if self.progress%100==0 or self.progress==len(self.windows): save()
                if self.progress%10==0 or i==0 or self.progress==len(self.windows):
                    self.status(loss=r['loss'],last_update_seconds=r['seconds'],
                                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        assert sch.last_epoch==len(self.windows)
        atomic_save(folder/'final.pt',dict(state_dict=cpu_tree(self.model.state_dict()),domain=self.domain,arm=arm,
            completed_updates=self.progress,parent_sha256=self.cfg['parent_sha256'],training_config=self.config))
        write(folder/'training_summary.json',dict(state='COMPLETED',training_rows=len(self.dataset),
            updates=len(self.windows),checkpoint_sha256=sha(folder/'final.pt'),completed_at=stamp()))
        del opt,sch; gc.collect(); torch.cuda.empty_cache()

    def run(self):
        self.profile()
        self.evaluate('parent')
        for arm in ARMS:
            self.train(arm)
            self.evaluate(arm)
        assert sha(ROOT/self.cfg['parent_checkpoint'])==self.cfg['parent_sha256']
        self.stage='finished'; self.status(state='COMPLETED')


def worker(domain):
    folder=OUT/domain
    def alarm(*_): raise TimeoutError('Seven-hour domain hard timeout')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(7*3600)
    try:
        Worker(domain).run()
    except BaseException as error:
        (folder/'failure.txt').write_text(traceback.format_exc())
        stopped = 'RESOURCE_BUDGET_STOP' in str(error)
        write(folder/'status.json',dict(state='RESOURCE_BUDGET_STOP' if stopped else 'FAILED',domain=domain,
              updated_at=stamp(),error=repr(error),pid=os.getpid(),scientific_evaluation='NOT_RUN' if stopped else 'INCOMPLETE'))
        raise
    finally: signal.alarm(0)


def finish(states):
    completed=all(s['state']=='COMPLETED' for s in states.values())
    resource_stopped=all(s['state']=='RESOURCE_BUDGET_STOP' or 'RESOURCE_BUDGET_STOP' in s.get('error','') for s in states.values())
    result=dict(state='COMPLETED' if completed else ('RESOURCE_BUDGET_STOP' if resource_stopped else 'PARTIAL_OR_FAILED'),
                created_at=stamp(),domains={},states=states,test_read=False)
    lines=['# 第21阶段：属性保留过夜实验','', '## Material Passport','',
           f'- 状态：{result["state"]}；更新时间：{stamp()}。',
               '- 每个已安排的域做三臂匹配一轮适配；固定终点真实beam50生成；未读取test。',
           '- Origin Skill：academic-research-suite，内联执行。','',
           '| 域/集合 | 方法 | NDCG@10 | H10 | H50 |','|---|---|---:|---:|---:|']
    for domain,s in states.items():
        if s['state']!='COMPLETED': continue
        folder=OUT/domain; samples=read(folder/'samples.json'); tables={}; ranks_by_arm={}; checks=0
        for arm in ('parent',)+ARMS:
            predictions=rows(folder/f'predictions_{arm}.jsonl')
            assert len(predictions)==len(samples)==1024
            rr=[]
            for row,sample in zip(predictions,samples):
                assert row['user_id']==sample['user_id'] and row['target']==sample['target']
                legal=[(score,item) for item,score in zip(row['ids'],row['scores']) if item not in sample['history']]
                order=[item for score,item in sorted(legal,key=lambda p:(-p[0],p[1]))]
                independent=next((i+1 for i,item in enumerate(order) if item==sample['target']),2**30)
                assert independent==row['rank']; rr.append(independent); checks+=1
            ranks_by_arm[arm]=np.array(rr)
        for cohort in ('combined1024','existing512','additional512'):
            ix=np.array([cohort=='combined1024' or s['cohort']==cohort for s in samples])
            table={arm:metrics(rr[ix]) for arm,rr in ranks_by_arm.items()}
            comparisons={arm:contrast(ranks_by_arm['retained'][ix],ranks_by_arm[arm][ix]) for arm in ('parent','original','shuffled')}
            tables[cohort]=dict(users=int(ix.sum()),metrics=table,comparisons=comparisons)
            for arm,m in table.items(): lines.append(f"| {domain}/{cohort} | {arm} | {m['N10']:.6f} | {m['H10']:.6f} | {m['H50']:.6f} |")
        m=tables['combined1024']['metrics']
        gate=bool(m['retained']['N10']>=1.01*m['original']['N10'] and m['retained']['H10']>=m['original']['H10']
                  and all(tables[c]['metrics']['retained']['N10']>=tables[c]['metrics']['original']['N10'] for c in ('existing512','additional512')))
        description=bool(gate and m['retained']['N10']>m['shuffled']['N10'] and m['retained']['H10']>=m['shuffled']['H10'])
        result['domains'][domain]=dict(groups=tables,layout_pass=gate,description_support=description,
                                       independently_recomputed_rankings=checks)
    result['layout_pass_both']=all(d['layout_pass'] for d in result['domains'].values()) if len(result['domains'])==2 else None
    result['description_support_both']=all(d['description_support'] for d in result['domains'].values()) if len(result['domains'])==2 else None
    result['scientific_evaluation']='COMPLETE' if completed else ('NOT_RUN' if not result['domains'] else 'PARTIAL')
    lines += ['', '## 投入判断','',f"双域布局门槛：{result['layout_pass_both']}；双域描述对应支持：{result['description_support_both']}。",'',
              '门槛仅约束本次短程适配。若未过，不自动增加训练轮数或学习率搜索；单域通过只记录单域信号。', '',
              '## 限制与复核','',
              '- retained相对original同时改变协同提示和属性的预算分配；不能将全部变化归于描述。retained/shuffled只有描述对应不同，长度和非描述词元逐值相同。',
              '- 每臂从相同干净父模型重置并遍历全部允许训练样本一次；额外训练由original控制。仅一个上游seed及一个适配seed。',
              '- 新增512人仍来自研究开发池，不是独立最终测试。用户配对区间为固定预测条件描述，不包含训练随机性或研究选择。',
              '- 完成域的全部输出由另一种排序写法复算；此为同一助手的代码路径核验，不是外部审稿。',
              '- 完整区间、增益/误伤和资源数据见产物summary.json、各域status.json和resource_profile.json。', '',
              '冻结计划：[21.1B属性保留](../../plan/第二十一阶段/21.1B_属性保留过夜匹配实验_20260923.md)。',
              '产物：[attribute_retention_20260923_v1](../../artifacts/phase21/attribute_retention_20260923_v1/)。','']
    if result['state']!='COMPLETED':
        lines += ['## 未完成项','',json.dumps(states,ensure_ascii=False,indent=2),'']
    if resource_stopped and not result['domains']:
        lines=['# 第21阶段：属性保留过夜实验','', '## Material Passport','',
               f'- 状态：RESOURCE_BUDGET_STOP；更新时间：{stamp()}。',
               '- 科学评价：NOT_RUN。完成输入准备及资源试跑，未开始正式适配或生成评价。',
               '- Origin Skill：academic-research-suite，内联执行。','',
               '**没有新增推荐效果结果，不能据此判断属性方案有效或无效。**','',
               '两域各完成4次试跑优化更新（最长历史压力检查1次、普通训练窗口3次）；这些试跑参数随进程退出丢弃。',
               '损失和梯度有限，显存检查通过；停止原因是吞吐外推超出冻结的6.5小时投入预算。', '',
               '| 域 | 随机窗口中位秒/更新 | 峰值reserved显存 MiB | 预计三臂总耗时 |',
               '|---|---:|---:|---:|']
        profiles={}
        for domain in states:
            profile=read(OUT/domain/'resource_profile.json'); profiles[domain]=profile
            median=float(np.median([r['seconds'] for r in profile['records'] if not r['stress']]))
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in profile['records'])
            lines.append(f"| {domain} | {median:.2f} | {profile['peak_reserved_mib']:.0f} | {profile['projected_seconds']/3600:.2f}小时 |")
        result.update(formal_training_started=False,new_generation=False,
                      probe_optimizer_updates=sum(len(p['records']) for p in profiles.values()),
                      probe_parameters_discarded=True)
        lines += ['', '外推为普通窗口中位耗时×每臂更新数×3，再预留1小时生成评价；不是实际消耗了10–11小时。',
                  '最初3–6小时的估计不准确。任务在00:59停止，01:00完成自动收尾；没有偷偷减少样本、轮数或对照。', '',
                  '输入准备、预算修订、词元一致性和描述错配检查已保留，可供后续重新安排。当前没有任何正式训练checkpoint或预测文件。',
                  '原自动报告错误地将“未评价”的门槛写为False；已改为null/NOT_RUN，旧报告与状态在postmortem_original/保留。', '',
                  '排期教训：有截止时间的任务应先获得实际资源吞吐，再决定同时开展的域和对照数量，并在启动前写好不依赖交互监控的资源不足分支。', '',
                  '计划：[冻结协议](../../plan/第二十一阶段/21.1B_属性保留过夜匹配实验_20260923.md)。',
                  '产物：[全部准备及试跑记录](../../artifacts/phase21/attribute_retention_20260923_v1/)。','']
    write(OUT/'summary.json',result); REPORT.write_text('\n'.join(lines)); (OUT/'report.md').write_text('\n'.join(lines))
    return result


def supervise():
    started=time.monotonic(); processes={}; waiting_since=started; stopped=set()
    while len(processes)+len(stopped)<len(GPUS):
        output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        free={line.split(',')[0].strip():int(line.split(',')[1]) for line in output.splitlines()}
        for domain,gpu in GPUS.items():
            if domain in processes or domain in stopped: continue
            if free[gpu]>=30720:
                env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu
                with (OUT/domain/'worker.log').open('x') as log:
                    processes[domain]=subprocess.Popen([sys.executable,'-m',MODULE,'worker','--domain',domain],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            elif time.monotonic()-waiting_since>1800:
                stopped.add(domain); write(OUT/domain/'status.json',dict(state='RESOURCE_WAIT_EXPIRED',updated_at=stamp(),gpu=gpu))
        write(OUT/'status.json',dict(state='RUNNING',stage='launch_workers',updated_at=stamp(),
             workers={d:p.pid for d,p in processes.items()},supervisor_pid=os.getpid()))
        if len(processes)+len(stopped)<len(GPUS): time.sleep(15)
    while any(p.poll() is None for p in processes.values()):
        if time.monotonic()-started>8*3600:
            for d,p in processes.items():
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    write(OUT/d/'status.json',dict(state='TIMED_OUT',updated_at=stamp(),error='8 hour supervisor timeout'))
            break
        states={d:read(OUT/d/'status.json') for d in GPUS}
        write(OUT/'status.json',dict(state='RUNNING',updated_at=stamp(),supervisor_pid=os.getpid(),
             workers={d:p.pid for d,p in processes.items()},elapsed_seconds=time.monotonic()-started,domains=states))
        time.sleep(15)
    states={d:read(OUT/d/'status.json') for d in GPUS}
    for domain,p in processes.items():
        if p.returncode!=0 and states[domain]['state'] not in ('FAILED','TIMED_OUT','RESOURCE_BUDGET_STOP'):
            states[domain]=dict(state='FAILED',error=f'Process exit {p.returncode}',previous=states[domain])
    result=finish(states)
    write(OUT/'status.json',dict(state=result['state'],updated_at=stamp(),elapsed_seconds=time.monotonic()-started,
          domains=states,layout_pass_both=result['layout_pass_both'],description_support_both=result['description_support_both']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','supervise','worker'));p.add_argument('--domain',choices=tuple(GPUS));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='supervise':supervise()
    else:worker(a.domain)
