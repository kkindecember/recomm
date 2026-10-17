"""Stage the existing expert screen; no new training or generation code.

Read fixed complete prediction prefixes, report paired exploratory metrics,
and stop each original worker at old512 + two new512 batches. The original
worker owns cleanup of its generator and GPU memory lease.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
from experiment.phase21 import old10_expert_selection as B
from experiment.phase20.catalog_scope_core import metrics,contrast

ROOT,RUN=B.ROOT,B.OUT
OUT=ROOT/'artifacts/phase21/old10_expert_screen_staged_20261017_v1'
PLAN=ROOT/'plan/第二十一阶段/21.44_固定原十维专家分批初筛_20261017.md'
DOMAINS=('Toys','Beauty')
CFG=dict(stages=[512,1024],old_users=512,stop_after_complete_rows=1536,
    original_run=str(RUN.relative_to(ROOT)),original_fuser_dimension=10,
    changed_training=False,changed_generation=False,history_filter=False,
    automatic_arm_elimination=False,test_read=False,monitor_seconds=10,
    resource_wait_timeout_seconds=21600,bootstrap_repeats=2000,bootstrap_seed=20261017)
read,write,sha,stamp=B.read,B.write,B.sha,B.stamp
KEYS=['native',*B.ARMS,'legacy_three_guided_reference']


def status(**extra):
    write(OUT/'status.json',dict(updated_at=stamp(),test_read=False,**extra))


def prepare():
    original=read(RUN/'status.json')
    assert original['state']=='RUNNING'
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',CFG)
    (OUT/'protocol.md').write_text(PLAN.read_text())
    manifest={}
    for path in (PLAN,Path(__file__),ROOT/'experiment/phase21/old10_expert_selection.py',
                 ROOT/'experiment/phase20/catalog_scope_core.py',RUN/'config.json',RUN/'manifest.json',OUT/'config.json'):
        manifest[str(path.relative_to(ROOT))]=sha(path)
    jobs={}
    for domain in DOMAINS:
        p=OUT/domain;p.mkdir()
        samples=read(RUN/domain/'samples.json')
        assert len(samples)==4608 and len({s['user_id'] for s in samples})==4608
        assert all(s['cohort']==('old512' if j<512 else 'wide4096') for j,s in enumerate(samples))
        write(p/'samples1536.json',samples[:1536])
        manifest[str((RUN/domain/'samples.json').relative_to(ROOT))]=sha(RUN/domain/'samples.json')
        manifest[str((p/'samples1536.json').relative_to(ROOT))]=sha(p/'samples1536.json')
        current=read(RUN/domain/'status.json')
        jobs[domain]=dict(worker_pid=original['active'][domain]['pid'],generator_pid=current['pid'],
            lease=read(RUN/domain/'lease_process.json'),initial_status=current,
            stop_after_complete_rows=1536)
    write(OUT/'original_jobs.json',jobs)
    manifest[str((OUT/'original_jobs.json').relative_to(ROOT))]=sha(OUT/'original_jobs.json')
    write(OUT/'manifest.json',dict(sha256=manifest))
    write(OUT/'original_root_before.json',original)
    status(state='PREPARED',stages=CFG['stages'],original_jobs=jobs)
    write(RUN/'staged_screen_handoff.json',dict(reason='User requested staged method screening, not code acceleration',
        new_status=str((OUT/'status.json').relative_to(ROOT)),new_protocol=str(PLAN.relative_to(ROOT)),
        original_scope_not_completed=True,changed_model=False,updated_at=stamp()))


def snapshot(domain,new_count):
    needed=512+new_count;samples=read(OUT/domain/'samples1536.json')[:needed]
    ranks={key:[] for key in KEYS};prefix=bytearray()
    with (RUN/domain/'predictions.jsonl').open('rb') as stream:
        for j in range(needed):
            line=stream.readline()
            assert line.endswith(b'\n'),('Incomplete requested prefix',domain,j)
            row=json.loads(line);sample=samples[j]
            assert all(row[k]==sample[k] for k in ('user_id','target','history','cohort'))
            assert row['reused_arms']==(list(B.REUSE) if j<512 else [])
            for key in KEYS:
                value=row[key];rank=B.rank(value,sample['target'])
                ids,scores=np.asarray(value['ids']),np.asarray(value['scores'])
                at=np.flatnonzero(ids==sample['target'])
                independent=2**30 if not len(at) else 1+int(((scores>scores[at[0]])|((scores==scores[at[0]])&(ids<sample['target']))).sum())
                assert rank==independent
                ranks[key].append(rank)
            prefix.extend(line)
    destination=OUT/domain/f'stage{new_count}_predictions.jsonl'
    destination.write_bytes(prefix)
    write(OUT/domain/f'stage{new_count}_snapshot.json',dict(rows=needed,new_users=new_count,
        prefix_bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),all_ranks_independently_verified=True))
    arrays={key:np.asarray(value,np.int64) for key,value in ranks.items()}
    np.savez_compressed(OUT/domain/f'stage{new_count}_ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**arrays)
    return arrays


def analyze(new_count):
    groups=[('old512',np.arange(512)),('new_batch1_512',np.arange(512,1024))]
    if new_count==1024:
        groups += [('new_batch2_512',np.arange(1024,1536)),('new_all1024',np.arange(512,1536))]
    results={}
    pairs=[(arm,'native') for arm in B.ARMS]
    pairs += [(arm,name+'__only') for arm,names in B.PAIRS.items() for name in names]
    pairs += [('SASRec__only',arm) for arm in B.ARMS if arm!='SASRec__only']
    pairs=list(dict.fromkeys(pairs))
    for domain in DOMAINS:
        ranks=snapshot(domain,new_count);results[domain]={}
        for name,ix in groups:
            measured={key:metrics(value[ix]) for key,value in ranks.items()}
            comparisons={a+'_vs_'+b:contrast(ranks[a][ix],ranks[b][ix],repeats=2000,seed=20261017) for a,b in pairs}
            results[domain][name]=dict(users=len(ix),metrics=measured,comparisons=comparisons,
                relative_to_gram_N10_pct={key:100*(value['N10']/measured['native']['N10']-1) for key,value in measured.items()})
    write(OUT/f'stage{new_count}_summary.json',dict(domains=results,new_users_per_domain=new_count,
        comparisons_per_group=len(pairs),test_read=False,automatic_elimination=False,
        statistical_review=dict(coverage='11/11',confidence='Exploratory partial validation',
            simpson='Domains, old512 and two new batches separately retained; no pooled winner.',
            ecological='Mean benefit does not imply every-user benefit; paired gains/losses saved.',
            berkson='Pre-existing fixed hash sample of eligible validation, not raw population.',
            collider='No target- or hit-based sample selection.',base_rate='All fixed prefix users retained.',
            regression_to_mean='Old512 selected repeatedly; second new batch needed before interpreting first-batch means.',
            survivorship='All11 arms included on every analyzed user; later incomplete tail ignored by fixed count.',
            look_elsewhere='All paired comparisons saved; unadjusted descriptive intervals, no confirmatory significance.',
            forking_paths='Scope reduced after observing first-batch results; explicitly exploratory, no automatic early elimination.',
            causation='Contributions conditional on original10 fuser and current frozen decoder, not intrinsic model quality.',
            reverse_causality='No new features, fits or target-driven sampling.')))
    label='首批新增512' if new_count==512 else '两批新增1024'
    report=ROOT/f'report/第二十一阶段/固定原十维专家初筛_{label}_20261017.md'
    lines=[f'# 固定原十维专家初筛：{label}','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED；固定预测前缀独立复算，未训练或重新生成',
        f'- Version Label: staged_old10_stage{new_count}','',
        '原10维451参数联合用户融合器，六组双专家、五组单专家；beam50，不过滤历史。旧512与新增批次分开报告。单专家使用重复输入槽，按相同NLL预算学习alpha。',
        '单专家当前较高只能说明在本套固定融合规则下值得优先确认，不能据此断定文本信息普遍无用。首批不确定性大，不自动淘汰其他组合。','']
    for name,_ in groups:
        lines += [f'## {name}','','| 方法 | Toys N10 | 对GRAM | Beauty N10 | 对GRAM |','|---|---:|---:|---:|---:|']
        for key in KEYS:
            a,b=[results[d][name] for d in DOMAINS]
            lines.append(f'| {key} | {a["metrics"][key]["N10"]:.6f} | {a["relative_to_gram_N10_pct"][key]:+.2f}% | '
                f'{b["metrics"][key]["N10"]:.6f} | {b["relative_to_gram_N10_pct"][key]:+.2f}% |')
    lines += ['', '## 新增用户上的重点配对区间','','| 数据集 | 比较 | N10绝对差 | 描述性95%区间 |','|---|---|---:|---|']
    primary='new_batch1_512' if new_count==512 else 'new_all1024'
    for domain in DOMAINS:
        for pair in ('SASRec__only_vs_native','TiSASRec__only_vs_native',
                     'SASRec__only_vs_SASRec__BGE','SASRec__only_vs_SASRec__Qwen4',
                     'SASRec__only_vs_TiSASRec__only'):
            value=results[domain][primary]['comparisons'][pair];ci=value['descriptive_95pct_N10']
            lines.append(f'| {domain} | {pair} | {value["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    lines += ['', '旧三专家参考保留原已学习历史惩罚和旧融合器，不是本轮匹配的三专家消融。',
        '统计解释覆盖11/11；bootstrap2000次、seed20261017，未经多重比较校正，也不包含模型训练和研究选择不确定性。区间跨0不表示等效，均值领先不表示确定胜出。',
        '两批完成后再决定扩大哪些候选，不自动给全部11组继续4096。所有原模型、权重和预测均保留；未来仍可继续未完成部分。']
    report.write_text('\n'.join(lines)+'\n')
    print('ANALYZED',new_count,str(report),flush=True)
    return str(report.relative_to(ROOT))


def alive(pid):
    text=Path(f'/proc/{pid}/stat')
    if not text.exists():return False
    return text.read_text().rsplit(')',1)[1].split()[0]!='Z'


def stop_at_cap(domain,job):
    p=OUT/domain;current=read(RUN/domain/'status.json')
    assert current.get('users_completed',0)>=1536
    pid=job['worker_pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'experiment.phase21.old10_expert_selection worker' in command and '--domain '+domain in command
    assert Path(f'/proc/{pid}').stat().st_uid==os.getuid()
    record=dict(reason='Intentional stop at fixed screening sample size; not a scientific failure',
        worker_pid=pid,generator_pid=job['generator_pid'],before=current,requested_at=stamp())
    write(p/'controlled_stop.json',record)
    os.kill(pid,signal.SIGTERM)
    started=time.monotonic()
    while any(alive(x) for x in (pid,job['generator_pid'],job['lease']['pid'])):
        assert time.monotonic()-started<75,'Original worker cleanup did not finish'
        time.sleep(1)
    record.update(state='STOPPED_AT_SCREENING_CAP',finished_at=stamp(),
        original_after=read(RUN/domain/'status.json'),lease_after=read(RUN/domain/'lease_status.json'))
    assert record['lease_after']['state']=='released'
    write(p/'controlled_stop.json',record)
    print('STOPPED_AT_CAP',domain,'lease released',flush=True)
    return record


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    for rel,digest in read(OUT/'manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest,rel
    jobs=read(OUT/'original_jobs.json');stopped={};reports={};start=time.monotonic()
    while len(stopped)<2:
        latest={d:read(RUN/d/'status.json') for d in DOMAINS}
        if '512' not in reports and all(latest[d].get('users_completed',0)>=1024 for d in DOMAINS):
            reports['512']=analyze(512)
        for domain in DOMAINS:
            if domain in stopped:continue
            if latest[domain].get('users_completed',0)>=1536:
                stopped[domain]=stop_at_cap(domain,jobs[domain])
            elif latest[domain]['state'] in ('FAILED','INCOMPLETE'):
                raise RuntimeError(('Original experiment failed before screening cap',domain,latest[domain]))
        status(state='RUNNING',scope='two fixed new512 batches, all11 arms',reports=reports,
            elapsed_seconds=time.monotonic()-start,
            domains={d:dict(state='STOPPED_AT_SCREENING_CAP' if d in stopped else 'COLLECTING_SECOND_NEW512',
                original_status=stopped[d]['before'] if d in stopped else latest[d],
                analysis_rows=1536 if d in stopped else min(latest[d].get('users_completed',0),1536),
                analysis_target_rows=1536,lease_released=d in stopped) for d in DOMAINS})
        assert time.monotonic()-start<CFG['resource_wait_timeout_seconds'],'Screening-stage wait limit'
        if len(stopped)<2:time.sleep(10)
    reports['1024']=analyze(1024)
    status(state='COMPLETED',scope='expert screening with two new512 batches only',reports=reports,
        original_workers_stopped_intentionally=True,all_original_models_and_predictions_preserved=True,
        leases_released=True,final_method_selected=False,no_automatic_followup=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','supervise','analyze'))
    parser.add_argument('--new-users',type=int,choices=(512,1024));args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        else:analyze(args.new_users)
    except BaseException as error:
        if OUT.exists():
            write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updated_at=stamp()))
            status(state='FAILED',error=repr(error))
        raise
