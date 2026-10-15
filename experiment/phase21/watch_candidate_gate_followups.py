"""Bounded CPU monitor/finalizer. Never launches, retries or changes experiments."""
import contextlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
GEN=ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'
CTRL=ROOT/'artifacts/phase21/candidate_gate_controls_20261010_v1'
SCREEN=ROOT/'artifacts/phase21/candidate_gate_20261010_v1'
REPORT=ROOT/'report/第二十一阶段'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def write(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def verify_controls():
    result=read(CTRL/'summary.json');checks=0
    for arm in ('unbounded_gate','direct_residual'):
        for d in ('Toys','Beauty'):
            users=read(SCREEN/d/'validation/users.json');r=np.load(CTRL/arm/d/'ranks.npz')
            for scope,row in result[arm][d]['cohorts'].items():
                mask=np.ones(len(users),bool) if scope=='all' else np.array([u['cohort']=='primary' for u in users])
                for seed in (0,1,2):
                    rr=r[str(seed)][mask];n=float(np.where(rr<=10,1/np.log2(rr+1),0).mean())
                    assert abs(n-row['metrics'][f'control_{seed}']['N10'])<1e-12;checks+=1
    write(CTRL/'summary_verification.json',json.dumps(dict(state='PASSED',checks=checks,
        scope='All-user N10 recomputation from saved ranks; not an independent model-forward reproduction'),indent=2)+'\n')


def render_controls(state):
    lines=['# 候选校正结构对照：自动运行记录','',f'状态：{state}。此文件由监控程序更新，不据中途结果选择种子或修改配置。','']
    data=read(CTRL/'summary.json')
    if state=='COMPLETE':
        lines+=['全部两对照×两域×三个种子完成；逐种子N10已从排名复算。',
                '','| 控制 | 域 | 控制N10 | 原C N10 | 控制−C及95%区间 |','|---|---|---:|---:|---|']
        for arm in ('unbounded_gate','direct_residual'):
            for d in ('Toys','Beauty'):
                r=data[arm][d]['cohorts']['primary'];ci=r['ci95_vs_candidate']
                lines.append(f"| {arm} | {d} | {r['control_mean_N10']:.6f} | {r['candidate_mean_N10']:.6f} | {r['delta_vs_candidate']:+.6f} [{ci[0]:+.6f},{ci[1]:+.6f}] |")
        lines+=['','unbounded_gate只取消eta0.1的小幅限制，仍是凸组合，不是无界任意分数。',
                '若该控制更好，不能主张小幅限制提高准确率；若direct_residual更好或相当，应收窄专属门控贡献。这些仍是商品排序结果。']
    else:
        lines+=['尚未完成全部预定模型，不用已完成子集形成双域结论。', '', '运行状态：', '```json',json.dumps(read(CTRL/'status.json'),ensure_ascii=False,indent=2),'```']
    lines+=['','[完整机器结果](../../artifacts/phase21/candidate_gate_controls_20261010_v1/summary.json)',
            '[固定方案](../../plan/第二十一阶段/21.19C_候选校正结构对照_20261010.md)','']
    write(REPORT/'候选校正结构对照结果_20261010.md','\n'.join(lines))


def render_generation(state,states):
    lines=['# 候选门控真实生成迁移：自动运行记录','',f'状态：{state}。固定每域2048名primary用户、头seed0、alpha0.3、beam50。','']
    if state=='COMPLETE':
        data=read(GEN/'summary.json')
        lines+=['| 域 | 旧引导A N10 | 用户门控B N10 | 候选门控C N10 | C同信号后排N10 |','|---|---:|---:|---:|---:|']
        for d in ('Toys','Beauty'):
            m=data[d]['metrics'];lines.append(f"| {d} | {m['original']['N10']:.6f} | {m['user_guided']['N10']:.6f} | {m['score_guided']['N10']:.6f} | {m['score_late']['N10']:.6f} |")
        lines+=['','| 域 | 比较 | ΔN10 | 配对95%描述性区间 | 前十新增/丢失 |','|---|---|---:|---|---|']
        for d in ('Toys','Beauty'):
            for key in ('score_guided_vs_original','score_guided_vs_user_guided','score_guided_vs_score_late'):
                r=data[d]['comparisons'][key];ci=r['descriptive_95pct_N10']
                lines.append(f"| {d} | {key} | {r['delta_N10']:+.6f} | [{ci[0]:+.6f},{ci[1]:+.6f}] | {r['top10_gained']}/{r['top10_lost']} |")
        positive=all(data[d]['comparisons'][key]['delta_N10']>0 for d in ('Toys','Beauty')
            for key in ('score_guided_vs_original','score_guided_vs_user_guided'))
        lines+=['','双域相对A/B点估计均为正：'+str(positive)+'。',
                '这是固定子集的迁移筛查。三个头种子的完整商品排序结果与本轮单头种子生成结果不能混为一谈；正点估计或跨零区间均不代替完整生成和骨干重复。']
    else:
        lines+=['| 域 | 运行状态 | 已保存用户 / 2048 | 阶段 |','|---|---|---:|---|']
        for d,s in states.items():
            old=s.get('previous',{});lines.append(f"| {d} | {s.get('state','UNKNOWN')} | {s.get('users_completed',old.get('users_completed',0))} | {s.get('stage',old.get('stage',''))} |")
        lines+=['','未完成全部预定用户时不计算完整实验结论。若失败或超时，原始输出与失败原因保留；监控器不会自动重启或缩小样本。']
    lines+=['','[机器结果目录状态](../../artifacts/phase21/candidate_gate_generation_20261010_v1/status.json)',
            '[固定方案](../../plan/第二十一阶段/21.19B_候选门控真实生成迁移筛查_20261010.md)','']
    write(REPORT/'候选门控真实生成迁移结果_20261010.md','\n'.join(lines))


def main():
    gen_done=ctrl_done=False;start=time.time()
    # Each domain's own worker enforces 7200s and an external 7500s timeout.
    # Monitor is more generous only so it can record exit/lease cleanup.
    ctrl_deadline=(CTRL/'manifest.json').stat().st_mtime+1980
    while not(gen_done and ctrl_done):
        states={d:read(GEN/d/'status.json') for d in ('Toys','Beauty')}
        gs='RUNNING'
        if all(s.get('state')=='COMPLETED' for s in states.values()):
            if not gen_done:
                from experiment.phase21 import candidate_gate_generation as G
                with (GEN/'finalizer.log').open('w') as log,contextlib.redirect_stdout(log):G.summarize()
            gs='COMPLETE';gen_done=True
        elif all(s.get('state') in ('COMPLETED','FAILED','INTERRUPTED','RESOURCE_TIME_LIMIT') for s in states.values()):
            gs='INCOMPLETE';gen_done=True
        elif time.time()-start>8100:
            gs='MONITOR_TIMEOUT_INCOMPLETE';gen_done=True
        if not(gen_done and gs=='COMPLETE'):write(GEN/'status.json',json.dumps(dict(state=gs,domains=states),ensure_ascii=False,indent=2)+'\n')
        render_generation(gs,states)
        cs=read(CTRL/'status.json').get('state','RUNNING')
        if cs=='COMPLETE':
            if not ctrl_done:verify_controls()
            ctrl_done=True
        elif cs=='FAILED':ctrl_done=True
        elif time.time()>ctrl_deadline:cs='DEADLINE_INCOMPLETE';ctrl_done=True
        render_controls(cs)
        write(GEN/'monitor_status.json',json.dumps(dict(state='COMPLETE' if gen_done and ctrl_done else 'RUNNING',
             updated_unix=time.time(),generation=gs,controls=cs),indent=2)+'\n')
        if not(gen_done and ctrl_done):time.sleep(10)


if __name__=='__main__':
    try:main()
    except BaseException:
        write(GEN/'monitor_failure.txt',traceback.format_exc());raise
