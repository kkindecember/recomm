"""One authorized frozen Beauty evaluation, with bounded supervision/reporting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
child=None
stopping=False


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return {}


def write(path,value):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def table(methods):
    keys=[f'{m}@{k}' for m in ['ndcg','hit'] for k in [5,10,20,50]]
    lines=['| 方法 | N5 | N10 | N20 | N50 | H5 | H10 | H20 | H50 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in methods.items():
        lines.append('| '+name+' | '+' | '.join(f'{row[k]:.6f}' for k in keys)+' |')
    return '\n'.join(lines)


def refresh(config,queue):
    report=ROOT/config['report']
    toys_dir=ROOT/'artifacts/phase20/gacr_score_controls_toys_v2'
    toys=read(toys_dir/'summary.json')
    toys_status=read(toys_dir/'status.json')
    smoke=read(ROOT/config['smoke_output']/'status.json')
    parent=read(ROOT/config['shared_parent_output']/'status.json')
    beauty_status=read(ROOT/config['output']/'status.json')
    beauty=read(ROOT/config['output']/'summary.json')
    content='# Stage20 后续固定对照与Beauty迁移进展报告\n\n'
    content+='## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n- Origin Mode: run / validate；已完成结果仅标记ANALYZED。\n'
    content+=f'- 更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %z")}；Beauty队列：**{queue.get("state","NOT_STARTED")}**。\n'
    content+='- 用户于2026-09-12同意继续执行上一轮验证计划；零新增训练，仅读prepared train/validation及既有validation缓存。\n\n'
    content+='## 运行范围与状态\n\n| 工作 | 当前状态 | 进度/信息 |\n|---|---|---|\n'
    for name,row in [('Toys固定对照v2',toys_status),('Beauty工程检查',smoke),('Beauty同精度父参考',parent),('Beauty组合评估',beauty_status)]:
        detail=row.get('stage',row.get('mode',''))
        if 'examples' in row:detail+=f" {row['examples']:,}/{row.get('total',0):,}"
        if 'error' in row:detail+=' '+str(row['error'])[:400]
        content+=f"| {name} | {row.get('state','NOT_STARTED')} | {detail} |\n"
    content+=f"\nBeauty使用GPU UUID `{config['gpu_uuid']}`，至少{config['minimum_free_mib']:,}MiB空闲准入，allocator上限40%；父参考和C1/GACR顺序执行。工程检查30分钟，正式两部分合计24小时硬上限，资源等待最多6小时，无自动重试、训练或后续seed扩展。\n"
    if queue.get('reason'):content+='\n队列说明：'+str(queue['reason'])+'\n'
    content+='\n## Toys对照\n\n'
    content+='首次CPU重建因约3e-8的残差数值差异改变Top50边界，在新对照指标产生前停止，保留v1。v2只将129参数冻结残差前向恢复到FP32/TF32关闭的GPU，CF、PCRF和统计仍在CPU。逐用户一致性门槛、候选、参数和三个控制定义均不改变。\n\n'
    if toys:
        content+=table(toys['groups']['all_validation'])+'\n\n'
        key='gacr_v6_pcrf_minus_c1_reciprocal_pcrf'
        x=toys['comparisons'][key]['all_validation']; n=x['ndcg@10']
        content+=f"预定主比较（完整组合减C1倒数rank+PCRF）：N10 {n['delta']:+.6f}，描述性95%区间[{n['ci95'][0]:+.6f},{n['ci95'][1]:+.6f}]，H10 {x['hit@10']['delta']:+.6f}。\n\n"
    else:
        content+='尚无完整新对照summary，不从部分进度推断科学结论。\n\n'
    content+='[Toys完整对照报告](Stage20_Toys_GACR分数与候选固定对照报告.md)。\n\n'
    content+='## Beauty冻结协议与结果\n\n'
    content+='官方validation共22,363人，使用既有Beauty C1、GACR-v6 residual seed2023、PCRF head seed2023。原GRAM新生成FP32/TF32关闭/原生beam50参考；组合规则沿用Toys，lambda1/beta0.5/gamma1固定，Beauty q1=6预先固定。目录头按原文件行映射；零optimizer updates。工程阶段核查64人历史PCRF适配、16人重复父预测及16人原/批量GACR候选/特征/排序；正式阶段再次核对父参考与smoke一致性。\n\n'
    if beauty:
        content+=table(beauty['metrics'])+'\n\n'
        for baseline,x in beauty['paired_comparisons'].items():
            n=x['ndcg@10'];h=x['hit@10'];ci=n['paired_bootstrap_95ci']
            content+=f"- 组合对{baseline}：N10 {n['delta']:+.6f}，描述性95%区间[{ci[0]:+.6f},{ci[1]:+.6f}]；H10 {h['delta']:+.6f}。\n"
        for group,value in beauty['groups'].items():
            if group=='all_validation':continue
            users=next(iter(value['metrics'].values()))['users']
            content+=f'\n### {group}（{users:,}人）\n\n'+table(value['metrics'])+'\n'
        content+=f"\n原固定组合决策：`{beauty['decision']}`。单域单残差seed的跨域验证不会自动触发继续调参或多seed；需结合Toys简单控制解释机制。\n"
    else:
        content+='尚无Beauty组合全量summary；工程通过、父参考完成及部分用户推理均不等于组合提点。\n'
    content+='\n## 解释边界与证据\n\n'
    content+='Toys固定控制用3000次bootstrap、seed20260912；Beauty沿用原组合3000次bootstrap、seed20260910。均是反复开发validation上的描述性用户区间，未校正多比较，不代表骨干/残差seed不确定性或全新独立确认。完整报告保留全部臂、频次分组、C1用户内外及工程失败；不将覆盖变化或score相关写成因果证明。没有新test读取。\n\n'
    content+=f'- [Beauty冻结配置](../../{config["config_path"]})；正式目录：`{config["output"]}`；父参考：`{config["shared_parent_output"]}`。\n'
    content+=f'- [监督状态](../../{config["queue_output"]}/status.json)；每30秒刷新本报告，终态后退出。\n'
    content+='- [后续计划](../../plan/第二十阶段/第20阶段收尾后的验证与方法决策计划.md)、[上一批终态复盘](第20阶段_终态复盘与下一步决策_20260912.md)。\n'
    report.write_text(content)


def stop(signum,frame):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)


def main():
    global child
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='experiment/phase20/gacr_v6_pcrf_beauty_v1.json')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/args.config)
    out=ROOT/config['queue_output']
    if args.report_only:
        refresh(config,read(out/'status.json'))
        return
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'status.json').exists():
        raise FileExistsError('Existing queue attempt; refusing automatic restart')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)

    def save(state,**extra):
        row={'state':state,'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'automatic_retry':False,'test_read':False,**extra}
        write(out/'status.json',row)
        refresh(config,row)
    try:
        smoke=read(ROOT/config['smoke_output']/'status.json')
        assert smoke.get('state')=='PASSED','Independent smoke must pass before queue launch'
        started=time.time()
        while not stopping:
            if time.time()-started>config['max_resource_wait_seconds']:
                save('STOPPED',reason='Resource wait budget reached; no evaluation launched')
                return
            free=int(subprocess.check_output(['nvidia-smi','--id='+config['gpu_uuid'],
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
            if free>=config['minimum_free_mib']:break
            save('WAITING_FOR_RESOURCE',free_mib=free,required_mib=config['minimum_free_mib'])
            time.sleep(30)
        if stopping:
            save('STOPPED',reason='Stopped before launch');return
        output=ROOT/config['output']
        if (output/'manifest.json').exists() or (output/'status.json').exists():
            raise FileExistsError('Existing formal attempt; no restart')
        output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','experiment.phase20.evaluate_gacr_transfer','--config',args.config,'--mode','run']
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=config['gpu_uuid'])
        with (output/'run.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launched=time.time();timeout_sent=None
            while child.poll() is None:
                save('STOPPING' if stopping else 'RUNNING',child_pid=child.pid,elapsed_seconds=round(time.time()-launched,1),
                    output=config['output'],formal_budget_seconds=config['max_wall_seconds'])
                if time.time()-launched>config['max_wall_seconds']+60 and timeout_sent is None:
                    save('TIMEOUT_TERMINATING',child_pid=child.pid,reason='24-hour hard budget reached')
                    os.killpg(child.pid,signal.SIGTERM);timeout_sent=time.time()
                if timeout_sent is not None and time.time()-timeout_sent>60:
                    os.killpg(child.pid,signal.SIGKILL)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        detail=read(output/'status.json');summary=read(output/'summary.json')
        success=child.returncode==0 and detail.get('state')=='COMPLETED' and summary.get('state')=='COMPLETED'
        save('COMPLETED' if success else 'STOPPED' if stopping or detail.get('state')=='STOPPED' else 'FAILED',
            child_exit_code=child.returncode,child_status=detail,
            reason='Frozen evaluation finished; read matched metrics and controls' if success else 'No retry; inspect saved error/partial progress')
    except BaseException as error:
        save('STOPPED' if stopping else 'FAILED',reason=repr(error))
        raise


if __name__=='__main__':main()
