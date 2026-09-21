"""Background status and final audit; no GPU jobs or experiments are added."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1'
RUN = OUT / 'bounded_v3'
BEAUTY = RUN / 'beauty_gpu2'
REPORT = ROOT / 'report/第二十阶段/LIME信号引导原GRAM生成_限时筛查结果_20260921.md'
MODULE = 'experiment.phase20.finalize_lime_decode_screen'
TERMINAL = {'COMPLETED', 'FAILED', 'INCOMPLETE', 'TIMEOUT', 'RESOURCE_TIMEOUT'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def progress(root):
    value = read(root / 'status.json')
    for name in ('tuning', 'evaluation'):
        p = root / (name + '.jsonl')
        if p.exists():
            with p.open() as f:
                value[name + '_users_written'] = sum(1 for _ in f)
    selection = read(root / 'selection.json')
    if selection:
        value['selected_alpha'] = selection['selected_alpha']
    return value


def finish(domains):
    beauty_root = BEAUTY if (BEAUTY / 'Beauty/status.json').exists() else RUN
    completed = {d: read(p / 'summary.json') for d, p in
                 (('Toys', RUN / 'Toys'), ('Beauty', beauty_root / 'Beauty'))
                 if read(p / 'status.json').get('state') == 'COMPLETED'}
    audit = None
    if completed:
        command = [sys.executable, '-m', 'experiment.phase20.audit_lime_decode_available',
                   '--output', str(RUN.relative_to(ROOT))]
        if beauty_root == BEAUTY:
            command += ['--beauty-output', str(BEAUTY.relative_to(ROOT))]
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (OUT / 'audit.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError('Independent audit failed; see audit.log')
        audit = str((RUN / 'available_domains_audit.json').relative_to(ROOT))
    def performance(s):
        c = s['comparisons']['lime_vs_GRAM']
        return c['relative_N10'] is not None and c['relative_N10'] >= .01 and c['net_H10'] >= 0
    if len(completed) == 2:
        signal = all(performance(s) for s in completed.values())
        deltas = [s['comparisons']['lime_vs_late']['delta_N10'] for s in completed.values()]
        extra = all(v >= 0 for v in deltas) and any(v > 0 for v in deltas)
        decision = ('PROMISING_DEVELOPMENT_SIGNAL' if signal and extra else
                    'PERFORMANCE_ONLY_SIGNAL' if signal else 'NO_CONTINUATION_SIGNAL')
    elif completed:
        decision = ('POSITIVE_SINGLE_DOMAIN_NEEDS_REPLICATION' if all(performance(s) for s in completed.values())
                    else 'NO_CONTINUATION_SIGNAL_ON_COMPLETED_DOMAIN')
    else:
        decision = 'INCOMPLETE_NO_EFFECT_CONCLUSION'
    result = dict(decision=decision, domains=completed, run_status=domains,
                  independent_audit=audit, test_read=False, development_diagnostic=True,
                  caveat='Single-seed reused-development pilot. Cost failure is not scientific failure; small samples cannot establish absence of an effect.')
    write(OUT / 'screen_summary.json', result)
    lines = ['## 后台自动汇总', '', f'完成时间：{time.strftime("%Y-%m-%d %H:%M:%S %z")}。决策代码：`{decision}`。', '',
             '这是单种子、旧开发集的小规模诊断；成本未通过不等于效果无效。', '']
    for domain, s in completed.items():
        lines += [f'### {domain}', '',
                  f'选参 {s["tuning_users"]} 人，评价 {s["evaluation_users"]} 人；选中权重 {s["selected_alpha"]}。', '',
                  '| 方法 | NDCG@10 | Hit@10 | Hit@50 |', '|---|---:|---:|---:|']
        for name, m in s['metrics'].items():
            lines.append(f'| {name} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} |')
        c = s['comparisons']['lime_vs_GRAM']
        relative = '基线为零，不能计算' if c['relative_N10'] is None else f'{100*c["relative_N10"]:+.2f}%'
        lines += ['', f'LIME引导相对GRAM的NDCG@10变化：{relative}；Top10新增命中{c["gained_H10"]}人、损失{c["lost_H10"]}人。',
                  f'相对同规则事后打分的NDCG@10绝对差：{s["comparisons"]["lime_vs_late"]["delta_N10"]:+.6f}。',
                  f'评价NDCG@10差的描述性95%区间：{c["descriptive_95_interval"]}。', '']
        if s['selected_alpha']['lime'] == 0:
            lines += ['LIME权重选中了0，评价输出回退为GRAM。这说明选参阶段未支持启用引导，不能解释成所有非零权重在评价集都已被证明无效。', '']
    for domain in ('Toys', 'Beauty'):
        if domain not in completed:
            lines += [f'{domain} 未完成效果评价，查看统一状态中的原因；不记为科学负结果。', '']
    lines += ['双域通过才进入干净流程复测；没有继续信号则停止本配方扩展。不会自动追加参数或启动下一套实验。', '',
              '[结构化结果](../../artifacts/phase20/lime_decode_screen_20260921_v1/screen_summary.json)', '']
    original = REPORT.read_text().split('## 后台自动汇总')[0]
    REPORT.write_text(original.rstrip() + '\n\n' + '\n'.join(lines))
    return result


def run():
    while True:
        toy = progress(RUN / 'Toys')
        beauty = progress(BEAUTY / 'Beauty')
        queue = read(BEAUTY / 'status.json')
        beauty['scheduler'] = queue
        domains = {'Toys': toy, 'Beauty': beauty}
        status = dict(state='RUNNING', updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      background=True, coordinator_pid=os.getpid(), domains=domains,
                      results='screen_summary.json', report=str(REPORT.relative_to(ROOT)),
                      test_read=False, automatic_new_experiments=False)
        ready = toy.get('state') in TERMINAL and queue.get('state') in TERMINAL
        if ready:
            try:
                result = finish(domains)
                n = len(result['domains'])
                status.update(state='COMPLETED' if n == 2 else 'PARTIAL' if n else 'INCOMPLETE',
                              decision=result['decision'], independent_audit=result['independent_audit'])
            except Exception:
                status.update(state='AUDIT_FAILED', error=traceback.format_exc())
            write(OUT / 'status.json', status)
            return
        write(OUT / 'status.json', status)
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'run'])
    if parser.parse_args().action == 'run':
        run()
        return
    backup = OUT / 'initial_attempt_status.json'
    if not backup.exists():
        os.link(OUT / 'status.json', backup)
    old_launch = OUT / 'coordinator_launch.json'
    if old_launch.exists() and not (OUT / 'coordinator_sandbox_launch.json').exists():
        os.link(old_launch, OUT / 'coordinator_sandbox_launch.json')
    with (OUT / 'coordinator.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-m', MODULE, 'run'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(OUT / 'coordinator_launch.json', dict(pid=child.pid, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps({'pid': child.pid, 'status': str(OUT / 'status.json')}), flush=True)


if __name__ == '__main__':
    main()
