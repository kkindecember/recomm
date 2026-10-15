"""Finish the authorized screen without requiring an interactive polling session."""
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/relation_direction_20260922_v1'


def state(value, **extra):
    p = OUT / 'finalization_status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=value, pid=os.getpid(),
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra), indent=2)+'\n')
    tmp.replace(p)


def main():
    started = time.monotonic()
    while True:
        current = json.loads((OUT / 'status.json').read_text())
        if current['state'] == 'COMPLETED':
            break
        if current['state'] in ('FAILED', 'TIMED_OUT'):
            state('EXPERIMENT_FAILED', experiment=current)
            return
        if time.monotonic()-started > 3000:
            state('FINALIZER_TIMED_OUT', experiment=current)
            return
        state('WAITING_FOR_EXPERIMENT', experiment=current)
        time.sleep(15)
    state('VERIFYING')
    from experiment.phase21.verify_relation_direction import main as verify
    verify()
    result = json.loads((OUT / 'summary.json').read_text())
    verification = json.loads((OUT / 'independent_verification.json').read_text())
    lines = ['\n## 完成复核与工程记录\n',
             '- 4项边界检查通过；另一路数值实现重放全部72个校准头、全部预测排名、用户分折、训练标准化、覆盖计数及投入判定。复核由同一助手编写，非外部独立审稿。',
             '- 初次沙箱后台启动被清理，尚未运行科学实验；随后以tmux启动同一冻结配置。工程记录见 engineering_launch_note.json。',
             '- 数据采用按用户末次事件划分，不是全局时间切分；结论限定于该协议。',
             '- 完整结果已写入 summary.json；independent_verification.json 记录数值复核；没有自动开始下一项训练。', '']
    for path in (OUT / 'report.md', ROOT / 'report/第二十一阶段/关系方向首轮探索_20260922.md'):
        path.write_text(path.read_text()+'\n'.join(lines))
    plan = ROOT / 'plan/第二十一阶段/第21阶段_专家辅助生成推荐_总计划.md'
    text = plan.read_text()
    verdict = '通过筛查，待生成接口验证' if result['passes'] else '未通过双域门槛，方向图暂不进入生成大实验'
    text = text.replace('21.1 两张候选卡完成、关系方向证据筛查运行中；21.2 生成机制验证待筛查结果决定。',
                        f'21.1 候选卡与筛查完成；{verdict}。')
    text = text.replace('21.1，候选卡完成、筛查运行中', '21.1，候选卡与筛查完成')
    text += '\n筛查及数值复核已完成。' + verdict + '。见[首轮结果报告](../../report/第二十一阶段/关系方向首轮探索_20260922.md)。\n'
    plan.write_text(text)
    state('VERIFIED', passes=result['passes'], verification=verification)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (OUT / 'finalization_failure.txt').write_text(traceback.format_exc())
        state('FAILED', error=repr(error))
        raise
