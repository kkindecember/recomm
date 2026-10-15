"""Finish an already computed ablation after correcting an audit I/O defect."""
import json
import os
from pathlib import Path
import threading
import time
import traceback

from experiment.phase20 import lime_expert_ablation as run
from experiment.phase20 import verify_lime_expert_ablation_fast as fast

OUT = run.OUT


def main():
    begin = time.monotonic()
    prior = run.read(OUT / 'status.json')
    assert prior['state'] in ('FAILED', 'VERIFYING')
    assert not (OUT / 'verification.json').exists()
    assert not (OUT / 'io_recovery.json').exists()
    summaries = {d: run.read(OUT / d / 'summary.json') for d in ('Toys', 'Beauty')}
    result_inputs = {str(p.relative_to(run.ROOT)): run.sha(p)
        for d in summaries for p in (OUT / d / 'summary.json', OUT / d / 'predictions.jsonl')}
    sources = {str(p.relative_to(run.ROOT)): run.sha(p) for p in
               (Path(__file__).resolve(), Path(fast.__file__).resolve())}
    receipt = dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
        prior_status=prior, original_results_sha256=result_inputs, recovery_sources_sha256=sources,
        cause='NPZ archive member re-decompressed for every scalar/row access in independent auditor.',
        correction='Eagerly materialize the archive once; run original frozen auditor unchanged.',
        changed_predictions=False, changed_hyperparameters=False, regenerated_candidates=False)
    run.write(OUT / 'io_recovery.json', receipt)
    stop = threading.Event()
    def heartbeat():
        while not stop.is_set():
            extra = run.read(OUT / 'audit_progress.json') if (OUT / 'audit_progress.json').exists() else {}
            run.status('VERIFYING', stage='independent_audit_io_fixed',
                recovery_elapsed_seconds=time.monotonic() - begin, audit=extra,
                note='Main computation already completed; unchanged outputs undergoing corrected I/O audit.')
            stop.wait(5)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        fast.main()
        for path, expected in {**result_inputs, **sources}.items():
            assert run.sha(run.ROOT / path) == expected, ('Changed during recovery', path)
        flags = {}
        for name in run.VARIANTS:
            if name == 'full':
                continue
            good = all(s['comparisons'][name][scope]['relative_N10'] >= 0
                       and s['comparisons'][name][scope]['net_H10'] >= 0
                       for s in summaries.values() for scope in ('catalog', 'GRAM50'))
            increment = any(s['comparisons'][name][scope]['relative_N10'] >= .005
                            for s in summaries.values() for scope in ('catalog', 'GRAM50'))
            flags[name] = 'WORTH_CONSIDERING_IN_DECODE' if good and increment else 'NO_CLEAR_DUAL_SCOPE_DUAL_DOMAIN_SIGNAL'
        run.write(OUT / 'summary.json', dict(domains=summaries, screening_flags=flags,
            no_automatic_GPU_followup=True, io_recovery='io_recovery.json',
            limitation='Frozen-gate deletion is not retraining or a new generation experiment; historical development reused.'))
        labels = {'full': '完整LIME', 'without_itemcf': '去ItemCF', 'without_text': '去文本',
            'without_sasrec': '去SASRec', 'without_history_penalty': '去历史校准',
            'equal_weights': '三专家等权', 'sasrec_only': 'SASRec单独'}
        lines = ['# LIME专家删减：CPU并行诊断', '',
            'Origin Skill：academic-research-suite / experiment-agent，内联执行。', '',
            '状态：COMPLETED / VERIFIED。两域各512用户；没有新增训练、生成或test读取。', '',
            '主计算于16:57:31完成，含输入准备约39秒。原独立核验反复解压NPZ数组，造成超过20分钟额外耗时。',
            '已保留原代码、预测和汇总，改为每个压缩数组只读一次，再执行原核验全部数学逻辑及断言。',
            '操作和哈希记录见io_recovery.json；核验阶段现每5秒更新进度。此次问题不能归因于GPU竞争。', '',
            '| 信号 | Toys全目录N10 | Beauty全目录N10 | Toys固定GRAM50 N10 | Beauty固定GRAM50 N10 |',
            '|---|---:|---:|---:|---:|']
        for name in run.VARIANTS:
            a, b = summaries['Toys']['metrics'][name], summaries['Beauty']['metrics'][name]
            lines.append(f'| {labels[name]} | {a["catalog"]["N10"]:.6f} | {b["catalog"]["N10"]:.6f} | {a["GRAM50"]["N10"]:.6f} | {b["GRAM50"]["N10"]:.6f} |')
        lines += ['', '## 判断', '',
            '没有变体达到预定双域、双口径继续门槛，不追加同配方GPU实验。',
            '去ItemCF只在Beauty全目录有局部提升，其他口径不能保住完整融合；去文本、去SASRec或等权均无一致正向线索。',
            '去历史校准的这些TopK指标不变，但不能推断其在逐词元引导中无用，因为前缀聚合仍可能受全目录分数变化影响。',
            '这是已经看过的512用户上的冻结配方诊断，不是全局未见评价；删后重归一没有重新拟合门控，不能证明某专家在最优系统中无用。',
            '固定候选排序不代表生成效果。完整Hit@10/50、guided50、描述区间和信号熵见summary.json。', '',
            '当前两种生成机制实验继续独立执行，本轮没有读取其评价输出或改变其配置。', '',
            json.dumps(flags, ensure_ascii=False, indent=2)]
        report = '\n'.join(lines) + '\n'
        run.REPORT.write_text(report)
        (OUT / 'report.md').write_text(report)
        receipt.update(completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                       recovery_seconds=time.monotonic() - begin, original_results_unchanged=True,
                       independent_verification='verification.json')
        run.write(OUT / 'io_recovery.json', receipt)
        stop.set(); thread.join()
        run.status('COMPLETED', stage='done', independently_verified=True, screening_flags=flags,
                   recovery_seconds=time.monotonic() - begin, summary='summary.json',
                   report=str(run.REPORT.relative_to(run.ROOT)), io_recovery='io_recovery.json')
    except Exception:
        stop.set(); thread.join()
        run.status('FAILED', stage='io_recovery_failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
