"""CPU-only endpoint analysis for the preregistered two-arm L0 screen."""
import csv
import json
from pathlib import Path

import numpy as np

from experiment.phase20.analyze_gacr_list_distill_diagnostic import logprob, rank, ndcg, mean_by_user

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('CE_anchor', 'true_teacher')


def read(path):
    return json.loads(path.read_text())


def decision(comparison):
    checks = {}
    for part in ('fit', 'calibration'):
        row = comparison[part + '_covered']
        for metric, sign in (('teacher_kl', -1), ('promoted_demoted_satisfaction', 1)):
            value = row[metric + '_delta']
            checks[part + '_' + metric] = value is not None and np.isfinite(value) and sign * value > 0
    value = comparison['calibration_all']['ndcg10_delta_user_mean']
    checks['calibration_user_ndcg10'] = value is not None and np.isfinite(value) and value > 0
    return dict(checks={k: bool(v) for k, v in checks.items()}, passed=bool(all(checks.values())))


def analyze(out):
    config, panel = read(out/'config.json'), read(out/'panel.json')
    for arm in ARMS:
        summary = read(out/arm/'summary.json')
        assert summary['state'] == 'COMPLETED' and summary['optimizer_steps'] == 32
    indices = np.asarray(panel['fit_indices'] + panel['calibration_indices'])
    users = np.asarray([s['user_id'] for s in panel['samples']])
    with np.load(ROOT/config['distillation_cache']) as cache:
        pos, q = cache['positions'][indices], cache['true_probability'][indices]
        parent, fit = cache['parent_scores'][indices], cache['fit'][indices]
    parent_log, teacher_log = logprob(parent), np.log(q)
    parent_order = np.argsort(-parent, axis=1, kind='stable')
    teacher_order = np.argsort(-q, axis=1, kind='stable')
    pairs, disagreements = [], []
    for p, t in zip(parent_order, teacher_order):
        pairs.append([(int(t[j]), int(t[k])) for j in range(10) for k in range(j+1, 50)])
        up, down = set(t[:10])-set(p[:10]), set(p[:10])-set(t[:10])
        disagreements.append([(int(i), int(j)) for i in sorted(up) for j in sorted(down)])
    masks = {part+'_'+name: m & s for part, m in (('fit', fit), ('calibration', ~fit))
             for name, s in (('all', np.ones(len(indices), dtype=bool)), ('covered', pos >= 0), ('uncovered', pos < 0))}
    groups, models, csv_rows = {}, {}, []
    for arm in ARMS:
        for step in (0, 32):
            name = arm + '_step' + str(step)
            with np.load(out/arm/f'step{step}.scores.npz') as data:
                assert np.array_equal(indices, data['indices'])
                scores, ce = data['scores'].astype(np.float64), data['ce']
            assert np.isfinite(scores).all() and np.isfinite(ce).all()
            logs, ranks = logprob(scores), rank(scores, pos)
            def satisfaction(relations):
                return np.asarray([np.mean([float(row[i] > row[j]) + .5*float(row[i] == row[j]) for i, j in rel])
                                   if rel else np.nan for row, rel in zip(scores, relations)])
            values = dict(ce=ce, teacher_kl=np.sum(q*(teacher_log-logs), axis=1),
                          anchor_kl=np.sum(np.exp(parent_log)*(parent_log-logs), axis=1),
                          ndcg10=ndcg(ranks), hit10=(ranks <= 10).astype(float),
                          teacher_top10_pair_satisfaction=satisfaction(pairs),
                          promoted_demoted_satisfaction=satisfaction(disagreements),
                          score_change_l2=np.linalg.norm(scores-parent, axis=1))
            models[name] = dict(values=values, ranks=ranks, scores=scores)
            groups[name] = {}
            for group, mask in masks.items():
                result = dict(prefixes=int(mask.sum()), users=len(set(users[mask])))
                for key, value in values.items():
                    valid = mask & np.isfinite(value)
                    result[key] = float(value[valid].mean()) if valid.any() else None
                    result[key+'_user_mean'] = mean_by_user(value, users, valid)
                    result[key+'_valid_prefixes'] = int(valid.sum())
                    result[key+'_valid_users'] = len(set(users[valid]))
                result['teacher_ndcg10'] = float(ndcg(rank(q, pos))[mask].mean())
                result['teacher_ndcg10_user_mean'] = mean_by_user(ndcg(rank(q, pos)), users, mask)
                groups[name][group] = result
                csv_rows.append(dict(model=name, group=group, **result))
    np.testing.assert_allclose(models['CE_anchor_step0']['scores'], models['true_teacher_step0']['scores'], rtol=0, atol=1e-6)
    comparisons = {}
    for other in ('CE_anchor_step32', 'true_teacher_step0'):
        comparisons[other] = {}
        a, b = models['true_teacher_step32'], models[other]
        for group, mask in masks.items():
            result = {}
            for key, value in a['values'].items():
                delta = value-b['values'][key]
                valid = mask & np.isfinite(delta)
                result[key+'_delta'] = float(delta[valid].mean()) if valid.any() else None
                result[key+'_delta_user_mean'] = mean_by_user(delta, users, valid)
            result['hit10_gained'] = int(((a['ranks'] <= 10) & (b['ranks'] > 10) & mask).sum())
            result['hit10_lost'] = int(((a['ranks'] > 10) & (b['ranks'] <= 10) & mask).sum())
            result['target_rank_changed'] = int(((a['ranks'] != b['ranks']) & mask).sum())
            comparisons[other][group] = result
    gate = decision(comparisons['CE_anchor_step32'])
    result = dict(state='ANALYZED', groups=groups, true_teacher_minus=comparisons, gate=gate,
                  formal_optimizer_steps=64, engineering_optimizer_steps=2, test_read=False,
                  gate_definition=config['gate'], training_subset=read(out/'training_subset.json'),
                  recommendation=('仅进入完整同lr三臂与fresh beam的预算评审；小面板未证明生成收益。' if gate['passed'] else
                                  '当前时间预算下停止内部KL路线扩展；不追加lr/权重网格。'))
    (out/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (out/'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    return result


def render_report(out, state, statuses, analysis=None, error=None):
    config = read(out/'config.json')
    report = ROOT/config['report']
    lines = ['# Stage20 Toys L0：有限预算学习率筛查', '',
             f"当前状态：**{state}**。用户于2026-09-15诊断后授权启动；更新时间见产物status.json。", '',
             'CE_anchor、true_teacher分别从原C1重启；峰值lr=1e-5。原固定顺序前4,096个FIT前缀，effective batch=128，固定32次optimizer调用（首步lr=0，31次非零更新）。保留原171步scheduler、9步warmup、chunk4、损失、冻结范围与21,839/12,399 covered归一系数。', '',
             '两张独立GPU共享运行，各先做8例profile和1次可丢弃工程更新，恢复C1并重置optimizer/RNG后才开始正式训练。两臂profile均通过后放行。每臂计算硬上限90分钟，资源等待最多15分钟；不重试、不自动扩展。', '',
             '评估严格固定step0/32：原Q0 FIT 256前缀/256用户（192 covered），calibration 256前缀/201用户（144 covered）。FIT面板为分层诊断；其与本轮前4,096条训练的重合数见training_subset.json。calibration是既有开发数据。固定列表结果不能代替fresh beam或泛化验证。', '',
             '**结果产生前冻结的继续门槛：** 真教师相对CE的FIT covered和calibration covered逐前缀平均教师KL均严格更低、U×D关系满足率均严格更高，且calibration all用户等权NDCG@10严格更高。U×D为空时排除该行并报告有效分母；五项须全部满足。另报告全部组的前缀/用户均值，不能事后换门槛。', '',
             '| 臂 | 状态/阶段 | 正式步数 | GPU |', '|---|---|---:|---|']
    for arm in ARMS:
        s = statuses.get(arm, {})
        lines.append(f"| {arm} | {s.get('state', 'PENDING')} / {s.get('stage', 'pending')} | {s.get('optimizer_steps', 0)}/32 | {s.get('gpu_uuid', config['gpus'][arm])} |")
    if error:
        lines += ['', '工程/资源停止原因：`'+str(error).replace('`', "'").replace('\n', ' ')+'`。未完成结果不解释为方法负结果。']
    if analysis:
        lines += ['', '**固定末步结论：'+analysis['recommendation']+'**', '',
                  '| 真教师 − CE | FIT covered | calibration covered |', '|---|---:|---:|']
        delta = analysis['true_teacher_minus']['CE_anchor_step32']
        for label, key in (('教师KL（前缀均值）', 'teacher_kl_delta'), ('U×D满足率（前缀均值）', 'promoted_demoted_satisfaction_delta'),
                           ('NDCG@10（用户等权）', 'ndcg10_delta_user_mean')):
            lines.append(f"| {label} | {delta['fit_covered'][key]:+.9f} | {delta['calibration_covered'][key]:+.9f} |")
        lines += ['', f"calibration all用户等权ΔNDCG@10 = {delta['calibration_all']['ndcg10_delta_user_mean']:+.9f}。", '',
                  '门槛逐项：`'+json.dumps(analysis['gate'], ensure_ascii=False)+'`。', '',
                  '| 模型/固定末步 | FIT covered KL | calibration covered KL | calibration all用户N10 |', '|---|---:|---:|---:|']
        for name in ('CE_anchor_step0', 'CE_anchor_step32', 'true_teacher_step32'):
            g = analysis['groups'][name]
            lines.append(f"| {name} | {g['fit_covered']['teacher_kl']:.9f} | {g['calibration_covered']['teacher_kl']:.9f} | {g['calibration_all']['ndcg10_user_mean']:.9f} |")
        lines += ['', '本轮只有两个臂、一个seed、固定小面板；通过门槛也不能排除一般继续训练或新lr下打乱控制的影响。未通过是预算下止损，不证明更长训练必然无效。']
    rel = '../../'+str(out.relative_to(ROOT))
    lines += ['', f"产物：[状态]({rel}/status.json)、[冻结配置]({rel}/config.json)、[训练子集审计]({rel}/training_subset.json)、[完整指标]({rel}/metrics.csv)、[分析]({rel}/analysis.json)。",
              '', '状态命令：`bash experiment/phase20/run_gacr_list_distill_l0.sh status`。', '']
    temp = report.with_suffix('.tmp.md'); temp.write_text('\n'.join(lines)); temp.replace(report)
