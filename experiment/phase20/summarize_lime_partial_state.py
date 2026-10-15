"""Export every score and a verified report; no parameter/model selection."""
import csv
import json
from pathlib import Path
import time
from experiment.phase20.review_lime_clean_benefit import read, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/phase20/lime_partial_state_20260922_v1'
REPORT = ROOT/'report/第二十阶段/LIME当前生成状态替代最终信息_20260922.md'


def main():
    s = read(OUT/'summary.json'); assert s['state'] == 'COMPLETED' and not s['pending_verification']
    passed = [d for d in (2, 3) if s['decision'][f'prefix{d}_all_seeds']]
    rows = []
    for domain, r in s['domains'].items():
        for cohort in ('oof', 'development'):
            base = r[cohort]['metrics']['fixed_late']
            for name, m in r[cohort]['metrics'].items():
                row = dict(domain=domain, cohort=cohort, method=name, users=r[cohort]['users'], **m,
                    relative_N10_percent=100*(m['N10']/base['N10']-1))
                if name in r[cohort]['contrasts']:
                    c = r[cohort]['contrasts'][name]['fixed_late']
                    row.update(delta_N10=c['delta_N10'], gained=c['top10_gained'], lost=c['top10_lost'],
                               ci_low=c['descriptive_95pct_N10'][0], ci_high=c['descriptive_95pct_N10'][1],
                               **r[cohort]['movement'][name], **r['checks'][name])
                rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    lines = ['# LIME 当前生成状态替代最终信息：实验结果', '', '## Material Passport', '',
        '- 日期：2026-09-22；Origin Skill：academic-research-suite，内联执行与数值验证。',
        '- Verification Status：VERIFIED；输出：`artifacts/phase20/lime_partial_state_20260922_v1`。',
        '- 问题：第2/3次词元扩展时的信息，能否替代最终信息帮助普通纠错网络？',
        '- 这是信息可用性检查；原生轨迹、冻结最终候选、生成后底分仍保留，未实现或验证新的beam引导机制。', '',
        '## 事先固定的判定结果', '',
        ('通过两域、全部三个seed联合投入门槛的时点：'+str(passed)+'。这只支持进一步验证，不代表新机制成立。') if passed else
        '两个时点都未通过两域、全部三个seed的联合投入门槛。本轮停止，不自动扩展超参、不接入新控制器。失败只针对当前状态特征、损失和训练预算，不能证明全部生成中机制不可能。', '',
        '以下均以同一候选上的固定0.3引导评分fixed_late为参照。范围包含全部三个seed，不选最好一次；OOF为训练用户五折外预测。', '',
        '| 数据集 | 信息组合 | 开发N10相对变化范围 | OOF N10相对变化范围 | 开发H10范围 |',
        '|---|---|---:|---:|---:|']
    labels = {'full':'完整最终信息+专家', 'early':'仅提前专家/历史信息',
              'prefix2':'第2步状态+专家', 'prefix3':'第3步状态+专家',
              'state2':'仅第2步状态', 'state3':'仅第3步状态'}
    for domain in ('Toys', 'Beauty'):
        r = s['domains'][domain]
        for group, label in labels.items():
            ranges = []
            for cohort in ('development','oof'):
                m = r[cohort]['metrics']; b = m['fixed_late']['N10']
                v = [100*(m[f'{group}_seed{k}']['N10']/b-1) for k in range(3)]
                ranges.append(f'{min(v):+.2f}%～{max(v):+.2f}%')
            h = [r['development']['metrics'][f'{group}_seed{k}']['H10'] for k in range(3)]
            lines.append(f'| {domain} | {label} | {ranges[0]} | {ranges[1]} | {min(h):.6f}～{max(h):.6f} |')
    lines += ['', '## 本轮信息边界', '',
        '输入四个当前状态量：当前原生累计分数/当前长度；同路径固定引导累计更新/当前长度；当步全部合法扩展的原生倒数名次；当前混合分数距当步第50名边界/当前长度。长度只取已经扩展的2/3个词元，不读未来商品路径长度。', '',
        '竞争集合来自实际原生beam的所有活跃父前缀，剔除-1e9填充beam。特征先对所有扩展计算，再按原冻结候选的对应前缀取值；不从最终50名反推当前竞争范围。已用删除/污染后一步记录的测试确认较早特征不受影响。', '',
        'prefix2/prefix3使用当前4维+原6维专家/历史信息；state2/state3只使用当前4维。网络、同seed初始化、训练折、标准化、30次AdamW更新、有界2残差和最终底分都保持不变。全信息full与仅提前信息early从上一轮逐值复制，不重新选模型。', '',
        '仍有两个重要限制：总评分含10*fixed_late的完整路径底分；训练与评价在原生生成结束后选出的50候选上进行。即使当前状态有收益，本轮也不能证明它能在引导轨迹上救回被剪枝目标。第3步在一些商品上已唯一识别商品，但完整生成分数尚不可得；不能把“未结束”直接解释为仍有相同的候选扩展空间。', '',
        '## 全部开发分数', '',
        '| 域 | 方法 | NDCG@10 | HR@10 | 相对N10 | 新增/丢失前10 | ΔN10描述性95%区间 |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        if row['cohort'] != 'development': continue
        gain = f"{row['gained']}/{row['lost']}" if 'gained' in row else '见上一轮'
        ci = f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]" if 'ci_low' in row else '—'
        lines.append(f"| {row['domain']} | {row['method']} | {row['N10']:.6f} | {row['H10']:.6f} | {row['relative_N10_percent']:+.2f}% | {gain} | {ci} |")
    lines += ['', '## 新组的OOF与各折门槛', '',
        '| 域 | 方法 | OOF N10 | OOF H10 | 相对N10 | 正向折数 | 开发通过 | OOF通过 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for row in rows:
        if row['cohort'] != 'oof' or 'positive_folds_vs_both' not in row: continue
        lines.append(f"| {row['domain']} | {row['method']} | {row['N10']:.6f} | {row['H10']:.6f} | {row['relative_N10_percent']:+.2f}% | {row['positive_folds_vs_both']}/5 | {row['development_pass']} | {row['oof_pass']} |")
    lines += ['', '两域总用户分别3285/3822，其中训练2773/3310、开发各512。训练合法目标覆盖562/767，开发92/115；全部评价保留未覆盖用户。更完整的N5/N10/N20/N50、H5/H10/H20/H50、对GRAM/full/early的配对比较及每折数据见metrics.csv与各域analysis.json。', '',
        '## 工程执行与复核', '',
        '- 两域6位smoke用户：前三步与完整生成逐值一致，完整候选及评分与旧缓存逐值一致。两个检查点前后digest相同，未重训GRAM、SASRec或融合器。',
        '- 完成7107位用户的前三步记录，另有6次完整生成smoke；144个新头、4320次小头优化更新。保存用户、所有合法扩展的原始轨迹、特征、模型、预测及排名。',
        '- 3项CPU单元测试和两域各20位真实用户的独立重算预检查通过；预检查当前特征最大差异0。正式验证另重建全部特征及评分，不把抽查当完整核验。']
    for d in ('Toys','Beauty'):
        c = read(OUT/d/'cache_summary.json'); f = read(OUT/d/'feature_verification.json'); h = s['verification']['domains'][d]
        lines += [f"- {d}：缓存{c['seconds']:.1f}秒；独立核验{f['checked_feature_values']}个特征值，最大差异{f['max_feature_error']:.3g}；重算{h['heads']}个头、{h['checked_ranks']}个目标排名，纠错分最大差异{h['max_correction_error']:.3g}，排名全部一致。"]
    lines += [f"- 拟合/评价耗时{s['seconds']:.1f}秒，头与指标复核{s['verification']['seconds']:.1f}秒。数值复核由同一助手会话中的独立实现完成，不是外部独立审稿。", '',
        '## 解释范围与下一步', '',
        '本轮只回答当前状态在固定候选排序中的信息价值。原生轨迹上的效果、生成后重排效果、真正改变beam后的效果是不同问题。若后续推进，必须明确新增的信息或训练问题，并与同信号后重排、固定引导匹配，检查新增正确候选与误伤、运行代价；不能直接把前缀MLP命名为论文创新。', '',
        '旧第七阶段整请求优势开关、第十八阶段PCPS局部存活监督、第二十阶段启发式前缀专家权重已有失败记录。本轮未重启这些模型；它增加的是对实际当前全局竞争状态的信息消融。新机制仍需单独证据。', '',
        '## 统计解释核查：11/11', '',
        '1. 辛普森悖论：两域、OOF/开发、各折分别报告，不用合并均值覆盖异质性。',
        '2. 生态谬误：用户为评价单位，平均改进不推断每位用户获益；保存改善/受损人数。',
        '3. Berkson选择偏差：原生50候选及训练覆盖筛选限定支持域，评价不按覆盖删人。',
        '4. 碰撞偏差：不将条件于原模型候选覆盖的效果推广为全目录购买因果。',
        '5. 基准率：同时报告用户总数、目标覆盖、新增/丢失及不变人数。',
        '6. 均值回归：不挑最差用户或最好seed；所有预选用户和固定seed保留。',
        '7. 幸存者偏差：全部预选用户及方法均报告，未因失败表现删除配置。',
        '8. 多重寻找：两个预定时点、四组、三个seed全部呈现；192个配对比较区间只作描述，无多重校正或确认性显著性声称。',
        '9. 分叉路径：协议和输入代码在运行前冻结，CPU拟合/验证代码在GPU缓存过程中冻结；不按新结果改模型/门槛。',
        '10. 因果误读：消融局限于该模型、损失、预算；状态与专家互补若出现也不证明一般因果机制。',
        '11. 未来信息/反向解释：四维状态只读当步或此前，目标仅作训练标签；明确保留的最终底分与候选选择限制。', '',
        '用户配对bootstrap不覆盖共享训练折、上游随机性、重复开发研究或后续模型选择；三seed不是独立的三份用户数据。开发集此前用于研究，不能充当未触碰的最终测试。', '']
    REPORT.write_text('\n'.join(lines))
    # The detached coordinator updates its heartbeat after this command exits.
    files = [p for p in OUT.rglob('*') if p.is_file() and p.suffix != '.log'
             and p.name not in ('completion_manifest.json', 'cpu_pipeline_status.json')]
    files += list((ROOT/'experiment/phase20').glob('*lime_partial_state*.py'))
    files += [ROOT/'experiment/phase20/run_lime_partial_state.sh', REPORT,
              ROOT/'plan/第二十阶段/LIME当前生成状态替代最终信息_20260922.md']
    receipt = dict(state='VERIFIED', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))})
    (OUT/'completion_manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(report=str(REPORT), csv_rows=len(rows), passed_snapshots=passed)), flush=True)


if __name__ == '__main__': main()
