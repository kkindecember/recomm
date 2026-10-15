"""Publish all expert directions, signed sources and a target-free feedback prototype."""
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

from analyze_selective_evidence import sha, write, csv_write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/expert_evidence_alignment_20260926_v1'
REPORT = ROOT / 'report/第二十一阶段/专家实际依据与语义解释对齐检查_20260926.md'
NAMES = {'sas': 'SASRec', 'cf': 'ItemCF', 'text': '文本专家'}


def table(headers, rows):
    def safe(x): return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(safe, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(safe, r)) + ' |' for r in rows])


def short(value, n=125):
    value = ' '.join(html.unescape(value).split())
    return value if len(value) <= n else value[:n] + '…'


def link(name, title=None):
    return f'[{title or name}](../../artifacts/phase21/expert_evidence_alignment_20260926_v1/{name})'


def feedback(a, b, expert_scores, source_lists):
    """Only observed candidates, model scores and attributed history enter this API."""
    lines = [f'A：{short(a["title"])}', f'B：{short(b["title"])}', '']
    for e, score in expert_scores.items():
        lines.append(f'{NAMES[e]}：A={score["a"]:.6f}，B={score["b"]:.6f}，'
                     f'分差A−B={score["margin"]:+.6f}，方向={score["vote"]}。')
        if e == 'sas':
            lines.append('本卡仅有SASRec的打分方向，没有该模型的逐条历史归因。')
            continue
        for role in ('A', 'B'):
            selected = source_lists[e][role]
            if not selected:
                lines.append(f'未找到超过数值容差、相对支持{role}的差分历史。')
            for r in selected:
                lines.append(f'相对支持{role}：{r["ref"]}（历史位置{r["positions"]}）'
                             f'{short(r["title"])}；对原始分差的贡献={r["difference"]:+.6f}。')
        lines.append('上述最多列两边各3条；其余历史仍参与原分数，完整贡献保存在来源JSON。')
    lines += ['', '分数为各专家原有的目录内归一化分数；不同专家的原始贡献单位不同。',
              'ItemCF依据是训练共现邻居强度，文本依据是向量相似度；均不等于已确认的功能兼容或即时购买需要。',
              '保留两边依据与冲突，不由这张卡自动推断下一次需求。标题功效文字仅为原商品标称。']
    return '\n\n'.join(lines)


def main():
    assert not REPORT.exists() and not (OUT / 'completion_manifest.json').exists()
    summary = json.loads((OUT / 'summary.json').read_text())
    verification = json.loads((OUT / 'verification.json').read_text())
    assert verification['state'] == 'VERIFIED'
    cards = sorted(json.loads((OUT / 'cards.json').read_text()), key=lambda c: c['card_id'])
    by_id = {c['card_id']: c for c in cards}
    derived = {}
    for d in ('Toys', 'Beauty'):
        with (OUT / (d + '_users.csv')).open() as f:
            rows = list(csv.DictReader(f))
        # Descriptive joint counts; no new action rule is selected or tested.
        both = [r for r in rows if r['cf_vote'] != 'tie' and r['text_vote'] != 'tie']
        disagreement = [r for r in both if r['cf_vote'] != r['text_vote']]
        derived[d] = dict(users=len(rows), cf_text_both_decisive=len(both),
                          cf_text_opposite=len(disagreement),
                          cross_votes=dict(Counter(r['cf_vote'] + '/' + r['text_vote'] for r in rows)),
                          opposite_strata=dict(Counter(r['stratum'] for r in disagreement)))
    write(OUT / 'derived_joint_counts.json', derived)
    definite = [c for c in cards if c['review']['preference'] != 'abstain']
    alignment = []
    for c in definite:
        r, review = c['row'], c['review']
        row = dict(card_id=c['card_id'], domain=r['domain'], target_role=r['stratum'],
                   manual_choice=review['chosen_role'], manual_agrees_target=review['agrees_with_observed_target'])
        row.update({e: r[e + '_vote'] for e in NAMES})
        row['frozen_reason'] = review['reason']
        alignment.append(row)
    csv_write(OUT / 'all24_alignment.csv', alignment)
    feedback_cards = []
    for c in cards:
        r = c['row']
        # Pass a restricted payload without target, outcome or prior human judgment.
        payload = dict(a=c['a'], b=c['b'], expert_scores={
            e: {k: r[e + '_' + k] for k in ('a', 'b', 'margin', 'vote')} for e in NAMES},
            source_lists={e: {'A': c[e]['supports_a'], 'B': c[e]['supports_b']} for e in ('cf', 'text')})
        feedback_cards.append(dict(card_id=c['card_id'], inputs=payload, feedback=feedback(**payload)))
    write(OUT / 'feedback_cards.json', feedback_cards)
    (OUT / 'feedback_cards.md').write_text('# 128张固定模板反馈卡\n\n'
        '仅还原既有模型依据，不含真实目标、人工选择或新推荐决定；尚未输入GRAM。\n\n' +
        '\n\n'.join('## ' + c['card_id'] + '\n\n' + c['feedback'] for c in feedback_cards) + '\n')

    alignment_table = table(['与上轮24次明确判断的关系', 'SASRec', 'ItemCF', '文本专家'], [
        [label, *[summary['manual_alignment']['all24']['experts'][e][key] for e in NAMES]]
        for label, key in [('同向', 'agrees_manual'), ('反向', 'opposes_manual'), ('专家打分平局', 'tie')]])
    errors_table = table(['卡片', '原融合B', '专家候选A', '人工判断', '真实目标', 'SASRec', 'ItemCF', '文本'], [
        [cid, short(by_id[cid]['b']['title'], 65), short(by_id[cid]['a']['title'], 65),
         by_id[cid]['review']['chosen_role'], by_id[cid]['row']['stratum'][0].upper(),
         *[by_id[cid]['row'][e + '_vote'] for e in NAMES]] for cid in ('T016', 'B018', 'B029', 'B041')])
    example = by_id['B018']
    score_table = table(['专家', 'A：另一规格钻饰', 'B：美甲画笔', '偏向'], [
        [NAMES[e], f'{example["row"][e + "_a"]:.6f}', f'{example["row"][e + "_b"]:.6f}',
         example['row'][e + '_vote']] for e in NAMES])
    source_example = {e: {r['ref']: r for r in example[e]['sources']} for e in ('cf', 'text')}
    source_table = table(['同一条历史', 'ItemCF对A−B贡献', '文本对A−B贡献'], [
        [ref, f'{source_example["cf"][ref]["difference"]:+.6f}',
         f'{source_example["text"][ref]["difference"]:+.6f}'] for ref in ('H2', 'H3', 'H4')])
    full_table = table(['域', '全部用户', 'CF与文本都非平局', '其中两专家方向相反', '占非平局对数'], [
        [d, v['users'], v['cf_text_both_decisive'], v['cf_text_opposite'],
         f'{v["cf_text_opposite"] / v["cf_text_both_decisive"]:.2%}'] for d, v in derived.items()])
    contribution_rows = []
    for d in ('Toys', 'Beauty'):
        for e in ('cf', 'text'):
            s = summary['domains'][d]['all']['experts'][e]
            contribution_rows.append([d, NAMES[e], s['conflicting_events'], s['single_event_flip'],
                                      s['no_contribution'], f'{s["cancellation_quantiles"][2]:.4f}'])
    contribution_table = table(['域', '专家', '历史贡献正反并存', '删一条可能反向', '无差分贡献', '抵消比例中位数'], contribution_rows)
    target_table = table(['域', '真实目标在A/B的用户数', '专家', '与目标同向', '反向', '平局'], [
        [d, summary['domains'][d]['all']['a_hit'] + summary['domains'][d]['all']['b_hit'], NAMES[e],
         *[summary['domains'][d]['all']['experts'][e][k] for k in ('target_agree', 'target_disagree', 'target_tie')]]
        for d in ('Toys', 'Beauty') for e in NAMES])
    strata_table = table(['域', '层', '人数', 'SASRec同向／反向／平局', 'ItemCF同向／反向／平局', '文本同向／反向／平局'], [
        [d, label, summary['domains'][d]['strata'][label]['users'],
         *[' / '.join(str(summary['domains'][d]['strata'][label]['experts'][e][k])
                       for k in ('target_agree', 'target_disagree', 'target_tie')) for e in NAMES]]
        for d in ('Toys', 'Beauty') for label in ('a_hit', 'b_hit')])
    split_table = table(['域', '已审查／未审查', '用户数', 'A命中', 'B命中', '均未命中'], [
        [d, label, *[summary['domains'][d][label][k] for k in ('users', 'a_hit', 'b_hit', 'neither')]]
        for d in ('Toys', 'Beauty') for label in ('reviewed', 'not_reviewed')])
    all24_table = table(['卡片', '人工选择', '与真实目标一致', 'SASRec', 'ItemCF', '文本'], [
        [r['card_id'], r['manual_choice'], '不适用' if r['manual_agrees_target'] is None else '是' if r['manual_agrees_target'] else '否',
         r['sas'], r['cf'], r['text']] for r in alignment])
    report = f'''# 专家实际依据与语义解释对齐检查（21.18）

**这轮发现：上轮的文字判断明显更接近文本专家，不能当作三位专家意见的语义化汇总。** 24次明确判断中，21次与文本专家同向、13次与ItemCF同向、2次与SASRec同向。四次与目标不一致的人工判断全部与文本专家同向；但改成听SASRec也不能解决，因为上轮13次正确判断中，SASRec与其中12次反向。

在“钻饰还是画笔”的案例中，同样三条钻饰历史通过文本相似度支持钻饰，通过ItemCF训练共现支持画笔。**部分相反依据已经在现有专家中，不能只由文字相似关系推断“还缺新的需求信息”。** 原融合在这个案例本来正确；它是单侧语义意见可能误导替换的反例，不是“原融合已丢掉ItemCF”的证据。

## Material Passport

- Origin Skill：academic-research-suite，内联执行；用户授权继续本地分析。日期2026-09-26。
- 冻结方案：[21.18 协议](../../plan/第二十一阶段/21.18_专家实际依据与语义解释对齐检查_20260926.md)。沿用21.17固定A/B；A是三专家Top10并集中原融合Top10之外分数最高者，B是原融合第10名。不是原GRAM列表，也没有重新生成。
- 全量15399名Toys、17762名Beauty用户，共33161对；排除前期已展示的11位非中性案例用户。上轮128张卡均包含于此，另列其余用户。A选择、打分归因、反馈模板均不使用目标。
- 21.17结果已知，本轮不是新盲审；不修改其冻结关系、偏好和理由。单助手语义判断仍未经独立复评。
- 主计算{summary['seconds']:.2f}秒、数值复核{verification['seconds']:.2f}秒；CPU2线程。新拟合、训练、GPU推理均0，未读test，21.6继续USER_PAUSED。
- 状态COMPLETED，数值／来源复核VERIFIED；没有得到新的推荐性能提升，没有自动启动后续训练。

**先核对旧探索，避免重复**

[21.1关系方向](关系方向首轮探索_20260922.md)已比较前向、反向、无向、二值与近期加权关系，未通过冻结双域投入条件；[21.4B文本聚合](文本专家ABC优化_20260923.md)已比较共享历史与候选相关聚合，亦未通过原定双域条件。21.11–13已经拟合历史支持、集中度、近期性和类别等依据。本轮检查专家的真实分差来源，不再换一个名字重复这些训练。

**文字判断实际靠近哪位专家**

{alignment_table}

这些计数来自按命中结果富集的24次人工明确判断，不能推广为全体用户的偏好比例；相同方向也不能证明两个系统采用完全相同的理由。进一步查看逐条来源：在人工与文本专家同向的21例中，20例的文本最大同向历史出现在人工引用里；人工与ItemCF同向的13例中，对应计数为11例。它说明文字解释与模型来源有较多重合，仍不是独立语义正确性证明。

在目标确实位于A/B的17次明确判断中，人工与文本15次同向、与ItemCF9次同向（3次CF平局）、与SASRec1次同向。上轮13次与目标一致的判断中，文本同向11次，CF同向7次、平局3次，SASRec同向1次。这组结果不能支持统一改听另一位专家。

{errors_table}

**具体到同一段历史，两位专家如何得出相反意见**

B018：H2是MoYou钻饰，H3是珍珠甲饰，H4是Sodial钻饰；A为另一规格Sodial钻饰，B为美甲画笔。专家目录内归一化分数如下，只在每一行内比较A与B：

{score_table}

逐条历史对原始分差A−B的贡献如下。正值推向A，负值推向B；不同专家的数值单位不能直接比大小。

{source_table}

文本通过商品向量相似度，更偏向钻饰；ItemCF通过训练序列中建立的共现邻居，更偏向画笔。我们没有据此证明画笔与钻饰的功能因果或用户此刻的任务，只能确认**同一历史在两个模型中承担不同的评分依据**。实际目标为B，原融合也保留B。若把“三位专家意见”压成“历史反复买钻饰，建议A”，就漏掉了两位专家反对A的事实。

**真正被原融合排低的正确建议，以及它的反例**

B031的真实目标A是Toppik喷施器，B是Light Brown纤维。历史H3是Toppik Dark Brown纤维。ItemCF给A/B的归一化分数为{by_id['B031']['row']['cf_a']:.6f}/{by_id['B031']['row']['cf_b']:.6f}，来源为H3共现边；文本给A/B为{by_id['B031']['row']['text_a']:.6f}/{by_id['B031']['row']['text_b']:.6f}，更偏向形态类似的纤维；SASRec也偏向B。原融合保留B，A的正确机会被排低。这是可核对的专家冲突与遗漏案例，不能因为人工选择A就说文本专家也建议A。

B025则提醒不能一律相信配套共现：A为Seche Restore稀释剂，B为OPI甲油，历史有Seche Vite顶油。ItemCF和SASRec都偏向A，文本偏向B，实际目标为B，原融合也保留B。稀释剂适配顶油的来源关系存在，但缺少顶油变稠这一使用条件；前轮人工因此弃权。两个例子一起保留，不能只展示B031就宣称“配套专家覆盖相似专家”已经有效。

T064也有边界：文本更支持不兼容Pad2的Pad1胶套，CF和SASRec更支持Pad2主机，原融合正确保留后者。文本向量相似度不是可靠的硬件兼容判定；要把明确兼容字段作为新增依据，必须另验抽取正确性和决策增量。

**分歧是否只出现在几个案例**

{full_table}

这是输出生成后的描述性交叉计数，未据此选择采纳政策。分母仅限当前固定边界对，其中A本来就是被原融合排低的专家候选，不能推广到随机候选或所有推荐。CF平局的用户也完整保留于原CSV；表中非平局计数没有将他们算作CF赞成或反对。

{contribution_table}

“历史贡献正反并存”指同一专家内部，有历史让A−B增大，也有历史让它减小。Toys／Beauty文本分别有12063/15399（78.34%）、14414/17762（81.15%）出现这种情况。只展示最支持某个候选的一条历史，会省略实际参与打分的反向部分；这不证明反向历史一定有可解释的业务冲突，更不证明完整展示能提升GRAM。

“删一条可能反向”是在其余历史权重固定时做单条代数删除；没有重新压缩历史位置、重新运行融合或生成，不能将4621/5833名文本用户的方向变化说成推荐不稳定率或性能损失。“无差分贡献”指A−B的所有逐事件贡献都为0附近，可包含两边相同非零贡献，不等于模型对两个商品完全没有分数。抵消比例为1−|净分差|/绝对分差总和，无差分贡献者不进入其中位数。

**真实目标在候选中的全部用户，也不支持盲目改听某位专家**

{target_table}

以上每域仅149/176位目标在固定A/B中的用户；其余15250/17586人也保留在全量分析。它不是全目录排名准确率，不能忽略其他商品或以此排序三位专家的总体价值。按目标位置展开：

{strata_table}

上轮按结果富集的128人与其余人的基率不同，完整分母如下。其余人也仍属反复探索的validation，不是独立test。

{split_table}

**现在的反馈原型具体给什么**

已保存{link('feedback_cards.md', '128张可阅读反馈卡')}和{link('feedback_cards.json', '不含目标的模板输入与输出')}。每张固定给A/B信息、三专家各自分数方向、ItemCF与文本两边各最多3条实际差分来源，并说明未提供的SASRec历史归因和即时需求未知。完整逐条贡献另存{link('cards.json', '来源与原审查对照')}，不是靠一句“很相似”代替专家意见。

这只是可以核对的模板原型：来源标题加数学贡献由本地代码生成，没有外部语言模型，没有输入GRAM，尚未做语言简化或证明模型会理解。它包含完整历史归因计算；显示只列最强几条，不意味着其余历史未使用。文本向量依赖原目录编码内容，历史标题只是指向来源的索引，不能认定标题中的某个词就是向量分差原因。SASRec现在仍只有分数方向，不满足“三个专家都有语义解释”。

后续待检验的问题可以更具体：**在专家相互冲突时，保留双方实际依据和未知条件，是否比只给分数或只给一边理由更能帮助GRAM采纳正确意见、保住已有正确判断？** 这需要在相同候选、干预预算和训练条件下比较真实差分依据、同样篇幅的单侧依据、错配来源与分数对照。现在只有可追溯输入原型，尚未证明所需决策规律可学或生成收益成立；不据本轮结果直接重启21.6训练。

**全部24次明确判断的三专家方向**

{all24_table}

对应冻结原理由见{link('all24_alignment.csv', '全部24次对齐CSV')}；上轮所有弃权和失败也保留于[21.17完整报告](扩大分歧样本的目标隐藏证据审查_20260926.md)。

**来源与数值核验**

原CF／文本评分器重放128张卡的全目录输出，共3075200个商品分数。CF与缓存最大误差0；文本两域最大误差3.10e−6／3.81e−6，在原缓存核对容差内。128卡的分差分解与原评分器最大误差CF 3.71e−7、文本1.81e−7。

另一路核验27份输入哈希、33161个目标无关候选和三专家方向、12个分组汇总、128张来源卡、1133条逐事件计算及2266次固定权重删除。独立逐事件重算最大差异CF 0、文本1.11e−16。首轮复核中标量向量范数与原批量float32范数的舍入路径不同，导致约1.7e−7差异；修正核验器以遵循原float32运算，无放宽阈值、无改动实验结果。21.17冻结哈希保持不变。

{link('verification.json', '复核记录')}、{link('input_manifest.json', '来源哈希')}、{link('summary.json', '全部分层汇总')}、{link('derived_joint_counts.json', '专家方向交叉表')}、{link('Toys_users.csv', 'Toys全量逐用户')}、{link('Beauty_users.csv', 'Beauty全量逐用户')}。

**统计解释检查：11/11**

| 检查项 | 处理及限制 |
| --- | --- |
| 辛普森悖论 | 两域、A命中／B命中、已审查／未审查分别列出，不用混合比例替代 |
| 生态谬误 | 一用户一对候选；总体贡献统计不推断个体真实任务 |
| Berkson选择偏差 | A限定为专家建议且原融合排低的最高分候选，分歧率不能推广至随机商品对 |
| 碰撞偏差 | 目标位于A/B属于条件子集；仅作描述，不将其比例解释为因果或全目录性能 |
| 基率忽略 | 同时报告33161全量及149/176可二选一用户，保留大量两者均非目标者 |
| 回归均值 | 没有挑四个错误调规则后宣称改善，也无新政策前后提升结论 |
| 幸存者偏差 | 三专家、全部24判断、128卡、平局、弃权与全部用户均保存 |
| 多重比较／择优 | 固定三专家及两种可解析归因全部报告，不按结果选一位专家作裁判 |
| 分析分支 | 协议先写再运行；交叉表是后续描述，未用于新增规则、阈值或拟合 |
| 相关与因果 | 共现不是功能互补，余弦不是属性正确，代数删除不是购买行为因果 |
| 反向因果 | 归因只用既有模型和预测前历史，目标仅评价；但数据按用户留出而非全局时间切分，不能推断时间因果 |

已有[两域全量生成验证](两域全量验证_20260924.md)仍是原有效果结果；本轮没有新的NDCG提升。语义机制的执行状态继续保持暂停。
'''
    REPORT.write_text(report)
    dashboard = ROOT / 'report/第二十一阶段/在跑实验总览_20260923.md'
    marker = '<!-- expert_evidence_alignment_20260926 -->'
    assert marker not in dashboard.read_text()
    dashboard.write_text(marker + '\n**9月26日专家实际依据对齐完成：全量33161对候选，重放128张卡。上轮24次人工明确判断中，与文本／ItemCF／SASRec同向21／13／2次；四次错误都与文本同向，但改听SASRec也会反对13个正确判断中的12个。同一钻饰历史在文本中支持钻饰、在CF中支持画笔，不能将单侧相似解释当成三专家意见。已生成128张双方差分来源模板；见[完整结果与反馈原型](专家实际依据与语义解释对齐检查_20260926.md)。主计算39.66秒，来源／数值复核通过；模板未输入GRAM，无新拟合、训练或生成，21.6仍暂停。**\n\n' + dashboard.read_text())
    for name in ['expert_evidence_alignment_core.py', 'analyze_expert_evidence_alignment.py',
                 'verify_expert_evidence_alignment.py', 'summarize_expert_evidence_alignment.py']:
        source = ROOT / 'experiment/phase21' / name
        target = OUT / name
        if target.exists():
            assert sha(target) == sha(source)
        else:
            shutil.copy2(source, target)
    shutil.copy2(REPORT, OUT / REPORT.name)
    status = json.loads((OUT / 'status.json').read_text())
    status.update(verification_state='VERIFIED', report=str(REPORT.relative_to(ROOT)),
                  feedback_cards=128, feedback_delivered_to_GRAM=False,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / 'status.json', status)
    manifest = json.loads((OUT / 'input_manifest.json').read_text())
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != 'completion_manifest.json':
            manifest[str(p.relative_to(ROOT))] = sha(p)
    for name in ['verify_expert_evidence_alignment.py', 'summarize_expert_evidence_alignment.py']:
        p = ROOT / 'experiment/phase21' / name
        manifest[str(p.relative_to(ROOT))] = sha(p)
    manifest[str(REPORT.relative_to(ROOT))] = sha(REPORT)
    write(OUT / 'completion_manifest.json', manifest)
    print(json.dumps(dict(report=str(REPORT), feedback_cards=128, state='COMPLETED',
                          verification='VERIFIED', files_hashed=len(manifest)), ensure_ascii=False))


if __name__ == '__main__':
    main()
