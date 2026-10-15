"""Publish the frozen, outcome-hidden review with all judgments and limitations."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase21/blind_disagreements_20260925_v1'
REPORT = ROOT / 'report/第二十一阶段/扩大分歧样本的目标隐藏证据审查_20260926.md'
DOMAINS = ('Toys', 'Beauty')
LABELS = {'a_hit': 'A命中', 'b_hit': 'B命中', 'neither': '两者均未命中'}
RELATIONS = {
    'broad': '宽泛兴趣／大类', 'product_type': '商品类型／用途',
    'named': '品牌／主题／产品线／角色', 'attribute': '规格／版本／配方／颜色',
    'compatibility': '来源明示的适配', 'repeat': '同一目录商品曾出现',
    'none': '可见来源不足',
}


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csv_write(name, rows):
    assert rows
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, r)) + ' |' for r in rows])


def link(name, label=None):
    return f'[{label or name}](../../artifacts/phase21/blind_disagreements_20260925_v1/{name})'


def main():
    assert not REPORT.exists(), 'Do not overwrite a published review report'
    assert not (OUT / 'completion_manifest.json').exists()
    summary = read('summary.json')
    verification = read('verification.json')
    frozen = read('review_freeze.json')
    reviews = read('unblinded_reviews.json')
    cards = {c['card_id']: c for c in read('blind_cards.json')}
    review_metadata = read('frozen_review.json')
    assert verification['state'] == 'VERIFIED'
    assert sha(OUT / 'frozen_review.json') == frozen['frozen_review_sha256']
    assert sha(OUT / 'review_entries.json') == frozen['raw_review_sha256']
    assert len(reviews) == len(cards) == 128
    definite = [r for r in reviews if r['preference'] != 'abstain']
    failures = [r['card_id'] for r in definite if r['agrees_with_observed_target'] is False]
    assert failures == ['T016', 'B018', 'B029', 'B041']
    assert len(definite) == 24

    population_rows, strata_rows, relation_rows = [], [], []
    metrics = ['cards', 'either_specific', 'both_specific', 'contrast', 'preferences',
               'abstentions', 'contrast_but_abstain', 'informative',
               'informative_answered', 'informative_correct', 'informative_wrong',
               'neither_answered']
    overview_rows = []
    for domain in DOMAINS:
        part = summary['domains'][domain]
        overview_rows.append(dict(domain=domain, **{m: part['review'][m] for m in metrics}))
        for label in LABELS:
            p = part['population']['strata'][label]
            population_rows.append(dict(domain=domain, stratum=label, **p))
            r = part['strata'][label]
            strata_rows.append(dict(domain=domain, stratum=label, **{
                m: r[m] for m in ['cards', 'contrast', 'preferences', 'abstentions',
                                  'choose_a', 'choose_b', 'informative_correct',
                                  'informative_wrong', 'neither_answered']}))
    pooled = summary['pooled_sample']
    overview_rows.append(dict(domain='pooled_sample', **{m: pooled[m] for m in metrics}))
    for domain in [*DOMAINS, 'pooled_sample']:
        part = pooled if domain == 'pooled_sample' else summary['domains'][domain]['review']
        for rel in RELATIONS:
            relation_rows.append(dict(domain=domain, relation=rel, **part['relation_counts'][rel]))
    flat = []
    for r in reviews:
        c = cards[r['card_id']]
        flat.append(dict(card_id=r['card_id'], domain=r['domain'], stratum=r['stratum'],
                         X_title=c['X']['title'], Y_title=c['Y']['title'],
                         X_relations=';'.join(r['X_relations']), Y_relations=';'.join(r['Y_relations']),
                         X_history=';'.join(r['X_history']), Y_history=';'.join(r['Y_history']),
                         contrast=r['contrast'], preference=r['preference'], chosen_role=r['chosen_role'],
                         agrees_with_observed_target=r['agrees_with_observed_target'], reason=r['reason']))
    csv_write('population.csv', population_rows)
    csv_write('overview.csv', overview_rows)
    csv_write('strata.csv', strata_rows)
    csv_write('relation_types.csv', relation_rows)
    csv_write('all_reviews.csv', flat)

    natural_table = table(['域', '固定A/B用户数', 'A命中', 'B命中', '两者均未命中', '本轮抽样'], [
        [d, summary['domains'][d]['population']['primary_after_exclusions'],
         *[summary['domains'][d]['population']['strata'][k]['population'] for k in LABELS], '24 / 24 / 16']
        for d in DOMAINS])
    overview_labels = [
        ('审查对数', 'cards'), ('至少一边有具体历史关联', 'either_specific'),
        ('两边都有具体历史关联', 'both_specific'), ('能描述具体差别', 'contrast'),
        ('明确倾向某一边', 'preferences'), ('弃权', 'abstentions'),
        ('能描述差别、仍弃权', 'contrast_but_abstain'),
        ('真实目标在两候选之一', 'informative'),
        ('上述样本中作出明确判断', 'informative_answered'),
        ('明确判断与真实目标一致', 'informative_correct'),
        ('明确判断与真实目标不一致', 'informative_wrong'),
        ('两者均未命中层中的明确判断', 'neither_answered'),
    ]
    overview_table = table(['审查结果', 'Toys', 'Beauty', '合计'],
                           [[label, *[r[key] for r in overview_rows]] for label, key in overview_labels])
    strata_table = table(['域', '抽样层', '对数', '有差别', '明确判断', '选A', '选B', '弃权', '一致', '不一致'], [
        [r['domain'], LABELS[r['stratum']], r['cards'], r['contrast'], r['preferences'],
         r['choose_a'], r['choose_b'], r['abstentions'],
         r['informative_correct'] if r['stratum'] != 'neither' else '不适用',
         r['informative_wrong'] if r['stratum'] != 'neither' else '不适用'] for r in strata_rows])
    relation_table = table(['关系（可重叠）', 'Toys对数', 'Beauty对数', '合计', '两边都有该关系',
                            '明确判断数', '有目标的明确判断：一致／不一致'], [
        [RELATIONS[k], *[summary['domains'][d]['review']['relation_counts'][k]['cards'] for d in DOMAINS],
         pooled['relation_counts'][k]['cards'], pooled['relation_counts'][k]['both_sides'],
         pooled['relation_counts'][k]['preferences'],
         f"{pooled['relation_counts'][k]['informative_correct']} / {pooled['relation_counts'][k]['informative_wrong']}"]
        for k in RELATIONS])
    definite_table = table(['编号', '目标位置', '盲审倾向', '解封角色', '与目标一致', '冻结的原理由'], [
        [r['card_id'], LABELS[r['stratum']], r['preference'], r['chosen_role'],
         '不适用' if r['agrees_with_observed_target'] is None else '是' if r['agrees_with_observed_target'] else '否',
         r['reason']] for r in definite])
    description_cards = len(review_metadata['source_description_reads'])
    description_fields = sum(map(len, review_metadata['source_description_reads'].values()))
    max_history = max(c['history_events'] for c in cards.values())
    long_histories = sum(c['history_events'] > 20 for c in cards.values())

    report = f'''# 扩大分歧样本的目标隐藏证据审查（21.17）

128 对审查已完成。118 对至少有一边能找到具体历史关联，79 对两边都能找到；104 对能描述差别，但只有 24 对给出明确倾向。真实下一物品在两候选之一的明确判断共 17 次，13 次一致、4 次不一致。**能说明“它和历史有什么关系”，仍不等于知道“这一次为什么选它”。本轮没有得到可直接采用的新机制。**

**Material Passport**

- Origin Skill：academic-research-suite，内联执行；用户授权继续本地实验分析。
- 2026-09-25 开始，2026-09-26（北京时间）完成；运行目录沿用启动日编号。协议为[21.17 固定审查方案](../../plan/第二十一阶段/21.17_扩大分歧样本的目标隐藏证据审查_20260925.md)。
- 材料：21.11 主要验证用户与候选缓存、两域完整已知历史、phase20 来源商品目录，以及21.14–21.16已展示案例的排除名单。
- 同一助手逐张阅读源卡、作出判断；隐藏结果与角色，不是独立审查者或双盲实验。完整审查包含所有128张，无自动填入的未审记录。
- 统计拟合、GRAM／专家训练、GPU推理均为0；未读取test；21.6语义反馈训练仍为USER_PAUSED。
- 本轮是回顾性、按命中结果富集的诊断样本。没有新推荐模型、没有重新运行GRAM，不能将这里的人工选择计为GRAM性能提升。

**先固定比较对象，再抽样、隐藏目标、冻结判断**

A：从三专家Top10并集中、原融合Top10之外的商品中，取原融合分最高者；并列按商品ID。B：原融合第10名。每位用户只有一个固定A/B，确定A时不读取目标；随后才用目标划分A命中、B命中、两者均未命中三层。

本轮比较的是**已有融合的边界候选**，B不等于原始GRAM的第10名，也不等于当前生成引导列表的第10名。不能据此声称新生成策略的收益或损失。

排除先前展示的非中性案例用户，Toys 7 人、Beauty 4 人。剩余固定候选总体及抽样如下：

{natural_table}

从每层按冻结哈希抽样，随机交换候选的X/Y位置，隐藏用户ID、目标、分数、A/B原角色。全部完整历史进入源卡：重复商品合并显示，但保留所有出现位置，共1133条卡内去重的历史商品记录；最长历史{max_history}次，{long_histories}张卡超过20次。不是只看最近商品。标题／类别／品牌为主要材料，另补读{description_cards}张卡的{description_fields}个来源描述字段，留有清单；没有对所有商品描述做穷尽审核。

两域合计33161个固定A/B与上一轮“三专家并集内任一遗漏目标”的1597个机会，候选选择与分母不同，不能互相当覆盖率分母。富集样本也没有保留自然命中比例：每域人为抽24/24/16人，不能直接计算总体准确率、总体净收益或拿它与原GRAM的NDCG对比。

冻结时间 UTC {summary['frozen_at_utc']}；解封时间 UTC {summary['unblinded_at_utc']}。冻结判断 SHA256：`{frozen['frozen_review_sha256']}`。解封后没有修改关系标签、偏好或理由。

**把“有差别”和“足以选择”分开记录**

具体关联包含商品类型／用途、明示名称、属性、适配、同目录重复；只有宽泛兴趣不算具体关联。“有差别”只表示源文或历史关系有可描述区别，不代表该区别决定本次需求；“明确倾向”是单一审查者的偏好假设，不是用户真实意图标签。证据不足可弃权，未知字段不当成不匹配。

{overview_table}

104对“有差别”和104对“弃权”是两个不同的计数，数值相同是巧合；其中80对同时满足两者。24次明确判断中，7次的真实目标不在X/Y里，不能把它们当成二选一错误，也不能忽略它们只报13/17。即使目标在候选中，也有79/96对弃权。13/17只描述这17次选择与单个观测目标的一致程度，不证明一般准确率。

**每层结果全部保留**

{strata_table}

两域各24个“A确实是目标”的抽样案例中，审查分别只选择了3次A；在各24个“B确实是目标”的案例中，Toys未主张换A，Beauty仍有3次主张换A。说明提高解释具体程度后，仍会提出替换原融合正确候选的建议。Toys还有1次偏向保留B而错过实际正确的A，保守本身也不能解决全部问题。

两者均未命中层中，真实目标还可能在原Top9；核对后每域16人中有2人属于此类，其余14人也不在原Top9。这里仅讨论A/B选择，不将“未命中A/B”误写为“原推荐必然没命中”。

**四次不一致，全部展开**

| 卡片 | 判断前看到了什么 | 冻结判断 | 解封结果 | 能说明的边界 |
| --- | --- | --- | --- | --- |
| B018 | H2–H4连续出现美甲钻饰，A是同品牌另一规格钻饰，B是美甲画笔 | 更偏向A的重复商品用途 | B才是观测目标；会误替换原融合正确候选 | 历史经常买钻饰，不代表下一次还买钻饰；用途能区分，当前任务没确定 |
| B029 | 历史有多款维C面部精华／面部护理；A是维C面霜，B标题为身体护理霜 | 更偏向A的面部用途 | B才是观测目标；会误替换 | 成分和使用部位的差别可核对，但历史常见部位不决定本次购买部位 |
| B041 | 至少5条历史是维C面部精华；A也是该类型，B是洁面海绵 | 更偏向A的形态与成分 | B才是观测目标；会误替换 | 重复同种产品的依据，未能区分继续买该类与选择另一护理步骤 |
| T016 | 最近H32是Black and Decker Jr电钻，B为同产品线电动拼图锯；A是Spider-Man发射器 | 更偏向B的近期系列关系 | A才是观测目标；会错过专家建议 | 偏向保留相似旧候选也会错，不能只防过度采纳A |

前三例的“继续同类／另一用途”是对商品差别的描述，不能倒推用户真实心理或宣称发生了确定的任务切换。Beauty标题中的功效文字仅作来源标称，不据此认定产品功效或用户身体状况。T016目录把工具标题归入jigsaw puzzles，审查明确按标题识别，保留了原始类别错误。

**确实更具体的证据，以及仍需弃权的情形**

T064：候选X为Leappad1胶套，描述明确不兼容Leappad2；Y为Leappad2主机，历史携包和胶套写明覆盖Pad2。盲审偏向Y，解封后Y为B且为目标。可核对的适配信息支持了这次选择；仍不能从配件历史断定用户不拥有Pad1或必定需要新主机。

B031：历史有Toppik Dark Brown纤维，Y喷施器的描述明确用于Toppik纤维；X是Light Brown纤维，与历史两处Dark Brown标称不同。盲审倾向Y，解封后Y为A且为目标。它是明示系统关系加颜色差别的例子，不是“兼容信息已经稳定有效”的证明。

B025：来源明确Seche Restore可稀释历史里的Seche Vite顶油，适配关系是真的；但不知道旧顶油是否变稠，无法因此判定它比新色甲油更需要，故弃权。**商品适配与此刻需要购买是两件事。**

B013：两个候选都是WEN Fig，32oz对6oz；历史有16oz Fig及其他WEN规格。大小差别明确，旅行、预算、剩余量未观测，故弃权。T015：一个候选同Generations系列，另一个同Deluxe级别及Bumblebee角色，维度冲突，弃权。T021：历史里两套娃娃屋体系都有支持，不能确定接下来扩充哪套。B048：历史有多套印花板，不足以断言现在需要收纳盒而非另一套印花板，也无尺寸适配证明。

**预先规定的关系类型，全量列出**

{relation_table}

同一卡可以多标签，不能将行数相加当样本数，也不是三个专家贡献消融。表中“一致／不一致”指含有该关系的卡片，不是该关系本身造成结果。适配关系共6对，只有2次明确判断、恰好都一致，事件太少，不能从中挑出一个已验证的成功机制。repeat=0仅指此样本中X/Y与历史没有同一目录ID的重复，不代表不存在同类复购或全库不存在重复。

**对机制设计的含义**

目前证据支持收窄问题：专家的商品分数可以指出值得比较的候选，商品／历史文本可以揭示类型、版本、配套关系，但这些关系不总能决定当前这次选择。把“和历史很像”写成流畅文字，仍可能重复同一个错误。这轮不能证明GRAM读不懂语义，也不能证明语义反馈没有用——尚未让GRAM读取这些证据并生成。

后续值得检验的假设是：在原融合与专家建议出现分歧时，明确比较A和B各自的来源依据、冲突和缺失条件，检验这些信息是否在既有分数以外帮助采纳或保留。比如“同系列”与“可接现有物品”应分开，“适配但当前是否需要未知”应保留未知。若以后生成此类反馈，应由可追溯的历史／商品字段产生可核对事实，不能由语言模型编造即时需求。

这仍只是待检验设计：抽取正确、决策有增量、GRAM实际使用有效是三个不同环节。本轮仅完成第一个环节的有限人工检查及决策边界诊断，未证明三个环节已经成立。后续若实施，需预先冻结规则，在未用于此次改法选择的评价材料上比较真实依据、错配依据和相同干预预算的分数对照，同时报告新增与丢失；不能用本轮4个失败例修改规则后，在同一批样本宣布修复。此轮按协议结束，未自动启动训练或下一轮机制实验。

**全部24次明确判断（不只展示成功例）**

{definite_table}

128条完整审查及弃权理由见{link('all_reviews.csv', '完整CSV')}、{link('unblinded_reviews.json', '完整解封JSON')}；24条连同来源卡见{link('all_definite_judgments.json', '全部明确判断及历史')}。全部源卡见{link('blind_cards.json', '隐藏结果的原卡片')}，原描述见{link('source_details.json', '商品来源字段')}，补读清单与冻结判断见{link('frozen_review.json', '冻结审查')}。

**数值、来源核验与推断限制**

独立重算脚本核验24份来源哈希、33161个目标无关的固定候选选择、哈希抽样和随机左右映射、1133条历史商品及完整位置、128条决定、63个关系汇总行、9个汇总分组，全部通过。程序核验的是可追溯性与计数，不是另一位审查者重新认定语义正确性；单一助手的主观判断没有独立复评或一致性系数。

| 统计与推断检查 | 本轮处理及剩余限制 |
| --- | --- |
| 1. 分组混合／辛普森悖论 | Toys、Beauty及三种命中层分别报告，合并数仅描述抽样材料 |
| 2. 伪重复／分析单位 | 一用户固定一对候选；历史商品与多标签不作为独立用户样本 |
| 3. 选择偏差／碰撞条件 | 先固定分数最高的遗漏A，再按目标富集；只代表该选取规则下的诊断样本 |
| 4. 条件比例误当总体或因果 | 13/17不是总体准确率；人工一致不等于机制增益或因果解释 |
| 5. 基率忽略 | 同时提供两域自然层人数65/84/15250与88/88/17586，不能用24/24/16替代 |
| 6. 回归均值／反复探索 | 仍是反复使用的validation；排除旧案例不使其变为新test，不报独立确认 |
| 7. 幸存者偏差／只选案例 | 发布全部128条、全部24次明确判断、全部4次不一致、全部弃权 |
| 8. 多重比较／事后择优 | 七类预先标签全部报告且说明重叠；不选择适配2/2作为成功政策 |
| 9. 分析分支／反复改阈值 | 判断先哈希冻结后解封，解封后不改标、不补测新规则 |
| 10. 相关代替原因 | 历史相似或适配不证明当前需要、真实意图或功效，不从结果倒写原因 |
| 11. 标签泄漏／时间顺序 | 目标仅用于固定A之后的分层抽样及解封评价；审查时不展示目标或A/B角色；同助手既有背景知识仍是限制 |

来源冻结：{link('preparation_manifest.json', '准备哈希')}、{link('review_manifest.json', '审查哈希')}、{link('review_freeze.json', '冻结记录')}、{link('analysis_manifest.json', '分析哈希')}。核验：{link('verification.json', '独立计数与来源复核')}。结构化汇总：{link('summary.json')}、{link('population.csv')}、{link('overview.csv')}、{link('strata.csv')}、{link('relation_types.csv')}。

已有全量生成结论仍以[两域全量验证](两域全量验证_20260924.md)为准：生成引导相对原GRAM的NDCG@10提升Toys 14.23%、Beauty 9.48%，且两域均优于同信号后重排；这是单上游种子的validation结果。本轮未改变该结果，也未得到新的语义反馈生成提升。
'''
    REPORT.write_text(report)

    dashboard = ROOT / 'report/第二十一阶段/在跑实验总览_20260923.md'
    marker = '<!-- blind_disagreements_20260926 -->'
    assert marker not in dashboard.read_text()
    block = marker + '\n**9月26日128对分歧的目标隐藏审查完成：Toys／Beauty各64对；104对能描述差别、24对给出明确倾向。目标在两候选之一的17次明确判断中，13次一致、4次不一致；Beauty有3次偏向更像历史的专家候选，却会替换原融合正确的第10名。此为按结果富集、同一助手审查的诊断样本，不能当全量准确率或新GRAM提升。见[完整结果、四次不一致与全部判断](扩大分歧样本的目标隐藏证据审查_20260926.md)。128条判断先冻结再解封；来源／抽样／计数复核通过，语义判断无独立复评。无新拟合、训练或GPU生成，21.6仍暂停；本轮已结束，未自动启动后续实验。**\n\n'
    dashboard.write_text(block + dashboard.read_text())
    for name in ['prepare_blind_disagreements.py', 'freeze_blind_disagreements.py',
                 'analyze_blind_disagreements.py', 'verify_blind_disagreements.py',
                 'summarize_blind_disagreements.py']:
        source = ROOT / 'experiment/phase21' / name
        target = OUT / name
        if target.exists():
            assert sha(target) == sha(source)
        else:
            shutil.copy2(source, target)
    shutil.copy2(REPORT, OUT / REPORT.name)
    status = read('status.json')
    status.update(verification_state='VERIFIED', semantic_annotation_independently_verified=False,
                  report=str(REPORT.relative_to(ROOT)), completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  report_local_date='2026-09-26', final_report_complete=True, next_experiment_started=False)
    write(OUT / 'status.json', status)
    manifest = read('analysis_manifest.json')
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != 'completion_manifest.json':
            manifest[str(p.relative_to(ROOT))] = sha(p)
    manifest[str(REPORT.relative_to(ROOT))] = sha(REPORT)
    for name in ['verify_blind_disagreements.py', 'summarize_blind_disagreements.py']:
        p = ROOT / 'experiment/phase21' / name
        manifest[str(p.relative_to(ROOT))] = sha(p)
    write(OUT / 'completion_manifest.json', manifest)
    print(json.dumps(dict(report=str(REPORT), state='COMPLETED', verification='VERIFIED',
                          frozen_review_unchanged=True, all_reviews=len(flat),
                          definite_judgments=len(definite), failures=failures,
                          final_manifest_files=len(manifest)), ensure_ascii=False))


if __name__ == '__main__':
    main()
