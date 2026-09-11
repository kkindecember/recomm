"""Persist current/final stage-20 reports without changing any experiment."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
P20 = ROOT / 'artifacts/phase20'
REPORT = ROOT / 'report/第二十阶段'
TERMINAL = {'COMPLETED', 'FAILED', 'STOPPED', 'ORPHANED'}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def progress(status):
    if not status:
        return '尚未启动。'
    state = status.get('state', 'UNKNOWN')
    message = f"状态：**{state}**。状态时间：{status.get('time', status.get('updated_at', '未知'))}。"
    if status.get('stage'):
        message += f" 当前步骤：`{status['stage']}`。"
    if status.get('examples') is not None and status.get('total'):
        message += f" 已处理{status['examples']:,}/{status['total']:,}（{100*status['examples']/status['total']:.1f}%）。"
    if status.get('error'):
        message += f" 记录：`{str(status['error'])[:500]}`。"
    return message


def eta(directory, status):
    if status.get('state') != 'RUNNING' or not status.get('total'):
        return ''
    try:
        lines = (directory/'events.jsonl').read_text().splitlines()
    except OSError:
        return ''
    events = []
    for line in lines[-200:]:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('stage') == status.get('stage') and event.get('total') == status.get('total') and event.get('examples') is not None:
            events.append(event)
    if len(events) < 2:
        return ''
    a,b = events[-8:][0],events[-1]
    elapsed = b['elapsed_seconds']-a['elapsed_seconds']
    count = b['examples']-a['examples']
    if elapsed <= 0 or count <= 0:
        return ''
    rate = count/elapsed
    remaining = max(0,b['total']-b['examples'])/rate/3600
    return f'近期速度约{rate:.2f}条/秒，当前步骤预计还需{remaining:.1f}小时。仅当前步骤外推，不含后续处理，也不是完成保证。\n\n'


def table(methods):
    lines = ['| 方法 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | Hit@5 | Hit@10 | Hit@20 | Hit@50 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ['ndcg@5','ndcg@10','ndcg@20','ndcg@50','hit@5','hit@10','hit@20','hit@50']
    for name, values in methods.items():
        lines.append('| '+name+' | '+' | '.join(f'{values[k]:.6f}' for k in keys)+' |')
    return '\n'.join(lines)


def compare_line(name, comparison):
    if not comparison:
        return ''
    n, h = comparison['ndcg@10'], comparison['hit@10']
    ci = n['paired_bootstrap_95ci']
    hci = h['paired_bootstrap_95ci']
    return (f"- {name}：N10绝对变化{n['delta']:+.6f}，描述性95%区间[{ci[0]:+.6f}, {ci[1]:+.6f}]；"
            f"H10变化{h['delta']:+.6f}，区间[{hci[0]:+.6f}, {hci[1]:+.6f}]。")


def verdict(delta, hit_delta):
    if delta > 0:
        return 'NDCG@10出现探索正点估计；'+('Hit@10也上升。' if hit_delta > 0 else 'Hit@10没有上升，须保留这一取舍。')+'这不是独立确认。'
    return '此配置未提供NDCG@10正增益；'+('Hit@10上升，指标方向存在取舍。' if hit_delta > 0 else 'Hit@10也未上升。')


def full_block(full, names=None):
    names = names or {'original_gram':'原GRAM','original_gram_pcrf':'原GRAM+PCRF','raw':'当前模型','pcrf':'当前模型+PCRF'}
    text = f"全量validation用户数：{full['raw']['users']:,}。\n\n"+table({names[k]: full[k] for k in names})+'\n\n'
    for label, value in full.get('descriptive_paired_ci', {}).items():
        text += compare_line(label, value)+'\n'
    return text


def selection_block(directory):
    selection = read(directory / 'selection.json')
    rows = selection.get('outcomes', [])
    if not rows:
        return ''
    lines = ['已有选择validation轨迹（3000人，非全量终态）：', '',
             '| 阶段 | raw N10 | 相对同用户父模型 | +PCRF N10 |', '|---|---:|---:|---:|']
    for row in rows:
        delta = 100*(row['raw']['ndcg@10']/row['original_gram']['ndcg@10']-1)
        lines.append(f"| {row['tag']} | {row['raw']['ndcg@10']:.6f} | {delta:+.2f}% | {row['pcrf']['ndcg@10']:.6f} |")
    return '\n'.join(lines)+'\n'


CAVEAT = ('本报告由后台观察程序依据已落盘状态和summary自动更新。ANALYZED表示读取并分析现有产物；'
          '新评估复算指标不等于重新训练复现。只使用train/validation，未读取test。'
          'validation长期重复开发，所有配对区间仅为描述性用户不确定性，未校正多重比较，不能替代独立域或训练seed确认。\n')


def snapshot():
    now = time.strftime('%Y-%m-%d %H:%M:%S %z')
    header = f'更新时间：{now}。\n\n'
    outer = read(P20 / 'status.json')
    receipt = read(P20 / 'sdpo_gram_toys_v1/stop_receipt.json')
    gacr_config = read(ROOT/'experiment/phase20/gacr_v6_pcrf_toys_v1.json')
    gacr_name = Path(gacr_config.get('output','artifacts/phase20/gacr_v6_pcrf_toys_v1')).name
    gacr_smoke = Path(gacr_config.get('smoke_output','artifacts/phase20/gacr_v6_pcrf_toys_smoke_v1')).name
    sft_dir, gacr_dir = P20 / 'round1_sft_full_v1', P20 / gacr_name
    sft_status, gacr_status = read(sft_dir / 'status.json'), read(gacr_dir / 'status.json')
    sft, gacr = read(sft_dir / 'summary.json'), read(gacr_dir / 'summary.json')
    queue = read(P20 / 'followup_queue/status.json')
    smoke = read(P20 / gacr_smoke / 'status.json')

    sft_text = '# Stage20 round1_sft全量验证报告\n\n'+header+progress(sft_status)+'\n\n'+eta(sft_dir,sft_status)
    sft_text += ('固定模型：`sprec_gram_toys_v1/round1_sft.pt`，SHA256 `d88678743b9d4650057647aa644ecb3e78dd82dcdb4138642f9a85e671e20193`。'
                 '完成第1轮SFT的855次训练更新；本任务零新增训练。GPU0，FP32/TF32关闭、原生beam50，复用同精度父参考。'
                 '该模型不是三轮SPRec的训练预算匹配SFT控制。12小时硬预算，无自动重试。\n\n')
    parity = read(sft_dir / 'smoke.json')
    sft_text += f"工程检查：{parity.get('state','等待检查')}；16名已有用户的候选和raw/PCRF rank须精确一致。\n\n"
    if sft:
        sft_text += full_block(sft['full_validation'])
        full = sft['full_validation']
        sft_text += '\n结论：'+verdict(sft['raw_ndcg_delta_vs_gram'], full['raw']['hit@10']-full['original_gram']['hit@10'])+'\n'
        sft_text += '组合相对原PCRF：'+verdict(sft['combined_ndcg_delta_vs_original_pcrf'], full['pcrf']['hit@10']-full['original_gram_pcrf']['hit@10'])+'\n'
    else:
        sft_text += '尚无全量终态结果，暂不判断SFT是否提点。\n'
    sft_text += '\n配置：[round1_sft_full_v1.json](../../experiment/phase20/round1_sft_full_v1.json)。证据目录：`artifacts/phase20/round1_sft_full_v1/`。\n\n'+CAVEAT
    write(REPORT / 'Stage20_round1_SFT全量验证报告.md', sft_text)

    sprec_text = '# Stage20 SPRec运行与最终结果报告\n\n'+header+progress(outer)+'\n\n'
    sprec_text += '原固定协议：Toys三轮；全量raw或相对PCRF的新增N10点估计为正时，原监督程序自动进入Beauty。此报告程序不改变训练、选模或触发规则。\n\n'
    for domain in ['toys', 'beauty']:
        directory = P20 / f'sprec_gram_{domain}_v1'
        status, summary = read(directory / 'status.json'), read(directory / 'summary.json')
        sprec_text += f'## {domain.title()}\n\n'+progress(status)+'\n\n'
        sprec_text += selection_block(directory)+'\n'
        if summary:
            sprec_text += '选中checkpoint：`'+json.dumps(summary['best'],ensure_ascii=False)+'`。\n\n'
            for tag, full in summary['full_validation'].items():
                sprec_text += f'### {tag} 全量结果\n\n'+full_block(full)+'\n'
            raw = summary['full_validation'][summary['best']['raw']['tag']]
            combo = summary['full_validation'][summary['best']['pcrf']['tag']]
            sprec_text += '模型单独：'+verdict(summary['raw_ndcg_delta_vs_gram'],raw['raw']['hit@10']-raw['original_gram']['hit@10'])+'\n\n'
            sprec_text += '组合相对原PCRF：'+verdict(summary['combined_ndcg_delta_vs_original_pcrf'],combo['pcrf']['hit@10']-combo['original_gram_pcrf']['hit@10'])+'\n\n'
            sprec_text += f"原协议决策：`{summary.get('decision','见summary')}`；预算完成不代表充分收敛。\n\n"
        elif status:
            sprec_text += '尚未生成该域全量summary，不作最终增益结论。\n\n'
    sprec_text += CAVEAT
    write(REPORT / 'Stage20_SPRec运行与最终结果报告.md', sprec_text)

    gacr_text = '# Stage20 GACR-v6与PCRF组合评估报告\n\n'+header+progress(gacr_status)+'\n\n'+eta(gacr_dir,gacr_status)
    gacr_text += f"队列：{progress(queue)}\n\n工程检查：{progress(smoke)}\n\n"
    gacr_text += ('固定Toys全量19,412用户、residual seed2023。复用C1、GACR-v6、PCRF，无新增训练。'
                  'C1是继续训练过的骨干，目录头按原文件顺序映射商品；并非未经修改的原GRAM。'
                  '组合为GACR top50内的冻结PCRF，base signal使用GACR分数，参数lambda1/beta0.5/gamma1/q1=5，不搜索参数。'
                  '正式任务使用当前机器配置指定的可用GPU，与SFT并行，24小时硬预算，无自动跨域或重试。\n\n')
    gacr_text += ('工程过程：首次实现吞吐偏低，显式停止并保留原目录；批量特征版在GPU5的首次检查遇到共享显存骤降导致OOM，失败目录保留。'
                  'GPU6的16人候选、base、特征及最终排序一致性检查已通过，父参考PCRF rank也16/16一致。'
                  '同GPU配对计时原实现/批量版约1.910/1.807秒每用户，不能把先前全部减速归因于代码；共享资源影响明显。工程检查不算推荐提点。\n\n')
    if gacr:
        gacr_text += table(gacr['metrics'])+'\n\n'
        for name, comp in gacr['paired_comparisons'].items():
            gacr_text += compare_line('组合相对'+name, comp)+'\n'
        values = gacr['metrics']
        gacr_text += '\n结论：'+verdict(gacr['combined_ndcg_delta_vs_original_pcrf'],values['gacr_v6_pcrf']['hit@10']-values['original_pcrf']['hit@10'])+'\n\n'
        for name, group in gacr['groups'].items():
            if name == 'all_validation':
                continue
            gacr_text += f'## 分组：{name}\n\n'+table(group['metrics'])+'\n\n'+compare_line('组合相对原PCRF',group['combined_vs_original_pcrf'])+'\n\n'
    else:
        gacr_text += '尚无正式组合全量结果，不将工程检查或历史缓存正信号写成本次提点。\n\n'
    gacr_text += f'计划：[GACR-v6/PCRF组合评估计划](../../plan/第二十阶段/GACR_v6_PCRF组合评估计划.md)。证据目录：`artifacts/phase20/{gacr_name}/`。\n\n'+CAVEAT
    write(REPORT / 'Stage20_GACR_v6_PCRF组合评估报告.md', gacr_text)

    total = '# GRAM 第20阶段实验总报告\n\n'+header
    total += ('## S-DPO：已按用户要求停止\n\n'
              f"停止时间：{receipt.get('stopped_at','待核对')}。已完成2个epoch和第3epoch的407次更新（52,096条前缀），总更新2117。"
              '最近checkpoint、optimizer及随机状态保留，原任务不重启。前两轮同一3000人队列raw N10相对父模型-46.29%/-86.38%；'
              'epoch2加PCRF后相对原PCRF仍-70.39%。第3轮未完成、无正式全量验证，不写成完整收敛失败。\n\n'
              '凭据：[停止记录](../../artifacts/phase20/sdpo_gram_toys_v1/stop_receipt.json)；[中期逐用户审计](第20阶段_20260911中期审计与继续停止建议.md)。\n\n')
    total += '## SPRec\n\n'+progress(outer)+'\n\n'
    for domain in ['toys','beauty']:
        directory = P20 / f'sprec_gram_{domain}_v1'
        summary = read(directory / 'summary.json')
        total += f"{domain.title()}：{progress(read(directory/'status.json'))}\n\n"
        if summary:
            total += f"全量结论：raw N10相对原GRAM {summary['raw_ndcg_delta_vs_gram']:+.6f}；组合N10相对原PCRF {summary['combined_ndcg_delta_vs_original_pcrf']:+.6f}。原协议决策 `{summary.get('decision')}`。\n\n"
        elif domain == 'toys':
            total += selection_block(directory)+'\n'
    total += '[SPRec完整指标及最终结论](Stage20_SPRec运行与最终结果报告.md)。\n\n'
    total += '## 已保存round1_sft全量验证\n\n'+progress(sft_status)+'\n\n'
    if sft:
        total += full_block(sft['full_validation'])+'\n'
    total += '[SFT专项报告](Stage20_round1_SFT全量验证报告.md)。\n\n'
    total += '## GACR-v6与PCRF组合\n\n'+progress(gacr_status)+'\n\n'+progress(queue)+'\n\n'
    if gacr:
        total += table(gacr['metrics'])+f"\n\n决策：`{gacr['decision']}`。\n\n"
    total += '[组合计划](../../plan/第二十阶段/GACR_v6_PCRF组合评估计划.md)；[组合专项报告](Stage20_GACR_v6_PCRF组合评估报告.md)。\n\n'
    total += '## 解释与验证边界\n\n'+CAVEAT
    total += '各结果只在相同用户、相同参考下比较。分组、命中得失和未完成阶段须保留；不按最优checkpoint宣布所有用户改善，不把模型退化症状当成单一原因的因果证明。\n'
    write(REPORT / 'GRAM_第二十阶段_实验总报告.md', total)
    done = outer.get('state') in TERMINAL and sft_status.get('state') in TERMINAL and queue.get('state') in TERMINAL
    status = {'state': 'COMPLETED' if done else 'RUNNING', 'pid': os.getpid(), 'updated_at': now,
              'all_monitored_jobs_finished': done, 'refresh_seconds': 30,
              'reports': ['GRAM_第二十阶段_实验总报告.md','Stage20_SPRec运行与最终结果报告.md','Stage20_round1_SFT全量验证报告.md','Stage20_GACR_v6_PCRF组合评估报告.md']}
    write(P20 / 'report_watch/status.json', json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    started = time.time()
    while True:
        if snapshot() or args.once:
            return
        if time.time()-started > 7*86400:
            status = read(P20 / 'report_watch/status.json')
            status['state'] = 'STOPPED'
            status['reason'] = '7-day reporting limit; workload statuses remain authoritative'
            write(P20 / 'report_watch/status.json', json.dumps(status,ensure_ascii=False,indent=2)+'\n')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
