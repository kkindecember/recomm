"""Nine-case frozen-frontier scoring audit; bounded, no parameter updates."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from experiment.phase18.protocol import s18_beam_boundary_diagnostic as d
from experiment.phase18.protocol import s18_s2_mechanism_probe as s2
from experiment.phase18.core.contracts import ROOT, load_json, sha256
from experiment.phase18.protocol.s18_s1_prepare import atomic_json, atomic_text, utc_now

BASE = ROOT / 'artifacts/phase18/score_numerics_audit'
CONFIG = ROOT / 'experiment/phase18/config/s18_score_numerics_audit.json'
CASES = ROOT / 'artifacts/phase18/beam_boundary_followup_design/case_manifest.json'
MODULE = 'experiment.phase18.protocol.s18_score_numerics_audit'
STATUS = ROOT / 'artifacts/phase18/status/s18_score_numerics_audit.status.json'
LEDGER = ROOT / 'artifacts/phase18/attempts/score_numerics_audit.attempts.jsonl'
REPORT = ROOT / 'report/第十八阶段/Stage18_固定案例评分数值复核报告.md'
ADDENDUM = ROOT / 'plan/第十八阶段/GRAM_固定案例评分数值复核_执行补遗v0.1.md'
CONDITIONS = ['frontier_cached', 'frontier_full', 'pair_full_shared', 'pair_full_duplicate_encoder',
              'pair_cached', 'single_full']
SEALED = dict(d.SEALED, new_frontier_selected=False, gradient_recomputed=False,
              s18_2_decision_unchanged='FAILED_SCIENTIFIC_GATE',
              original_gradient_measurement_status='MEASUREMENT_UNRESOLVED')


def verify():
    cfg = load_json(CONFIG)
    _, old = d.verify()
    for item in cfg['inputs']:
        s2.checked(item)
    for item in load_json(d.BASE / 'run-0001/run_manifest.json')['sources']:
        s2.checked(item)
    cases = load_json(CASES)
    if cases['actual_cases'] != 9 or cfg['conditions'] != CONDITIONS:
        raise RuntimeError('frozen engineering scope changed')
    return cfg, cases, old


def all_sources():
    from experiment.phase18.core import score_replay
    return list(dict.fromkeys(d.sources() + [Path(__file__), CONFIG, CASES, ADDENDUM,
        Path(score_replay.__file__), ROOT / 'experiment/phase18/tests/test_s18_score_replay.py',
        ROOT / 'experiment/phase18/run_stage18_score_numerics_audit.sh']))


def read_trace(domain, split, user):
    path = d.BASE / 'run-0001' / domain / (split + '_traces.jsonl')
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row['user'] == user:
                return row
    raise RuntimeError('frozen trace missing')


def native_capture(ctx, model, row, old_trace, paths):
    from experiment.phase18.core import score_replay as replay
    torch, depth = ctx.torch, len(paths[0])
    logits, logp, encoders = [], [], []
    calls = [0]

    def encoder_hook(module, args, output):
        encoders.append(output[0].detach().clone())

    def model_hook(module, args, output):
        index = calls[0]
        calls[0] += 1
        if index < depth:
            a, b = replay.select_frontier(output.logits[:, -1, :], old_trace['steps'][index], paths, index)
            logits.append(a)
            logp.append(b)

    hooks = [model.encoder.register_forward_hook(encoder_hook), model.register_forward_hook(model_hook)]
    try:
        result, trace = d.generate(ctx, model, row)
    finally:
        for hook in hooks:
            hook.remove()
    if len(encoders) != 1:
        raise RuntimeError('native generation did not encode exactly once')
    if trace.steps[:depth] != old_trace['steps'][:depth] or trace.final != old_trace['final']:
        raise RuntimeError('current native replay differs from frozen native trace')
    return encoders[0], replay.pack(torch.stack(logits, 1), torch.stack(logp, 1), paths, ctx.device), result


def one_case(ctx, model, split, case, out, ordinal):
    from experiment.phase18.core import score_replay as replay
    torch = ctx.torch
    row = (ctx.train if split == 'training' else ctx.evaluation)[case['user']]
    if row['target'] != case['target']:
        raise RuntimeError('selected target mismatch')
    event, old_trace = case['event'], read_trace(ctx.domain, split, case['user'])
    paths = [event['target_prefix'], event['boundary_prefix']]
    if len(paths[0]) != len(paths[1]) or any(1 in p for p in paths):
        raise RuntimeError('audit requires equal-depth, non-EOS prefixes')
    for p in paths:
        for depth, token in enumerate(p):
            if token not in ctx.trie.get([0] + p[:depth]):
                raise RuntimeError('illegal fixed prefix')
    batch = ctx.batch(row)
    before = d.state_hash(model)
    torch.cuda.reset_peak_memory_stats()
    packs, hidden_differences = {}, {}
    with torch.no_grad(), replay.precision(True):
        native_hidden, packs['native_on'], _ = native_capture(ctx, model, row, old_trace, paths)
        native_scores = packs['native_on']['summary']['sequential_float32_scores']
        if native_scores != [event['target_score'], event['boundary_score']]:
            raise RuntimeError('selected native token probabilities do not reconstruct cumulative scores')
    for tf32, name in [(True, 'on'), (False, 'off')]:
        with torch.no_grad(), replay.precision(tf32):
            hidden = replay.encode(model, batch, ctx.device)
            if tf32 and not torch.equal(hidden, native_hidden):
                raise RuntimeError('single-encoder replay differs from captured native encoder')
            duplicate = replay.encode(model, batch, ctx.device, copies=2)
            hidden_differences[name] = {
                'single_vs_duplicate_max_absolute': float((hidden.repeat(2, 1, 1) - duplicate).abs().max()),
                'duplicate_rows_exact': torch.equal(duplicate[0], duplicate[1]),
                'single_vs_native_on_max_absolute': float((hidden - native_hidden).abs().max())}
            packs[name + '_frontier_cached'] = replay.fixed_frontier(model, hidden, batch['item_text_masks'],
                                   old_trace['steps'], paths, ctx.device, True)
            packs[name + '_frontier_full'] = replay.fixed_frontier(model, hidden, batch['item_text_masks'],
                                   old_trace['steps'], paths, ctx.device, False)
            packs[name + '_pair_full_shared'] = replay.path_full(model, hidden, batch['item_text_masks'], paths, ctx.device)
            packs[name + '_pair_full_duplicate_encoder'] = replay.path_full(model, duplicate, batch['item_text_masks'], paths, ctx.device)
            packs[name + '_pair_cached'] = replay.path_cached(model, hidden, batch['item_text_masks'], paths, ctx.device)
            packs[name + '_single_full'] = replay.path_single(model, hidden, batch['item_text_masks'], paths, ctx.device)
            # The original differentiable scorer is rerun without backward, to verify the decomposition.
            original = d.prefix_scores(ctx, model, row, paths).cpu().tolist()
            if original != packs[name + '_pair_full_duplicate_encoder']['summary']['sum_float32_scores']:
                raise RuntimeError('decomposed scorer differs from original scorer value')
            del duplicate, hidden
    comparisons = {}
    for name in ['on', 'off']:
        for a, b, label in [
            ('pair_full_duplicate_encoder', 'frontier_cached', 'total_scorer_gap'),
            ('pair_full_duplicate_encoder', 'pair_full_shared', 'encoder_batch_effect'),
            ('frontier_full', 'frontier_cached', 'cache_effect_at_width50'),
            ('pair_full_shared', 'pair_cached', 'cache_effect_at_width2'),
            ('pair_cached', 'frontier_cached', 'batch_effect_with_cache'),
            ('single_full', 'pair_full_shared', 'batch1_vs_batch2_full')]:
            comparisons[name + '_' + label] = replay.compare(packs[name + '_' + a], packs[name + '_' + b])
    comparisons['native_vs_frozen_frontier_replay'] = replay.compare(packs['native_on'], packs['on_frontier_cached'])
    if not comparisons['native_vs_frozen_frontier_replay']['log_probabilities_exact']:
        raise RuntimeError('fixed-frontier cache replay differs from captured native logits/probabilities')
    for condition in CONDITIONS:
        comparisons['precision_effect_' + condition] = replay.compare(packs['off_' + condition], packs['on_' + condition])
    old_tf = [case['score_audit']['tf_target'], case['score_audit']['tf_boundary']]
    reproduced_tf = packs['on_pair_full_duplicate_encoder']['summary']['sum_float32_scores']
    after = d.state_hash(model)
    if before != after:
        raise RuntimeError('numerical audit changed weights')
    result = {'ordinal': ordinal, 'domain': ctx.domain, 'split': split, 'user': case['user'], 'reasons': case['reasons'],
              'prefixes': paths, 'historical_native_margin': event['native_margin'],
              'historical_tf_scores': old_tf, 'historical_tf_reproduced_exact': old_tf == reproduced_tf,
              'historical_tf_replay_differences': [a - b for a, b in zip(reproduced_tf, old_tf)],
              'native_trace_identity': True, 'shared_encoder_identity': True, 'scorer_decomposition_identity': True,
              'hidden_comparisons': hidden_differences, 'conditions': {k: p['summary'] for k, p in packs.items()},
              'comparisons': comparisons, 'parameters_sha256_before': before, 'parameters_sha256_after': after,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20, **SEALED}
    if not all(math.isfinite(c['max_absolute_score_difference']) for c in comparisons.values()):
        raise FloatingPointError('nonfinite comparison')
    s2.append(out / 'cases.jsonl', result)
    s2.progress(out, 'CASE_COMPLETED', cases_done=ordinal, cases_total=9, domain=ctx.domain,
                split=split, user=case['user'], on_gap=comparisons['on_total_scorer_gap']['max_absolute_score_difference'],
                off_gap=comparisons['off_total_scorer_gap']['max_absolute_score_difference'], **SEALED)
    return result


def summarize(rows):
    on = [r['comparisons']['on_total_scorer_gap']['max_absolute_score_difference'] for r in rows]
    off = [r['comparisons']['off_total_scorer_gap']['max_absolute_score_difference'] for r in rows]
    return {'cases': len(rows), 'max_score_gap_tf32_on': max(on), 'max_score_gap_tf32_off': max(off),
            'cases_with_smaller_gap_when_tf32_off': sum(b < a for a, b in zip(on, off)),
            'ratio_of_maxima': max(on) / max(off) if max(off) else None,
            'historical_native_trace_exact_cases': sum(r['native_trace_identity'] for r in rows),
            'historical_tf_exact_cases': sum(r['historical_tf_reproduced_exact'] for r in rows),
            'max_encoder_batch_effect_on': max(r['comparisons']['on_encoder_batch_effect']['max_absolute_score_difference'] for r in rows),
            'max_encoder_batch_effect_off': max(r['comparisons']['off_encoder_batch_effect']['max_absolute_score_difference'] for r in rows),
            'max_cache_effect_width50_on': max(r['comparisons']['on_cache_effect_at_width50']['max_absolute_score_difference'] for r in rows),
            'max_cache_effect_width50_off': max(r['comparisons']['off_cache_effect_at_width50']['max_absolute_score_difference'] for r in rows),
            'peak_reserved_mib': max(r['peak_reserved_mib'] for r in rows)}


def report(result, out):
    lines = ['# 第十八阶段：固定案例评分数值复核', '', f"状态：`{result['decision']}`。",
             f"完成时间：{result.get('completed_at', utc_now())}。", '',
             '本轮最多 9 个工程案例；固定权重、原生 frontier 和输入，无训练步或新确认样本。']
    if 'aggregate' in result:
        a = result['aggregate']
        lines += ['', f"完成 {a['cases']} 个案例；旧 native trace 精确复现 {a['historical_native_trace_exact_cases']} 个，",
                  f"旧 teacher-forcing 值精确复现 {a['historical_tf_exact_cases']} 个。",
                  f"开启 TF32 时，两种评分最大累计差为 {a['max_score_gap_tf32_on']:.10g}；",
                  f"关闭 TF32 后为 {a['max_score_gap_tf32_off']:.10g}。",
                  f"关闭后差异缩小的案例数：{a['cases_with_smaller_gap_when_tf32_off']}/{a['cases']}。", '',
                  '| 案例 | 原设置累计分差 | 关闭 TF32 累计分差 |', '|---|---:|---:|']
        for row in result['case_overview']:
            lines.append(f"| {row['domain']}/{row['split']}/{row['user']} | {row['on_gap']:.9g} | {row['off_gap']:.9g} |")
        lines += ['', '这些配对结果用于定位数值计算路径的影响，不代表 CF 的科学增量或梯度稳定性已被验证。',
                  '原梯度状态 MEASUREMENT_UNRESOLVED 与 S18-2 FAILED_SCIENTIFIC_GATE 保持不变。',
                  '逐条件 logits/log-softmax/累计分数差、encoder 差、缓存差与权重 SHA 均保存在 cases.jsonl。']
    lines += ['', f"[完整产物](../../{out.relative_to(ROOT)})。", '']
    atomic_text(REPORT, '\n'.join(lines))


def worker(args):
    cfg, cases, old = verify()
    out = BASE / args.run_name
    for source in load_json(out / 'run_manifest.json')['sources']:
        s2.checked(source)
    from experiment.phase18.protocol import s18_s1_runtime as rt
    rt.torch.set_num_threads(4)
    rt.set_seed(2023)
    if rt.torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one visible GPU required')
    runtime = {'torch': rt.torch.__version__, 'cuda': rt.torch.version.cuda,
               'device': rt.torch.cuda.get_device_name(), 'matmul_tf32_default': rt.torch.backends.cuda.matmul.allow_tf32,
               'cudnn_tf32_default': rt.torch.backends.cudnn.allow_tf32, 'threads': rt.torch.get_num_threads(),
               'grad_enabled_during_scoring': False}
    atomic_json(out / 'runtime.json', runtime)
    started, rows, ctx, model = time.monotonic(), [], None, None
    for stratum in cases['strata']:
        if ctx is None or ctx.domain != stratum['domain']:
            if model is not None:
                del model, ctx
                gc.collect()
                rt.torch.cuda.empty_cache()
            ctx = s2.Context(stratum['domain'], old)
            model = ctx.parent().eval()
        for case in stratum['selected_engineering_cases']:
            rows.append(one_case(ctx, model, stratum['split'], case, out, len(rows) + 1))
            if len(rows) == 1:
                smoke = {'decision': 'ENGINEERING_PASS', 'source_manifest_sha256': sha256(out / 'run_manifest.json'),
                         'peak_reserved_mib': rows[0]['peak_reserved_mib'], 'case': case['user'],
                         'next_action': 'continue_remaining_eight_fixed_cases', **SEALED}
                atomic_json(out / 'smoke_summary.json', smoke)
                # Recheck external free memory; the smoke allocation is already resident.
                d.resources(args.gpu, 1024)
                s2.progress(out, 'SINGLE_CASE_SMOKE_PASS', **smoke)
    verify()
    result = {'decision': 'NUMERICAL_COMPARISON_COMPLETED', 'aggregate': summarize(rows),
              'case_overview': [{'domain': r['domain'], 'split': r['split'], 'user': r['user'],
                  'on_gap': r['comparisons']['on_total_scorer_gap']['max_absolute_score_difference'],
                  'off_gap': r['comparisons']['off_total_scorer_gap']['max_absolute_score_difference']} for r in rows],
              'runtime': runtime, 'elapsed_seconds': time.monotonic() - started, 'completed_at': utc_now(), **SEALED}
    atomic_json(out / 'summary.json', result)
    report(result, out)
    s2.progress(out, 'COMPLETED', decision=result['decision'], **SEALED)
    return 0


def supervisor(args):
    out = BASE / args.run_name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    started = time.monotonic()
    with (out / 'worker.log').open('a') as log:
        p = subprocess.Popen([s2.PYTHON, '-m', MODULE, 'worker', '--run-name', args.run_name, '--gpu', str(args.gpu)],
                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            elapsed, rc = time.monotonic() - started, p.poll()
            timed_out = elapsed > 1800
            if timed_out and rc is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    rc = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    rc = p.wait()
            complete = rc == 0 and (out / 'summary.json').exists()
            status = {'state': 'RUNNING' if rc is None else 'COMPLETED' if complete else 'INFRASTRUCTURE_FAILED',
                      'pid': p.pid, 'supervisor_pid': os.getpid(), 'run_name': args.run_name,
                      'physical_gpu': args.gpu, 'elapsed_seconds': elapsed, 'heartbeat_at': utc_now(),
                      'exit_code': rc, 'timed_out': timed_out, 'hard_timeout_seconds': 1800,
                      'progress': load_json(out / 'progress.json') if (out / 'progress.json').exists() else {}, **SEALED}
            atomic_json(STATUS, status)
            atomic_json(out / 'status.json', status)
            if rc is not None:
                s2.append(LEDGER, {'event': 'FINISHED', **status})
                if not complete:
                    failure = dict(status, decision='INFRASTRUCTURE_FAILED')
                    atomic_json(out / 'failure.json', failure)
                    report(failure, out)
                return 0 if complete else 1
            time.sleep(10)


def launch(args):
    from experiment.phase17.core.run_manager import launch_background_tmux, wait_for_tmux_startup
    verify()
    if LEDGER.exists():
        raise RuntimeError('attempt already exists; automatic retry prohibited')
    resource = d.resources(args.gpu, 12000)
    out = BASE / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    snap = out / 'source_snapshot'
    snap.mkdir()
    refs = []
    for index, path in enumerate(all_sources()):
        refs.append(d.ref(path))
        (snap / f'{index:03d}_{path.name}').write_bytes(path.read_bytes())
    manifest = {'created_at': utc_now(), 'sources': refs, 'inputs': load_json(CONFIG)['inputs'],
                'resource': resource, 'config_sha256': sha256(CONFIG), **SEALED}
    atomic_json(out / 'run_manifest.json', manifest)
    session = 's18_score_numerics_' + args.run_name.replace('-', '_')
    s2.append(LEDGER, {'event': 'STARTED', 'at': utc_now(), 'manifest': d.ref(out / 'run_manifest.json'),
                       'tmux_session': session, 'gpu': args.gpu, **SEALED})
    launch_background_tmux(experiment_id='s18_score_numerics_audit', cwd=ROOT, tmux_session=session,
        startup_log_path=out / 'master.log', argv=[s2.PYTHON, '-m', MODULE, 'supervisor', '--run-name', args.run_name,
                                                '--gpu', str(args.gpu)])
    if not wait_for_tmux_startup(session):
        raise RuntimeError('tmux startup failed; inspect master.log')
    print(json.dumps({'state': 'STARTED', 'session': session, 'directory': str(out),
                      'gpu': args.gpu, 'free_mib': resource['free_mib'], 'hard_timeout_seconds': 1800}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['launch', 'supervisor', 'worker'])
    parser.add_argument('--run-name', default='run-0001')
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9_-]+', args.run_name):
        raise ValueError('invalid run name')
    return {'launch': launch, 'supervisor': supervisor, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
