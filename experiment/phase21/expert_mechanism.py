"""Isolated mechanism controls reusing the immutable contribution runner."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from experiment.phase21 import text_contribution as runner
from experiment.phase21.expert_mechanism_core import (
    ARMS, CHANNEL, ScoreOverride, flat_neighbor_scores, shuffle_older_history,
    transform_scores, fuse)
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = runner.ROOT
OUT = ROOT / 'artifacts/phase21/expert_mechanism_20260923_v1'
TEXT = ROOT / 'artifacts/phase21/text_contribution_20260923_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.3D_SASRec与ItemCF机制补齐_20260923.md'
REPORT = ROOT / 'report/第二十一阶段/SASRec与ItemCF机制补齐_20260923.md'
GPUS = dict(sasrec_misassigned='GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc',
            itemcf_misassigned='GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059',
            sasrec_order='GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2',
            itemcf_flat='GPU-07374397-b213-09c1-2988-718acb6ed231')
OriginalWorker = runner.Worker
original_free_memory = runner.free_memory


class Worker(OriginalWorker):
    def mechanism_signal(self):
        import torch
        from lime_rec.models import SASRec
        from lime_rec.experts import ItemCFExpert
        from lime_rec.evaluation import normalize_score_rows
        from experiment.phase20.lime_comparison_core import left_pad

        self.stage = 'mechanism_scores'; self.status()
        n, v = len(self.samples), self.raw.shape[-1]
        channel = CHANNEL[self.arm]
        histories = [s['full_history'] for s in self.samples]
        replacement = np.lib.format.open_memmap(self.folder / 'mechanism_scores.npy',
            mode='w+', dtype='float32', shape=(n, v))
        changed_inputs = 0
        mass_error = 0.
        if self.arm == 'sasrec_order':
            checkpoint = torch.load(ROOT / self.cfg['sasrec'], map_location='cpu')
            assert checkpoint['train_complete'] and checkpoint['epoch'] == 200 and checkpoint['training_only']
            sas = SASRec(v, 64, 50, 2, 2, 0., architecture='legacy').to(self.device)
            sas.load_state_dict(checkpoint['state_dict'], strict=True)
            sas.eval().requires_grad_(False); del checkpoint
            modified = [shuffle_older_history(h, s['user_id'], self.domain)
                        for h, s in zip(histories, self.samples)]
            for h, m in zip(histories, modified):
                assert sorted(h[-50:]) == sorted(m) and h[-1] == m[-1]
                changed_inputs += int(h[-50:] != m)
            del h, m
        else:
            assert self.arm == 'itemcf_flat'
            source = self.catalog['items'][1:]
            cf = ItemCFExpert(str(ROOT / self.cfg['itemcf']), source)
            original_weights = cf.neighbor_scores
            flat_weights = {k: flat_neighbor_scores(w) for k, w in original_weights.items()}
            for k, w in original_weights.items():
                new = flat_weights[k]
                np.testing.assert_allclose(new.sum(dtype=np.float64), w.sum(dtype=np.float64), rtol=2e-7, atol=1e-8)
                mass_error = max(mass_error, float(abs(new.sum(dtype=np.float64) - w.sum(dtype=np.float64))))
            changed_inputs = sum(any(i in original_weights and not np.array_equal(original_weights[i], flat_weights[i])
                                     for i in [source[x-1] for x in h[-cf.history_window:]]) for h in histories)

        parity_error = 0.; changed_rows = 0; absolute_delta = 0.
        with torch.no_grad():
            for start in range(0, n, 64):
                stop = min(start + 64, n)
                if self.arm == 'sasrec_order':
                    old = sas.predict(torch.from_numpy(left_pad(histories[start:stop])).to(self.device))[:, 1:].cpu().numpy()
                    new = sas.predict(torch.from_numpy(left_pad(modified[start:stop])).to(self.device))[:, 1:].cpu().numpy()
                else:
                    hs = [[source[x-1] for x in h] for h in histories[start:stop]]
                    cf.neighbor_scores = original_weights
                    old = np.stack([cf.score(self.samples[start+i]['user_id'], h) for i, h in enumerate(hs)])
                    cf.neighbor_scores = flat_weights
                    new = np.stack([cf.score(self.samples[start+i]['user_id'], h) for i, h in enumerate(hs)])
                    np.testing.assert_array_equal(old > 0, new > 0)
                old = normalize_score_rows(old); new = normalize_score_rows(new)
                cached = np.asarray(self.raw[self.source_indices[start:stop], channel])
                np.testing.assert_allclose(old, cached, atol=5e-6, rtol=1e-5)
                parity_error = max(parity_error, float(np.abs(old - cached).max()))
                assert np.isfinite(new).all()
                replacement[start:stop] = new
                changed_rows += int(np.any(np.abs(new - old) > 1e-6, axis=1).sum())
                absolute_delta += float(np.abs(new - old).sum(dtype=np.float64))
                self.status(score_rows=stop, score_total=n)
        replacement.flush()
        runner.write(self.folder / 'signal_audit.json', dict(state='PASSED', arm=self.arm,
            rows=n, input_changed_rows=changed_inputs, score_changed_rows=changed_rows,
            original_score_max_abs_error=parity_error, mean_abs_score_change=absolute_delta/(n*v),
            maximum_cf_source_mass_error=mass_error, other_channels_unchanged=True,
            targets_used_for_signal=False, upstream_optimizer_steps=0,
            interpretation=('frozen_model_order_sensitivity_with_latest_item_fixed' if self.arm == 'sasrec_order'
                            else 'within_source_strength_with_same_neighbors_and_source_mass')))
        self.raw = ScoreOverride(self.raw, self.source_indices, replacement, channel)
        if self.arm == 'sasrec_order': del sas
        else: del cf, original_weights, flat_weights
        gc.collect(); torch.cuda.empty_cache()

    def fit_signals(self):
        if self.arm in ('sasrec_order', 'itemcf_flat'):
            self.mechanism_signal()
        else:
            runner.write(self.folder / 'signal_audit.json', dict(state='PASSED', arm=self.arm,
                control='same_training_frequency_bucket_permutation_as_text',
                changed_item_ids=int((self.donors != np.arange(len(self.donors))).sum()),
                items=len(self.donors), score_multiset_preserved=True, targets_used_for_signal=False))
        super().fit_signals()


def free_memory():
    free = original_free_memory()
    for new_arm, old_arm in [('sasrec_misassigned', 'real'), ('itemcf_misassigned', 'without_text')]:
        path = TEXT / old_arm / 'Beauty/status.json'
        if not path.exists():
            free[GPUS[new_arm]] = 0; continue
        status = runner.read(path)
        # Require that the previous owned worker completed and exited. Other
        # users' processes are left untouched; ordinary free-memory checks apply.
        if status['state'] != 'COMPLETED' or Path(f"/proc/{status.get('pid', 0)}").exists():
            free[GPUS[new_arm]] = 0
    return free


def summarize():
    domains = {}; checks = {a: {} for a in ARMS}
    lines = ['# SASRec与ItemCF机制补齐', '', '状态：COMPLETED。四组两域真实生成及独立排名核验完成。', '',
        'Origin Skill：academic-research-suite。共享完整组合与既有删减对照；这里的四组是机制检查，不是四个新方法。', '',
        '| 域 | 组 | N10 | H10 | H50 | 完整组减对照 ΔN10 | 描述性95%区间 |', '|---|---|---:|---:|---:|---:|---|']
    for domain in runner.DOMAINS:
        previous = runner.base.arrays(runner.SOURCE / domain / 'ranks.npz')
        reference = dict(users=previous['development_users'], GRAM=previous['dev_GRAM'],
                         late=previous['dev_fixed_late'], guided=previous['dev_fixed_decode'])
        replay = TEXT / 'real' / domain / 'ranks.npz'
        if replay.exists():
            actual = runner.base.arrays(replay)
            for k in reference: np.testing.assert_array_equal(reference[k], actual[k])
        m = metrics(reference['guided'])
        lines.append(f'| {domain} | 完整组合 | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} | — | — |')
        results = {}
        for arm in ARMS:
            folder = OUT / arm / domain
            result = runner.read(folder / 'summary.json'); rr = runner.base.arrays(folder / 'ranks.npz')
            np.testing.assert_array_equal(rr['users'], reference['users'])
            np.testing.assert_array_equal(rr['GRAM'], reference['GRAM'])
            cc = {scope: contrast(reference[scope], rr[scope], repeats=2000, seed=20260923)
                  for scope in ('guided', 'late')}
            a = result['metrics']['guided']; c = cc['guided']; lo, hi = c['descriptive_95pct_N10']
            checks[arm][domain] = bool(m['N10'] > a['N10'] and m['H10'] >= a['H10'])
            results[arm] = dict(result=result, original_minus_control=cc,
                                signal_audit=runner.read(folder / 'signal_audit.json'))
            lines.append(f'| {domain} | {arm} | {a["N10"]:.6f} | {a["H10"]:.6f} | {a["H50"]:.6f} | {c["delta_N10"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
        domains[domain] = dict(reference_metrics=m, replay_checked=replay.exists(), arms=results)
    lines += ['', '解释边界：', '',
        '- 两个misassigned组与文本错配同层次：考察候选评分的正确对应，不证明某一内部机制。',
        '- sasrec_order固定最近一件，扰动此前位置。冻结模型的输入敏感性包含分布变化，不能等同重新训练无序模型，更不能单独归因于转移或近期性。',
        '- itemcf_flat保留邻居与每源总质量，只去掉同一源的邻居强弱差异；不是移除全部协同信息。',
        '- 差值为正表示原方案方向更好；小差值或跨域不同不能确定必要性。输入破坏后变差不是新方法收益。',
        '- 单上游种子、反复使用开发池；没有test读取，没有因结果重试或调参。', '',
        '[冻结协议](../../plan/第二十一阶段/21.3D_SASRec与ItemCF机制补齐_20260923.md)；[删减贡献报告](SASRec与ItemCF保留价值对照_20260923.md)。']
    REPORT.write_text('\n'.join(lines) + '\n')
    summary = dict(state='COMPLETED', domains=domains, directional_support=checks,
        dual_domain_support=all(all(d.values()) for d in checks.values()),
        aggregate_boolean_not_a_method_gate=True, test_read=False)
    runner.write(OUT / 'summary.json', summary)
    return summary


def bind_runtime():
    runner.OUT = OUT; runner.PLAN = PLAN; runner.REPORT = REPORT
    runner.MODULE = 'experiment.phase21.expert_mechanism'; runner.ARMS = ARMS; runner.GPUS = GPUS
    runner.Worker = Worker; runner.transform_scores = transform_scores; runner.fuse = fuse
    runner.free_memory = free_memory; runner.summarize = summarize


def prepare():
    frozen = runner.read(TEXT / 'manifest.json')['sha256']
    for name in ('text_contribution.py', 'text_contribution_core.py'):
        rel = 'experiment/phase21/' + name
        assert runner.sha(ROOT / rel) == frozen[rel]
    runner.prepare()
    receipt = runner.read(OUT / 'manifest.json')
    upstream = runner.read(runner.SOURCE / 'manifest.json')['sha256']
    for domain in runner.DOMAINS:
        cfg = runner.read(runner.SOURCE / domain / 'data_config.json')
        for name in ('sasrec', 'itemcf'):
            rel = cfg[name]; digest = runner.sha(ROOT / rel)
            assert digest == upstream[rel]
            receipt['sha256'][rel] = digest
    for path in [Path(__file__), ROOT / 'experiment/phase21/expert_mechanism_core.py',
                 ROOT / 'experiment/phase21/test_expert_mechanism.py',
                 ROOT / 'experiment/phase21/run_expert_mechanism.sh',
                 ROOT / 'experiment/phase20/lime_comparison_core.py']:
        rel = path.relative_to(ROOT); receipt['sha256'][str(rel)] = runner.sha(path)
        dest = OUT / 'sources' / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    receipt['process_local_adapter'] = True
    runner.write(OUT / 'manifest.json', receipt)
    REPORT.write_text('# SASRec与ItemCF机制补齐\n\n状态：PREPARED。四个机制对照已冻结，尚无新效果结论。\n')
    print(json.dumps(dict(state='PREPARED', frozen_files=len(receipt['sha256']))), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'supervise', 'worker'])
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--domain', choices=runner.DOMAINS)
    args = parser.parse_args(); bind_runtime()
    if args.action == 'prepare': prepare()
    elif args.action == 'worker':
        assert args.arm and args.domain
        runner.worker(args.arm, args.domain)
    else:
        runner.supervise()
        if runner.read(OUT / 'status.json')['state'] == 'FAILED' and REPORT.exists():
            REPORT.write_text(REPORT.read_text().replace('文本专家贡献三组并行对照', 'SASRec与ItemCF机制补齐'))


if __name__ == '__main__': main()
