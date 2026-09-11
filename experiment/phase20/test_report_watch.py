"""Lifecycle checks use isolated fixtures, never real experiment reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment.phase20 import report_watch


class ReportLifecycleTest(unittest.TestCase):
    def test_intermediate_domain_and_final_results_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, reports = root/'artifacts', root/'report'
            def put(relative, value):
                path = artifacts/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            put('status.json', {'state':'RUNNING','dataset':'Toys'})
            put('sprec_gram_toys_v1/status.json', {'state':'RUNNING'})
            put('round1_sft_full_v1/status.json', {'state':'RUNNING'})
            put('followup_queue/status.json', {'state':'WAITING'})
            with patch.object(report_watch, 'P20', artifacts), patch.object(report_watch, 'REPORT', reports), patch.object(report_watch, 'ROOT', root):
                self.assertFalse(report_watch.snapshot())
                self.assertIn('尚无全量终态结果', (reports/'Stage20_round1_SFT全量验证报告.md').read_text())
                metric = {'users':19412, **{f'{name}@{k}':.1 for name in ['hit','ndcg'] for k in [5,10,20,50]}}
                full = {k:dict(metric) for k in ['raw','pcrf','original_gram','original_gram_pcrf']}
                full['raw']['ndcg@10'] = .101
                full['pcrf']['ndcg@10'] = .102
                sprec = {'best':{'raw':{'tag':'round1_dpo'},'pcrf':{'tag':'round1_dpo'}},
                    'full_validation':{'round1_dpo':full}, 'raw_ndcg_delta_vs_gram':.001,
                    'combined_ndcg_delta_vs_original_pcrf':.002,'decision':'CROSS_DOMAIN_SCREEN'}
                put('sprec_gram_toys_v1/summary.json', sprec)
                put('sprec_gram_toys_v1/status.json', {'state':'COMPLETED'})
                put('status.json', {'state':'RUNNING','dataset':'Beauty'})
                put('sprec_gram_beauty_v1/status.json', {'state':'RUNNING'})
                put('round1_sft_full_v1/status.json', {'state':'COMPLETED'})
                put('round1_sft_full_v1/summary.json', {'full_validation':full,'raw_ndcg_delta_vs_gram':.001,'combined_ndcg_delta_vs_original_pcrf':.002})
                put('followup_queue/status.json', {'state':'FAILED','error':'fixture engineering failure'})
                self.assertFalse(report_watch.snapshot())
                self.assertIn('0.101000', (reports/'Stage20_round1_SFT全量验证报告.md').read_text())
                self.assertIn('Beauty', (reports/'Stage20_SPRec运行与最终结果报告.md').read_text())
                self.assertIn('fixture engineering failure', (reports/'Stage20_GACR_v6_PCRF组合评估报告.md').read_text())
                values = {name:dict(metric) for name in ['original_gram','original_pcrf','c1','c1_pcrf','gacr_v6','gacr_v6_pcrf']}
                values['gacr_v6_pcrf']['ndcg@10'] = .104
                paired = {name:{'delta':.004,'paired_bootstrap_95ci':[-.001,.008]} for name in ['ndcg@10','hit@10']}
                put('gacr_v6_pcrf_toys_v1/status.json', {'state':'COMPLETED'})
                put('gacr_v6_pcrf_toys_v1/summary.json', {'metrics':values,
                    'paired_comparisons':{'original_pcrf':paired}, 'combined_ndcg_delta_vs_original_pcrf':.004,
                    'decision':'POSITIVE_EXPLORATORY_POINT_ESTIMATE',
                    'groups':{'all_validation':{'metrics':values,'combined_vs_original_pcrf':paired},
                              'outside_c1_training_users':{'metrics':values,'combined_vs_original_pcrf':paired}}})
                put('followup_queue/status.json', {'state':'COMPLETED'})
                put('status.json', {'state':'COMPLETED','completed_domains':['Toys','Beauty']})
                put('sprec_gram_beauty_v1/status.json', {'state':'COMPLETED'})
                put('sprec_gram_beauty_v1/summary.json', sprec)
                self.assertTrue(report_watch.snapshot())
                self.assertIn('0.104000', (reports/'Stage20_GACR_v6_PCRF组合评估报告.md').read_text())
                self.assertIn('outside_c1_training_users', (reports/'Stage20_GACR_v6_PCRF组合评估报告.md').read_text())
                self.assertTrue(json.loads((artifacts/'report_watch/status.json').read_text())['all_monitored_jobs_finished'])
                self.assertEqual(len(list(reports.glob('*.md'))), 4)


if __name__ == '__main__':
    unittest.main()
