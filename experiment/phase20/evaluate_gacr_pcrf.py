"""Frozen C1/GACR-v6 -> PCRF cascade on complete Toys validation.

Uses the original GACR candidate/feature builder. Only captures generation
scores additionally; never trains and never constructs test examples.
"""
import argparse
import csv
import gc
import json
import os
import signal
import time
import traceback

import numpy as np
import torch

from experiment.phase20.run_preference import ROOT, Run, Stopped, metrics, paired_delta, seed_everything
from experiment.phase20.evaluate_saved import gpu_setup, load_parent_reference
from experiment.phase18.core.diff_data import sha256, write_json
from experiment.phase4.gcdh_p0 import CatalogDualHead
from experiment.phase4.gacr_s0 import BoundedResidualRanker, build_candidate_record, stable_ranking
from experiment.phase4.gacr_s0 import gt
from experiment.phase20.gacr_features import vector_features


class CombinationRun(Run):
    def setup(self):
        cfg = self.config
        for path, expected in cfg['frozen_inputs'].items():
            if sha256(ROOT / path) != expected:
                raise AssertionError(f'Frozen input SHA mismatch: {path}')
        self.model = self.new_model()
        # C1's catalog-head rows use file insertion order, not sorted item IDs.
        lexical = {}
        for line in (ROOT / cfg['c1_catalog_file']).read_text().splitlines():
            item, value = line.split(' ', 1)
            if item in lexical:
                raise AssertionError('Duplicate C1 catalog entry')
            lexical[item] = value
        self.numeric = {item: i for i, item in enumerate(self.catalog['items']) if i}
        names = list(lexical)
        assert set(names) == set(self.numeric)
        for raw, value in lexical.items():
            assert value == self.catalog['lexical_ids'][self.numeric[raw]]
        state = torch.load(ROOT / cfg['c1_checkpoint'], map_location='cpu')
        self.dual = CatalogDualHead(self.model, self.tokenizer, names, lexical, state['log_counts']).to(self.device)
        self.dual.load_state_dict(state, strict=True)
        del state
        self.dual.eval().requires_grad_(False)
        self.ranker = BoundedResidualRanker(6, 16, cfg['residual_bound']).to(self.device)
        self.ranker.load_state_dict(torch.load(ROOT / cfg['residual_checkpoint'], map_location='cpu'), strict=True)
        self.ranker.eval().requires_grad_(False)
        paths = [self.decoder.paths[self.numeric[name]-1] for name in names]
        # This metadata is derived only from already-prepared training events.
        pop_order = sorted(names, key=lambda name: (-int(self.frequencies[self.numeric[name]]), name))
        heads = set(pop_order[:max(1, int(np.ceil(len(names)*.2)))])
        self.prepared = {'model': self.dual, 'collator': self.collator, 'catalog': names,
            'heads': heads, 'item2lexid': lexical, 'encoded_candidates': paths,
            'sequence_to_item': {tuple(path): name for path, name in zip(paths, names)}}
        self.snapshot()
        load_parent_reference(self, ROOT / cfg['shared_parent_output'])
        write_json(self.output / 'frozen_model_manifest.json', {
            'inputs': cfg['frozen_inputs'], 'catalog_row_order': 'lexical file insertion order',
            'catalog_row_mapping_verified': True, 'new_optimizer_steps': 0,
            'combination': 'Original GACR top50, then frozen PCRF with GACR scores as its base signal',
            'parent_model': 'C1 continued backbone; independently compared to untouched GRAM'})

    def sample(self, index):
        row = self.validation.records[index]
        target = self.catalog['items'][row['target']]
        return {**self.validation[index], 'positive_item': target,
            'sample_key': f"{row['user_id']}:validation:{target}",
            'history_items': [self.catalog['items'][i] for i in row['history'][-20:]]}

    @torch.no_grad()
    def candidate(self, index, capture=True):
        if capture and self.config.get('vectorized_features', False):
            return self.fast_candidate(index)
        captured = {}
        original = self.model.generate
        def generate(*args, **kwargs):
            kwargs['output_scores'] = True
            result = original(*args, **kwargs)
            captured['scores'] = result.sequences_scores.detach().cpu().tolist()
            captured['sequences'] = result.sequences.detach().cpu().tolist()
            return result
        if capture:
            self.model.generate = generate
        try:
            record = build_candidate_record(self.sample(index), self.prepared, self.config, self.device)
        finally:
            if capture:
                del self.model.generate  # restore the unmodified class method
        if not capture:
            return record
        assert len(captured['scores']) == 50 and np.isfinite(captured['scores']).all()
        generated = [self.numeric[self.prepared['sequence_to_item'][tuple(s[:s.index(1)+1])]] for s in captured['sequences']]
        assert generated == [self.numeric[item] for item in record['union'][:50]]
        scores = record['base'] + self.config['residual_scale'] * self.ranker(record['features'])
        scores = scores.detach().cpu()
        order = stable_ranking(scores)[:50]
        assert len(order) == len(set(order)) == 50
        source = self.validation.records[index]
        common = {'user_id': source['user_id'], 'gold_item_id': source['target'], 'split': 'validation'}
        c1 = {**common, 'ranked_item_ids': generated, 'scores': captured['scores']}
        gacr = {**common, 'ranked_item_ids': [self.numeric[record['union'][i]] for i in order],
                'scores': [float(scores[i]) for i in order]}
        return record, c1, gacr

    @torch.no_grad()
    def fast_candidate(self, index):
        sample = self.sample(index)
        # As in the original builder, the decoder label is a fixed dummy item.
        inference = dict(sample, output=self.prepared['item2lexid'][self.prepared['catalog'][0]])
        batch = self.collator([inference])
        ids, mask = batch['item_text_ids'].to(self.device), batch['item_text_masks'].to(self.device)
        if '_gacr_trie' not in self.prepared:
            self.prepared['_gacr_trie'] = gt.Trie(self.prepared['encoded_candidates'])
        output = self.model.generate(input_ids=ids, attention_mask=mask,
            max_length=max(map(len,self.prepared['encoded_candidates'])),
            prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(self.prepared['_gacr_trie']),
            num_beams=50,num_return_sequences=50,return_dict_in_generate=True,output_scores=True,length_penalty=1.0)
        sequences = output.sequences.detach().cpu().tolist()
        generation_scores = output.sequences_scores.detach().cpu().tolist()
        gram = [self.prepared['sequence_to_item'][tuple(s[:s.index(1)+1])] for s in sequences]
        assert len(gram) == len(set(gram)) == 50 and np.isfinite(generation_scores).all()
        del output
        self.model.encoder.n_passages = ids.size(1)
        hidden = self.model.encoder(input_ids=ids.view(1,-1), attention_mask=mask.view(1,-1),return_dict=True)[0]
        pooled = self.dual.pool_coarse(hidden,mask,ids.shape[-1])[0]
        logits = self.dual.catalog_head(pooled)
        seen = [self.dual.item_to_index[x] for x in sample['history_items'] if x in self.dual.item_to_index]
        logits[seen] = -torch.inf
        catalog = [self.dual.catalog[i] for i in torch.topk(logits,k=50).indices.detach().cpu().tolist()]
        union = list(dict.fromkeys(gram+catalog))
        gr, cr = {x:i+1 for i,x in enumerate(gram)}, {x:i+1 for i,x in enumerate(catalog)}
        union_ids = torch.tensor([self.dual.item_to_index[x] for x in union],device=self.device)
        features = vector_features(logits,pooled,self.dual.catalog_head.weight,union_ids,
                                   [gr.get(x,0) for x in union],[cr.get(x,0) for x in union])
        base = torch.tensor([1/gr[x] if x in gr else 0 for x in union],dtype=torch.float32,device=self.device)
        target = sample['positive_item']
        record = {'sample_key':sample['sample_key'],'target_group':'head' if target in self.prepared['heads'] else 'tail',
            'union':union,'target_index':union.index(target) if target in union else None,
            'gram_rank':gr.get(target),'catalog_rank':cr.get(target),'base':base,'features':features}
        scores = (base+self.config['residual_scale']*self.ranker(features)).detach().cpu()
        order = stable_ranking(scores)[:50]
        source = self.validation.records[index]
        common = {'user_id':source['user_id'],'gold_item_id':source['target'],'split':'validation'}
        c1 = {**common,'ranked_item_ids':[self.numeric[x] for x in gram],'scores':generation_scores}
        gacr = {**common,'ranked_item_ids':[self.numeric[union[i]] for i in order],'scores':[float(scores[i]) for i in order]}
        return record,c1,gacr

    def smoke(self):
        self.stage = 'gacr_smoke'
        longest = max(range(len(self.validation)), key=lambda i: len(self.validation.records[i]['history']))
        indices = list(dict.fromkeys(list(range(self.config.get('smoke_parity_users',4)))+[longest]))
        timings = []
        for index in indices:
            self.check_stop()
            started = time.perf_counter()
            before = self.candidate(index, capture=False)
            torch.cuda.synchronize()
            original_seconds = time.perf_counter()-started
            started = time.perf_counter()
            after, c1, gacr = self.candidate(index)
            torch.cuda.synchronize()
            optimized_seconds = time.perf_counter()-started
            timings.append({'index':index,'original_seconds':original_seconds,'optimized_seconds':optimized_seconds})
            assert before['union'] == after['union']
            assert before['gram_rank'] == after['gram_rank']
            assert before['target_index'] == after['target_index']
            torch.testing.assert_close(before['features'], after['features'], atol=1e-6, rtol=0)
            torch.testing.assert_close(before['base'], after['base'], atol=0, rtol=0)
            original_scores = (before['base']+self.config['residual_scale']*self.ranker(before['features'])).detach().cpu()
            original_order = stable_ranking(original_scores)[:50]
            assert [self.numeric[before['union'][i]] for i in original_order] == gacr['ranked_item_ids']
            assert np.isfinite(gacr['scores']).all()
        # Validate the frozen PCRF adapter against saved matched-parent ranks.
        wanted = {self.validation.records[i]['user_id'] for i in range(16)}
        parent_path = ROOT / self.config['shared_parent_output'] / 'parent_full.predictions.jsonl'
        with parent_path.open() as handle:
            parent = [r for r in map(json.loads, handle) if r['user_id'] in wanted]
        ranks = self.pcrf(parent)
        assert all(rank == int(self.reference[r['user_id']]['pcrf_rank']) for r, rank in zip(parent, ranks))
        for path, expected in self.config['frozen_inputs'].items():
            assert sha256(ROOT / path) == expected
        write_json(self.output / 'smoke.json', {'state': 'PASSED', 'original_builder_score_capture_parity_users': len(indices),
            'longest_history': len(self.validation.records[longest]['history']), 'parent_pcrf_exact_users': len(parent),
            'frozen_input_hashes_unchanged': True, 'new_optimizer_steps': 0,
            'vectorized_features': self.config.get('vectorized_features',False), 'parity_timings':timings,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
            'test_read': False})

    def run_full(self):
        smoke_path = ROOT / self.config['smoke_output'] / 'smoke.json'
        if json.loads(smoke_path.read_text())['state'] != 'PASSED':
            raise AssertionError('Engineering smoke has not passed')
        smoke_manifest = json.loads((smoke_path.parent / 'manifest.json').read_text())
        for key in ['frozen_inputs', 'decoding', 'pcrf', 'residual_bound', 'residual_scale', 'generator_top_k', 'catalog_top_k', 'vectorized_features']:
            if smoke_manifest['config'][key] != self.config[key]:
                raise AssertionError(f'Smoke/formal mismatch: {key}')
        self.stage = 'gacr_full_validation'
        c1_rows, gacr_rows, previous = [], [], time.time()
        with (self.output / 'c1.predictions.jsonl').open('x') as fc, (self.output / 'gacr.predictions.jsonl').open('x') as fg, (self.output / 'candidate_features.jsonl').open('x') as ff:
            for index in range(len(self.validation)):
                self.check_stop()
                record, c1, gacr = self.candidate(index)
                c1_rows.append(c1)
                gacr_rows.append(gacr)
                fc.write(json.dumps(c1)+'\n')
                fg.write(json.dumps(gacr)+'\n')
                ff.write(json.dumps({'user_id': c1['user_id'], 'union_ids': [self.numeric[x] for x in record['union']],
                    'base': record['base'].cpu().tolist(), 'features': record['features'].cpu().tolist()})+'\n')
                if time.time()-previous >= 30:
                    for handle in [fc, fg, ff]:
                        handle.flush()
                    self.event('validation_progress', examples=index+1, total=len(self.validation),
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                    previous = time.time()
        self.stage = 'pcrf_and_summary'
        self.event('postprocessing_started', examples=len(c1_rows), total=len(self.validation))
        def raw_rank(row):
            return row['ranked_item_ids'].index(row['gold_item_id'])+1 if row['gold_item_id'] in row['ranked_item_ids'] else 51
        ranks = {
            'original_gram': [int(self.reference[r['user_id']]['baseline_rank']) for r in c1_rows],
            'original_pcrf': [int(self.reference[r['user_id']]['pcrf_rank']) for r in c1_rows],
            'c1': [raw_rank(r) for r in c1_rows], 'c1_pcrf': self.pcrf(c1_rows),
            'gacr_v6': [raw_rank(r) for r in gacr_rows], 'gacr_v6_pcrf': self.pcrf(gacr_rows)}
        # PCRF is a permutation within each respective top50 set.
        assert np.array_equal(np.array(ranks['gacr_v6']) <= 50, np.array(ranks['gacr_v6_pcrf']) <= 50)
        fit_users = set((ROOT / self.config['c1_training_users']).read_text().splitlines())
        groups = {'all_validation': np.ones(len(c1_rows), dtype=bool),
            'outside_c1_training_users': np.array([r['user_id'] not in fit_users for r in c1_rows]),
            'inside_c1_training_users': np.array([r['user_id'] in fit_users for r in c1_rows]),
            'target_frequency_le5': np.array([self.frequencies[r['gold_item_id']] <= 5 for r in c1_rows]),
            'target_frequency_gt5': np.array([self.frequencies[r['gold_item_id']] > 5 for r in c1_rows])}
        group_results = {}
        for name, mask in groups.items():
            if not mask.any():
                continue
            selected = {key: np.asarray(value)[mask] for key, value in ranks.items()}
            group_results[name] = {'metrics': {key: metrics(value) for key, value in selected.items()},
                'combined_vs_original_pcrf': paired_delta(selected['gacr_v6_pcrf'], selected['original_pcrf'])}
        comparisons = {name: paired_delta(ranks['gacr_v6_pcrf'], ranks[name])
                       for name in ['original_gram', 'original_pcrf', 'c1_pcrf', 'gacr_v6']}
        with (self.output / 'per_user.tsv').open('w') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['user_id', *ranks])
            for index, row in enumerate(c1_rows):
                writer.writerow([row['user_id'], *[r[index] for r in ranks.values()]])
        metric_values = group_results['all_validation']['metrics']
        gain = metric_values['gacr_v6_pcrf']['ndcg@10'] - metric_values['original_pcrf']['ndcg@10']
        for path, expected in self.config['frozen_inputs'].items():
            assert sha256(ROOT / path) == expected
        write_json(self.output / 'summary.json', {'state': 'COMPLETED', 'dataset': 'Toys', 'users': len(c1_rows),
            'metrics': metric_values, 'groups': group_results, 'paired_comparisons': comparisons,
            'combined_ndcg_delta_vs_original_pcrf': gain, 'positive_signal': gain > 0,
            'decision': 'POSITIVE_EXPLORATORY_POINT_ESTIMATE' if gain > 0 else 'NO_INCREMENTAL_N10_THIS_COMBINATION',
            'new_optimizer_steps': 0, 'test_read': False, 'auto_beauty': False,
            'ci_caveat': 'Repeatedly used validation; descriptive paired intervals, not independent confirmation.',
            'combination_scope': 'PCRF with original fixed parameters and GACR residual score as base, on GACR top50; not an isolated residual effect on original GRAM.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['smoke', 'run'], required=True)
    args = parser.parse_args()
    path = ROOT / args.config
    config = json.loads(path.read_text())
    if args.mode == 'smoke':
        config['gpu_uuid'] = config['smoke_gpu_uuid']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    output = ROOT / config['smoke_output' if args.mode == 'smoke' else 'output']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Existing attempt; refusing overwrite or automatic retry')
    run = None
    try:
        gpu_setup(config)
        seed_everything(config['seed'])
        run = CombinationRun(config, path, output, 'run')
        def stop(signum, frame):
            run.stop_reason = f'SIGNAL_{signum}'
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]:
            signal.signal(sig, stop)
        signal.alarm(1800 if args.mode == 'smoke' else config['max_wall_seconds'])
        run.setup()
        run.check_stop()
        if args.mode == 'smoke':
            run.smoke()
        else:
            run.run_full()
        run.check_stop()
        signal.alarm(0)
        write_json(output / 'status.json', {'state': 'PASSED' if args.mode == 'smoke' else 'COMPLETED',
            'dataset': config['dataset'], 'pid': os.getpid(), 'mode': args.mode,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'elapsed_seconds': time.time()-run.started,
            'test_read': False, 'new_optimizer_steps': 0})
    except BaseException as error:
        write_json(output / 'status.json', {'state': 'STOPPED' if isinstance(error, Stopped) else 'FAILED',
            'error': repr(error), 'traceback': traceback.format_exc(), 'pid': os.getpid(), 'mode': args.mode,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'automatic_retry': False, 'test_read': False})
        raise


if __name__ == '__main__':
    main()
