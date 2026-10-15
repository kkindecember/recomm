"""Read-only HF beam observation and post-generation target-path analysis."""
from contextlib import contextmanager
import math
import numpy as np


class BeamObserver:
    """No target or history input; return original processor/scorer outputs."""
    def __init__(self, guide, beams=50):
        self.guide, self.beams = guide, beams
        self.steps = []
        self.parent_scores = [0.] + [-1e9] * (beams - 1)
        self.final = None

    def __call__(self, input_ids, scores):
        prefixes = input_ids.detach().cpu().tolist()
        rr, tt, counts, native, masses = [], [], [], [], []
        token_lists = []
        for i, raw in enumerate(prefixes):
            if tuple(raw) in self.guide.trace:
                tokens, values, mass = self.guide.trace[tuple(raw)]
            else:
                # HF may carry masked -inf filler beams while final EOS queues settle.
                import torch
                assert bool(torch.isneginf(scores[i]).all())
                tokens, values, mass = [], np.asarray([]), -math.inf
            token_lists.append(tokens)
            rr.extend([i] * len(tokens)); tt.extend(tokens); counts.append(len(tokens))
            native.append(values.tolist()); masses.append(float(mass))
        import torch
        r = torch.tensor(rr, device=scores.device, dtype=torch.long)
        t = torch.tensor(tt, device=scores.device, dtype=torch.long)
        used = scores[r, t].detach().cpu().tolist()
        rows, offset = [], 0
        for tokens, values, mass, size in zip(token_lists, native, masses, counts):
            rows.append(dict(tokens=tokens, native=values, used=used[offset:offset+size], legal_log_mass=mass))
            offset += size
        self.steps.append(dict(depth=len(prefixes[0]), prefixes=prefixes,
                               parent_scores=list(self.parent_scores), rows=rows))
        return scores

    @staticmethod
    def hypotheses(scorer):
        assert len(scorer._beam_hyps) == 1
        return [dict(score=float(x[0]), prefix=x[1].detach().cpu().tolist())
                for x in scorer._beam_hyps[0].beams]

    @contextmanager
    def installed(self):
        from transformers.generation.beam_search import BeamSearchScorer
        original_process, original_finalize = BeamSearchScorer.process, BeamSearchScorer.finalize
        observer = self

        def process(scorer, input_ids, next_scores, next_tokens, next_indices, **kwargs):
            step = observer.steps[-1]
            assert step['prefixes'] == input_ids.detach().cpu().tolist()
            step['top'] = dict(scores=next_scores[0].detach().cpu().tolist(),
                               tokens=next_tokens[0].detach().cpu().tolist(),
                               indices=next_indices[0].detach().cpu().tolist())
            result = original_process(scorer, input_ids, next_scores, next_tokens, next_indices, **kwargs)
            step['selected'] = {k: v.detach().cpu().tolist() for k, v in result.items()}
            step['finished'] = observer.hypotheses(scorer)
            step['done'] = bool(scorer.is_done)
            observer.parent_scores = step['selected']['next_beam_scores']
            return result

        def finalize(scorer, input_ids, final_beam_scores, final_beam_tokens, final_beam_indices, **kwargs):
            observer.final = dict(prefixes=input_ids.detach().cpu().tolist(),
                                  scores=final_beam_scores.detach().cpu().tolist(),
                                  finished=observer.hypotheses(scorer), done=bool(scorer.is_done))
            return original_finalize(scorer, input_ids, final_beam_scores, final_beam_tokens,
                                     final_beam_indices, **kwargs)

        BeamSearchScorer.process, BeamSearchScorer.finalize = process, finalize
        try:
            yield self
        finally:
            BeamSearchScorer.process, BeamSearchScorer.finalize = original_process, original_finalize


def validate_steps(steps):
    """Rebuild actual cross-parent scores, without target-path assumptions."""
    errors = []
    for j, step in enumerate(steps):
        if j:
            previous = steps[j-1]
            selected = previous['selected']
            expected = [previous['prefixes'][i] + [t] for i, t in
                        zip(selected['next_beam_indices'], selected['next_beam_tokens'])]
            assert expected == step['prefixes']
            assert selected['next_beam_scores'] == step['parent_scores']
        for score, token, parent in zip(step['top']['scores'], step['top']['tokens'], step['top']['indices']):
            row = step['rows'][parent]
            # At final steps HF top100 can contain masked fillers with -inf.
            if not math.isfinite(score):
                assert token not in row['tokens']
                continue
            pos = row['tokens'].index(token)
            expected = np.float32(np.float32(row['used'][pos]) + np.float32(step['parent_scores'][parent]))
            errors.append(abs(float(expected) - score))
        for score, token, parent in zip(step['selected']['next_beam_scores'],
                                       step['selected']['next_beam_tokens'],
                                       step['selected']['next_beam_indices']):
            if not math.isfinite(score):
                continue
            pos = step['rows'][parent]['tokens'].index(token)
            expected = np.float32(np.float32(step['rows'][parent]['used'][pos]) + np.float32(step['parent_scores'][parent]))
            errors.append(abs(float(expected) - score))
    error = max(errors, default=0.)
    assert error <= 2e-6, error
    return dict(steps=len(steps), cumulative_scores_checked=len(errors), max_abs_error=error)


def target_path_analysis(steps, path, raw_ids, target, final_rank, expert_logs, item_scores, history):
    """The held-out answer is accepted only here, after generation ends."""
    events = []
    for step in steps:
        depth = step['depth']
        if depth >= len(path):
            continue
        matching = [i for i,p in enumerate(step['prefixes'])
                    if p == path[:depth] and step['parent_scores'][i] > -1e8]
        if not matching:
            continue
        parent = max(matching, key=lambda i: step['parent_scores'][i])
        token = path[depth]; row = step['rows'][parent]
        pos = row['tokens'].index(token)
        native = np.asarray(row['native']); used = np.asarray(row['used'], dtype=np.float32)
        expert = np.asarray([expert_logs[tuple(path[:depth])][t] for t in row['tokens']])
        cumulative = float(np.float32(used[pos] + np.float32(step['parent_scores'][parent])))
        all_values = np.concatenate([np.asarray(r['used'],dtype=np.float32) + np.float32(s)
                                     for r,s in zip(step['rows'],step['parent_scores']) if s > -1e8])
        selected = step['selected']
        kept = any(i == parent and t == token for i,t in
                   zip(selected['next_beam_indices'],selected['next_beam_tokens']))
        eos = token == 1
        admitted = any(h['prefix'] == path[:-1] for h in step['finished']) if eos else False
        finite_selected = [x for x in selected['next_beam_scores'] if math.isfinite(x)]
        event = dict(depth=depth, children=len(row['tokens']), forced=len(row['tokens'])==1,
                     native_child_rank=1+int(np.sum(native > native[pos])),
                     expert_child_rank=1+int(np.sum(expert > expert[pos])),
                     used_child_rank=1+int(np.sum(used > used[pos])),
                     native_conditional_probability=math.exp(float(native[pos])-row['legal_log_mass']),
                     expert_conditional_probability=math.exp(float(expert[pos])),
                     expert_log_ratio=float(expert[pos] - (native[pos]-row['legal_log_mass'])),
                     global_rank_min=1+int(np.sum(all_values>cumulative)),
                     global_rank_max=int(np.sum(all_values>=cumulative)),
                     cumulative_score=cumulative,
                     margin_to_active_cutoff=cumulative-min(finite_selected) if finite_selected else None,
                     kept_active=kept, eos=eos, admitted_finished=admitted, done=step['done'])
        events.append(event)
        if eos or not kept:
            break
    first_loss = next((e for e in events if not e['eos'] and not e['kept_active']), None)
    eos_event = next((e for e in events if e['eos']), None)
    if target in history:
        outcome = 'excluded_by_existing_history_rule'
    elif target in raw_ids:
        outcome = 'returned_top10' if final_rank <= 10 else 'returned_below_top10'
    elif first_loss:
        outcome = 'prefix_pruned'
    elif eos_event:
        outcome = 'finished_queue_or_final_selection' if eos_event['admitted_finished'] else 'eos_not_admitted'
    else:
        outcome = 'search_stopped_before_target_completion'
    scores = np.asarray(item_scores)
    allowed = np.ones(len(scores),dtype=bool)
    allowed[np.asarray(history,dtype=int)-1] = False
    target_score = scores[target-1]
    return dict(outcome=outcome, events=events, first_loss=first_loss,
                lime_catalog_rank=1+int(np.sum(scores[allowed]>target_score)),
                lime_catalog_rank_tie_end=int(np.sum(scores[allowed]>=target_score)),
                final_rank=final_rank, raw_target_present=target in raw_ids)
