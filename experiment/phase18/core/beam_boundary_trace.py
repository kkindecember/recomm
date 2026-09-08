"""Passive, single-example tracing of the pinned Transformers beam scorer.

No scores, candidate orders, scorer decisions, or model tensors are changed.
The scoped class hooks are intentionally single-threaded and not reentrant.
"""
import math
from contextlib import contextmanager

import torch
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessor


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


class BoundaryTrace(LogitsProcessor):
    _installed = False

    def __init__(self, target, width=50, eos=1):
        self.target, self.width, self.eos = list(target), width, eos
        self.steps = []
        self.scores = None
        self.cumulative = None
        self.valid = [True] + [False] * (width - 1)
        self.expected_input = None
        self.first_drop = None
        self.final = None

    def __call__(self, input_ids, scores):
        if self.scores is not None:
            raise RuntimeError("processor called twice without a scorer step")
        if input_ids.shape[0] != self.width:
            raise RuntimeError("tracer requires batch size 1, ordinary beam search")
        if self.expected_input is not None and not torch.equal(input_ids, self.expected_input):
            raise RuntimeError("native active set differs from reconstructed scorer output")
        self.scores = scores.detach()
        return scores

    def observe(self, scorer, input_ids, next_scores, next_tokens, next_indices, result, was_done):
        scores, self.scores = self.scores, None
        if scores is None or len(scorer._beam_hyps) != 1 or scorer.num_beam_groups != 1:
            raise RuntimeError("unsupported scorer or missing processed logits")
        if self.cumulative is None:
            self.cumulative = scores.new_full((self.width,), -1e9)
            self.cumulative[0] = 0
        total = scores + self.cumulative[:, None]
        # Verify that this processor observes the exact values consumed by topk.
        selected = total[next_indices[0], next_tokens[0]]
        if not torch.equal(selected, next_scores[0]):
            raise RuntimeError("a later logits processor changed the observed scores")
        inputs = input_ids.detach().cpu().tolist()
        depth = len(inputs[0]) - 1
        cumulative = self.cumulative.cpu().tolist()
        legal = torch.isfinite(scores)
        valid_mask = torch.tensor(self.valid, dtype=torch.bool, device=scores.device)
        valid_legal = legal & valid_mask[:, None]
        parents = [i for i, p in enumerate(inputs) if self.valid[i] and p[1:] == self.target[:depth]]
        target = {"parent_alive": bool(parents), "is_eos": depth < len(self.target) and self.target[depth] == self.eos}
        if len(parents) > 1:
            raise RuntimeError("duplicate valid target parent")
        if parents and depth < len(self.target):
            p, token = parents[0], self.target[depth]
            if not bool(legal[p, token]):
                raise RuntimeError("target token not legal under surviving target parent")
            s = total[p, token]
            target.update(parent_index=p, token=token, prefix=self.target[:depth + 1],
                          local_rank_strict=int((scores[p] > scores[p, token]).sum()) + 1,
                          local_tied_children=int((scores[p] == scores[p, token]).sum()),
                          legal_child_count=int(legal[p].sum()), score=number(s),
                          global_strictly_above=int(((total > s) & valid_legal).sum()),
                          global_tied=int(((total == s) & valid_legal).sum()))
        candidates = []
        ns, nt, ni = (x[0].detach().cpu().tolist() for x in (next_scores, next_tokens, next_indices))
        for rank, (s, token, p) in enumerate(zip(ns, nt, ni)):
            candidates.append({"rank": rank + 1, "parent_index": p, "token": token,
                               "prefix": inputs[p][1:] + [token], "score": number(s),
                               "valid": bool(self.valid[p] and math.isfinite(s) and legal[p, token]),
                               "is_eos": token == self.eos})
        indices = result["next_beam_indices"]
        tokens = result["next_beam_tokens"]
        self.expected_input = torch.cat([input_ids[indices], tokens[:, None]], dim=-1).detach().clone()
        kept = []
        for p, token, s in zip(indices.cpu().tolist(), tokens.cpu().tolist(), result["next_beam_scores"].cpu().tolist()):
            kept.append({"parent_index": p, "token": token, "prefix": inputs[p][1:] + [token],
                         "score": number(s), "padding": was_done,
                         "valid": bool(not was_done and self.valid[p] and math.isfinite(s) and legal[p, token])})
        eos_offered, reconstructed = [], []
        if not was_done:
            for c in candidates:
                if c["is_eos"]:
                    if c["rank"] <= self.width:
                        eos_offered.append(c)
                else:
                    reconstructed.append(c)
                if len(reconstructed) == self.width:
                    break
            if [(c["parent_index"], c["token"], c["score"]) for c in reconstructed] != [
                    (c["parent_index"], c["token"], c["score"]) for c in kept]:
                raise RuntimeError("trace cannot reconstruct native scorer selection")
        # EOS offered to BeamHypotheses can be discarded from its bounded heap.
        heap = [{"normalized_score": number(s), "prefix": ids.cpu().tolist()[1:]}
                for s, ids, _ in scorer._beam_hyps[0].beams]
        if "prefix" in target:
            target["retained_active"] = any(c["valid"] and c["prefix"] == target["prefix"] for c in kept)
            target["eos_offered"] = any(c["valid"] and c["prefix"] == target["prefix"] for c in eos_offered)
            if not target["is_eos"] and not target["retained_active"] and self.first_drop is None:
                self.first_drop = len(self.steps)
        valid_kept = [c for c in kept if c["valid"]]
        boundary = valid_kept[-1] if len(valid_kept) == self.width else None
        step = {"depth": depth + 1, "input_prefixes": [p[1:] for p in inputs],
                "input_scores": [number(s) for s in cumulative], "input_valid": self.valid,
                "done_before": was_done, "done_after": bool(scorer._done[0]),
                "legal_extensions": int(valid_legal.sum()), "target": target,
                "top2b": candidates, "retained": kept, "eos_offered": eos_offered,
                "eos_heap_after": heap, "boundary": boundary,
                "candidate_set_incomplete": len(valid_kept) < self.width,
                "native_reconstruction_exact": True}
        self.steps.append(step)
        self.valid = [c["valid"] for c in kept]
        self.cumulative = result["next_beam_scores"].detach().clone()

    @contextmanager
    def installed(self):
        if BoundaryTrace._installed:
            raise RuntimeError("beam tracer hooks are not reentrant")
        original_process, original_finalize = BeamSearchScorer.process, BeamSearchScorer.finalize
        trace = self

        def process(scorer, input_ids, next_scores, next_tokens, next_indices, **kwargs):
            was_done = bool(scorer._done[0])
            result = original_process(scorer, input_ids, next_scores, next_tokens, next_indices, **kwargs)
            trace.observe(scorer, input_ids, next_scores, next_tokens, next_indices, result, was_done)
            return result

        def finalize(scorer, *args, **kwargs):
            result = original_finalize(scorer, *args, **kwargs)
            trace.final = {k: v.detach().cpu().tolist() if v is not None else None for k, v in result.items()}
            return result

        BoundaryTrace._installed = True
        BeamSearchScorer.process, BeamSearchScorer.finalize = process, finalize
        try:
            yield self
        finally:
            BeamSearchScorer.process, BeamSearchScorer.finalize = original_process, original_finalize
            BoundaryTrace._installed = False
            self.scores = None
            self.expected_input = None
            self.cumulative = None

    def payload(self):
        return {"steps": self.steps, "first_non_eos_drop_index": self.first_drop, "final": self.final}


def first_drop_summary(trace, item_paths, final_items, cache):
    """Compare prefix sets; keep item denominators distinct from prefix counts."""
    if trace.first_drop is None:
        return {"event": False, "reason": "no_non_eos_drop_observed"}
    step = trace.steps[trace.first_drop]
    target, depth = step["target"], step["depth"]
    prefix = tuple(target["prefix"])
    competitors = {tuple(c["prefix"]) for c in step["retained"] if c["valid"]}
    sibling_items = [i for i in final_items if len(item_paths[i]) >= depth and
                     item_paths[i][:depth - 1] == prefix[:-1] and item_paths[i][depth - 1] != prefix[-1]]
    proxy = {item_paths[i][:depth] for i in sibling_items}
    boundary = step["boundary"]
    result = {"event": True, "depth": depth, "target_prefix": list(prefix),
              "target_score": target["score"], "local_rank": target["local_rank_strict"],
              "local_ties": target["local_tied_children"], "branch_degree": target["legal_child_count"],
              "global_above": target["global_strictly_above"], "global_ties": target["global_tied"],
              "competitor_prefixes": len(competitors),
              "cross_parent_competitor_prefixes": sum(p[:-1] != prefix[:-1] for p in competitors),
              "proxy_item_count": len(sibling_items), "proxy_prefix_count": len(proxy),
              "proxy_overlap_prefix_count": len(proxy & competitors),
              "eos_offered_count": len(step["eos_offered"]),
              "candidate_set_incomplete": step["candidate_set_incomplete"],
              "boundary_prefix": boundary["prefix"] if boundary else None,
              "boundary_score": boundary["score"] if boundary else None,
              "native_margin": target["score"] - boundary["score"] if boundary else None,
              "boundary_cross_parent": boundary["prefix"][:-1] != list(prefix[:-1]) if boundary else None}
    old_paths = {item_paths[i][:depth] for i in cache["path_items"] if len(item_paths[i]) >= depth}
    result["old_path_prefix_count"] = len(old_paths)
    result["old_path_overlap_prefix_count"] = len(old_paths & competitors)
    result["arm_prefix_overlap"] = {}
    for arm, info in cache.get("arms", {}).items():
        nodes = [n for n in info["nodes"] if n["depth"] == depth - 1]
        mined = {prefix[:-1] + (t,) for n in nodes for t in n["negative_children"]}
        result["arm_prefix_overlap"][arm] = {"mined_prefixes": len(mined), "overlap": len(mined & competitors)}
    return result
