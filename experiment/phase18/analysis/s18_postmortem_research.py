"""CPU-only retrospective measurement audit; never trains or changes an S18 gate.

Reads the already consumed S18-2 I0 artifacts, the S18-1R diagnostics, and the
frozen lexical catalog. No model checkpoint or protected fold is opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/phase18/s2_mechanism_probe/run-0001"
OUT = ROOT / "artifacts/phase18/postmortem_research/run-0001"
INPUTS = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8")


def obj(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def catalog(domain, config, tokenizer):
    info = config["domains"][domain]
    name = next(n for n in info["dataset"]["files"] if n.startswith("item_generative_indexing_"))
    path = ROOT / info["dataset"]["path"] / name
    lines = read(path).splitlines()
    assert INPUTS[str(path.relative_to(ROOT))] == info["dataset"]["files"][name]["sha256"]
    pairs = [line.split(" ", 1) for line in lines]
    # Match TestDatasetGRAM: retain original identifier text before encoding;
    # s18_s1_runtime.identifier_tokens removes the two lexical separator tokens.
    encoded = tokenizer([value for _, value in pairs], add_special_tokens=True)["input_ids"]
    paths = {item: tuple(t for t in tokens if t not in (1820, 9175))
             for (item, _), tokens in zip(pairs, encoded)}
    assert all(p[-1] == 1 for p in paths.values())
    assert len(set(paths.values())) == len(paths)
    children = defaultdict(set)
    for p in paths.values():
        for d, token in enumerate(p):
            children[p[:d]].add(token)
    return paths, children


def cache_audit(rs, paths, children):
    result = {"users": len(rs), "parent_hit50": 0, "parent_hit200": 0,
              "non_eos_prefix_drop_users": 0, "all_non_eos_survive_but_final_miss": 0,
              "drop_local_children_le_beam50": 0, "drop_local_single_child": 0}
    counts, depth_hist, node_positions = Counter(), Counter(), Counter()
    score_errors, negative_counts, sibling_child_counts = [], [], []
    comparisons = {a: {"same_child_nodes": 0, "nodes": 0, "all_nodes_same_users": 0,
                       "path_weight_tv": [], "child_jaccard": []}
                   for a in ("A0_LEGAL_GENERIC", "S0_SHUFFLED_CF")}
    for r in rs:
        p = paths[r["target"]]
        n = len(p) - 1
        hit50 = r["target"] in r["beam50_items"]
        result["parent_hit50"] += hit50
        result["parent_hit200"] += r["target"] in r["beam200_items"]
        surviving = r["parent_prefix_survival"] * n
        assert abs(surviving - round(surviving)) < 1e-7
        depth = round(surviving) + 1 if surviving < n - 1e-7 else None
        assert not hit50 or depth is None
        if depth is not None:
            result["non_eos_prefix_drop_users"] += 1
            depth_hist[str(depth)] += 1
            local = len(children[p[:depth - 1]])
            result["drop_local_children_le_beam50"] += local <= 50
            result["drop_local_single_child"] += local == 1
            # This remains a FINAL-beam descendant proxy, not a live frontier.
            sibling_prefixes = {paths[item][:depth] for item in r["beam50_items"]
                                if len(paths[item]) >= depth
                                and paths[item][:depth - 1] == p[:depth - 1]
                                and paths[item][depth - 1] != p[depth - 1]}
            sibling_child_counts.append(len(sibling_prefixes))
        elif not hit50:
            result["all_non_eos_survive_but_final_miss"] += 1
        score_map = dict(zip(r["beam200_items"], r["beam200_scores"]))
        score_errors.extend(abs(score_map[item] - score) for item, score in
                            zip(r["path_items"], r["parent_path_scores"]))
        if not r["arms"]:
            continue
        main = r["arms"]["M0_PCPS"]
        for node in main["nodes"]:
            d = node["depth"]
            assert node["target_child"] == p[d]
            neg = set(node["negative_children"])
            assert neg and p[d] not in neg and neg <= children[p[:d]]
            assert len(node["negative_items"]) <= 8
            assert neg == {paths[item][d] for item in node["negative_items"]}
            assert all(paths[item][:d] == p[:d] for item in node["negative_items"])
            negative_counts.append(len(neg))
            counts["legal_branch_nodes"] += 1
            counts["local_children_le_beam50_nodes"] += len(children[p[:d]]) <= 50
            if depth is not None:
                node_positions["before" if d + 1 < depth else "at" if d + 1 == depth else "after"] += 1
        for arm, stat in comparisons.items():
            other = r["arms"][arm]
            assert main["path_items"] == other["path_items"]
            assert len(main["nodes"]) == len(other["nodes"])
            same = True
            for x, y in zip(main["nodes"], other["nodes"]):
                assert (x["depth"], x["target_child"]) == (y["depth"], y["target_child"])
                a, b = set(x["negative_children"]), set(y["negative_children"])
                stat["same_child_nodes"] += a == b
                stat["nodes"] += 1
                stat["child_jaccard"].append(len(a & b) / len(a | b))
                same &= a == b
            stat["all_nodes_same_users"] += same
            stat["path_weight_tv"].append(.5 * sum(abs(a-b) for a, b in zip(main["weights"], other["weights"])))
    for stat in comparisons.values():
        for field in ("path_weight_tv", "child_jaccard"):
            stat["mean_" + field] = mean(stat.pop(field))
    result.update(counts)
    result.update(first_non_eos_drop_depth=dict(depth_hist),
                  training_loss_nodes_relative_to_parent_drop=dict(node_positions),
                  mean_unique_negative_children=mean(negative_counts),
                  mean_final_beam_sibling_prefixes_at_drop=mean(sibling_child_counts),
                  parent_teacher_force_vs_generation_score_max_abs_error=max(score_errors, default=0),
                  cf_comparisons=comparisons)
    return result


def s1r_audit(domain, paths, children):
    by_fold = {}
    for fold in ("im1", "i0"):
        p = ROOT / f"artifacts/phase18/s1r_disjoint_confirmation/run-0001/units/{domain.lower()}_{fold}/per_user_diagnostics.jsonl"
        rs = rows(p)
        events = [r for r in rs if r["beam200_only"]]
        counts = Counter(users=len(rs), beam200_only=len(events))
        for r in events:
            target, depth = paths[r["target"]], r["first_drop_depth"]
            legal = children[target[:depth-1]]
            counts["local_children_le_beam50"] += len(legal) <= 50
            counts["local_single_child"] += len(legal) == 1
            counts["eos_depth"] += depth == len(target)
            actual = r["actual_pruner_items"]
            counts["nonempty_final_sibling_item_proxy"] += bool(actual)
            counts["final_sibling_items"] += len(actual)
            counts["unique_final_sibling_prefixes"] += len({paths[item][:depth] for item in actual})
            assert all(paths[item][:depth-1] == target[:depth-1] and
                       paths[item][depth-1] != target[depth-1] for item in actual)
        by_fold[fold] = dict(counts)
    return by_fold


def paired_audit(summary):
    arms = {}
    max_error = 0
    for arm in summary["domains"]["Toys"]["probe"]:
        e = obj(BASE / f"Toys/probe/{arm}/evaluation.json")
        p = ROOT / e["per_user"]["path"]
        rs = list(csv.DictReader(read(p).splitlines(), delimiter="\t"))
        assert INPUTS[str(p.relative_to(ROOT))] == e["per_user"]["sha256"]
        assert len(rs) == len({r["user"] for r in rs}) == 1000
        arms[arm] = {r["user"]: r for r in rs}
        metrics = {"Hit@50": mean(int(r["rank"]) <= 50 for r in rs),
                   "NDCG@10": mean(1/math.log2(int(r["rank"])+1) if int(r["rank"]) <= 10 else 0 for r in rs),
                   "prefix_survival": mean(float(r["prefix_survival"]) for r in rs),
                   "path_margin": mean(float(r["path_margin"]) for r in rs)}
        max_error = max(max_error, *(abs(metrics[k]-e[k]) for k in metrics))
    paired = {}
    for arm, other in arms.items():
        if arm == "M0_PCPS":
            continue
        assert other.keys() == arms["M0_PCPS"].keys()
        differences = defaultdict(list)
        for user, m in arms["M0_PCPS"].items():
            o = other[user]
            assert m["target"] == o["target"]
            differences["Hit@50"].append(int(int(m["rank"]) <= 50)-int(int(o["rank"]) <= 50))
            differences["prefix_survival"].append(float(m["prefix_survival"])-float(o["prefix_survival"]))
        paired[arm] = {k: {"delta": mean(ds), "gain": sum(d > 1e-12 for d in ds),
                           "loss": sum(d < -1e-12 for d in ds), "tie": sum(abs(d) <= 1e-12 for d in ds)}
                       for k, ds in differences.items()}
    return {"max_summary_abs_error": max_error, "M0_minus_control": paired}


def main():
    if OUT.exists():
        raise RuntimeError("audit output already exists; preserve the original audit")
    from transformers import AutoTokenizer
    config = obj(ROOT / "experiment/phase18/config/s18_s2_mechanism_probe.json")
    s1 = obj(ROOT / "experiment/phase18/config/s18_s1_actionability.json")
    summary = obj(BASE / "summary.json")
    assert summary["decision"] == "FAILED_SCIENTIFIC_GATE"
    tokenizer = AutoTokenizer.from_pretrained(s1["backbone"]["snapshot"], local_files_only=True)
    results = {"verification_status": "ANALYZED", "kind": "retrospective_existing_cache_audit",
               "canonical_s18_2_decision": summary["decision"], "gate_changed": False,
               "model_loaded": False, "training_started": False,
               "protected_data_read": False, "live_frontier_available": False,
               "paired": paired_audit(summary), "domains": {}}
    for domain in ("Toys", "Beauty"):
        paths, children = catalog(domain, config, tokenizer)
        result = {"catalog_items": len(paths), "training_cache": cache_audit(
            rows(BASE / domain / "training_cache.jsonl"), paths, children),
            "s1r_existing_confirmation": s1r_audit(domain, paths, children)}
        if domain == "Toys":
            result["evaluation_anchors"] = cache_audit(rows(BASE/domain/"evaluation_anchors.jsonl"), paths, children)
        results["domains"][domain] = result
    # A constructive counterexample: target is its parent's best child but is
    # globally below two children of another parent, so beam2 discards it.
    examples = {"target": .4*.51, "sibling": .4*.49, "other_parent_a": .6*.5, "other_parent_b": .6*.5}
    assert examples["target"] > examples["sibling"]
    assert sum(s > examples["target"] for s in examples.values()) == 2
    results["local_rank_counterexample"] = examples
    results["input_sha256"] = INPUTS
    results["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.mkdir(parents=True)
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in results.items() if k != "input_sha256"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
