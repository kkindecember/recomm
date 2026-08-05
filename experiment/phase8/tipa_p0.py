#!/usr/bin/env python3
"""TIPA-P0: target-free item-to-lexical-path alignment pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gacr_s0 import select_stratified_samples
from experiment.phase4.gcdh_p0 import build_train_samples, collate, prepare, read_users, sha256, write_json
from experiment.phase7.gcgd_p1 import AdaptiveGraphPrefixLogitsProcessor, arm_metric_row, generate_arm_items, graph_prefix_inputs
from experiment.phase7.gcgd_p1_run import summarize_rows
from experiment.phase7.st_gcgd_v2 import audit_arm_rows
from experiment.phase7.st_gcgd_v21 import TransitionFirstGraph, load_inputs_v21
from experiment.phase7.st_gcgd_v21_p1 import transition_logits_for_sample


def stable_user_split(users: set[str], dataset: str, salt: str, fit_fraction: float) -> tuple[set[str], set[str]]:
    ordered = sorted(users, key=lambda user: hashlib.sha256(f"{salt}|{dataset}|{user}".encode()).hexdigest())
    cut = int(round(len(ordered) * float(fit_fraction)))
    return set(ordered[:cut]), set(ordered[cut:])


def normalized_child_features(
    gram_logits: torch.Tensor,
    teacher_log_probabilities: torch.Tensor,
    *,
    depth: int,
    maximum_depth: int,
    leaf_fraction: float,
) -> torch.Tensor:
    if gram_logits.ndim != 1 or teacher_log_probabilities.shape != gram_logits.shape:
        raise ValueError("child feature inputs must be aligned vectors")
    if not torch.isfinite(gram_logits).all() or not torch.isfinite(teacher_log_probabilities).all():
        raise ValueError("non-finite child feature input")
    gram = (gram_logits - gram_logits.mean()) / gram_logits.std(unbiased=False).clamp_min(1e-6)
    teacher_probability = teacher_log_probabilities.exp()
    entropy = 0.0 if len(teacher_probability) == 1 else float(
        -(teacher_probability * teacher_log_probabilities).sum() / math.log(len(teacher_probability))
    )
    ordered = teacher_probability.sort(descending=True).values
    margin = 1.0 if len(ordered) == 1 else float(ordered[0] - ordered[1])
    shared = gram.new_tensor([
        min(1.0, max(0.0, (depth - 1) / max(1, maximum_depth))),
        min(1.0, max(0.0, entropy)), min(1.0, max(0.0, margin)), float(leaf_fraction),
    ]).expand(len(gram), 4)
    return torch.cat((gram[:, None], teacher_log_probabilities[:, None], shared), dim=1)


class PathAlignmentAdapter(torch.nn.Module):
    """Shared per-child adapter; the zero-initialized output is exact identity."""

    def __init__(self, hidden_size: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(6, int(hidden_size)), torch.nn.LayerNorm(int(hidden_size)), torch.nn.GELU(),
            torch.nn.Linear(int(hidden_size), int(hidden_size)), torch.nn.GELU(),
            torch.nn.Linear(int(hidden_size), 1),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features).squeeze(-1)
        centered = raw - raw.mean(dim=-1, keepdim=False)
        return self.bound * torch.tanh(centered)


class TIPAProcessor(LogitsProcessor):
    def __init__(self, prefix_scores, leaf_fractions, adapter, maximum_depth: int) -> None:
        self.prefix_scores = prefix_scores
        self.leaf_fractions = leaf_fractions
        self.adapter = adapter
        self.maximum_depth = int(maximum_depth)
        self.calls = 0
        self.applied_rows = 0
        self.null_rows = 0
        self.max_abs_delta = 0.0

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.calls += 1
        output = scores.clone()
        for row, ids in enumerate(input_ids.tolist()):
            prefix = tuple(ids)
            teacher = self.prefix_scores.get(prefix)
            if not teacher or len(teacher) <= 1:
                self.null_rows += 1
                continue
            tokens = sorted(teacher)
            gram = scores[row, tokens].float()
            teacher_values = gram.new_tensor([teacher[token] for token in tokens])
            features = normalized_child_features(
                gram, teacher_values, depth=len(prefix), maximum_depth=self.maximum_depth,
                leaf_fraction=float(self.leaf_fractions[prefix]),
            )
            delta = self.adapter(features).to(scores.dtype)
            output[row, tokens] += delta
            self.max_abs_delta = max(self.max_abs_delta, float(delta.abs().max()))
            self.applied_rows += 1
        return output


def teacher_model(dataset: str, config: Mapping[str, object], device: torch.device):
    graph, sequence_path, catalog_path = load_inputs_v21(dataset, config["teacher"]["temporal_lineage"])
    spec = config["teacher"]["architecture"]
    model = TransitionFirstGraph(
        len(graph.users), len(graph.items), int(spec["embedding_dim"]), float(spec["maximum_ui_fraction"]),
        layers=int(spec["layers"]), dropout=float(spec["dropout"]),
        maximum_session_length=int(spec["maximum_session_length"]),
    ).to(device)
    checkpoint = ROOT / config["teacher"]["checkpoints"][dataset]["path"]
    if sha256(checkpoint) != config["teacher"]["checkpoints"][dataset]["sha256"]:
        raise ValueError(f"{dataset} teacher checkpoint SHA mismatch")
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    with torch.no_grad():
        propagated = model.propagate_deep(graph)
    return graph, model, propagated, sequence_path, catalog_path, checkpoint


@torch.no_grad()
def gram_child_logits(sample: dict, prepared: dict, prefix: Sequence[int], legal_tokens: Sequence[int], device: torch.device) -> torch.Tensor:
    batch = collate(prepared["collator"], [sample])
    output = prepared["model"].backbone(
        input_ids=batch["item_text_ids"].to(device), attention_mask=batch["item_text_masks"].to(device),
        decoder_input_ids=torch.tensor([list(prefix)], dtype=torch.long, device=device), return_dict=True,
    )
    return output.logits[0, -1, list(legal_tokens)].float()


def select_prefix_depth(path, prefix_scores, seed: int, dataset: str, sample_key: str, mode: str) -> int | None:
    if mode == "random_all_depths":
        depths = list(range(1, len(path)))
    elif mode == "branching_teacher_path":
        depths = [depth for depth in range(1, len(path)) if len(prefix_scores.get(tuple(path[:depth]), {})) > 1]
    else:
        raise ValueError(f"unknown prefix sampling mode: {mode}")
    if not depths:
        return None
    position = int(hashlib.sha256(f"{seed}|{dataset}|tipa-prefix|{sample_key}".encode()).hexdigest(), 16) % len(depths)
    return depths[position]


def adapter_record(sample, prepared, item_paths, leaf_fractions, teacher_logits, maximum_depth, device, seed, dataset, sampling_mode="random_all_depths"):
    prefix_scores, _ = graph_prefix_inputs(item_paths, teacher_logits)
    ranked = sorted(teacher_logits, key=lambda item: (-teacher_logits[item], item))
    chosen = next(item for item in ranked if item not in set(sample["history_items"]))
    path = tuple(item_paths[chosen])
    depth = select_prefix_depth(path, prefix_scores, seed, dataset, sample["sample_key"], sampling_mode)
    if depth is None:
        return None
    prefix = path[:depth]
    teacher = prefix_scores.get(prefix)
    if not teacher or len(teacher) <= 1:
        return None
    tokens = sorted(teacher)
    gram = gram_child_logits(sample, prepared, prefix, tokens, device).cpu()
    teacher_values = torch.tensor([teacher[token] for token in tokens], dtype=torch.float32)
    features = normalized_child_features(
        gram, teacher_values, depth=depth, maximum_depth=maximum_depth,
        leaf_fraction=float(leaf_fractions[prefix]),
    )
    return {"sample_key": sample["sample_key"], "prefix": prefix, "tokens": tokens, "features": features,
            "gram": gram, "teacher": teacher_values, "teacher_mass_error": abs(float(teacher_values.exp().sum()) - 1.0)}


def train_adapter(records: list[dict], config: Mapping[str, object], device: torch.device):
    spec = config["adapter"]
    torch.manual_seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"]))
    adapter = PathAlignmentAdapter(int(spec["hidden_size"]), float(spec["bound"])).to(device)
    initial_max = max(float(adapter(row["features"].to(device)).abs().max()) for row in records)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    history = []
    for step in range(1, int(spec["fixed_steps"]) + 1):
        selected = [records[((step - 1) * int(spec["batch_prefixes"]) + offset) % len(records)] for offset in range(int(spec["batch_prefixes"]))]
        optimizer.zero_grad(set_to_none=True); losses=[]; kls=[]; identities=[]
        for row in selected:
            gram=row["gram"].to(device); teacher=row["teacher"].to(device); delta=adapter(row["features"].to(device))
            log_distribution=F.log_softmax(gram + delta, dim=0)
            kl=F.kl_div(log_distribution, teacher.exp(), reduction="sum")
            identity=delta.square().mean(); loss=kl + float(spec["identity_weight"]) * identity
            losses.append(loss); kls.append(kl); identities.append(identity)
        total=torch.stack(losses).mean()
        if not torch.isfinite(total): raise ValueError("non-finite TIPA loss")
        total.backward(); norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(spec["gradient_clip"])); optimizer.step()
        if step == 1 or step % 10 == 0 or step == int(spec["fixed_steps"]):
            history.append({"step":step,"loss":float(total.detach()),"teacher_kl":float(torch.stack(kls).mean().detach()),
                            "identity_penalty":float(torch.stack(identities).mean().detach()),"gradient_norm":float(norm)})
    adapter.eval()
    return adapter, history, initial_max


def rank_agreement(items: Sequence[str], teacher_logits: Mapping[str, float]) -> float:
    if len(items) < 2: return 0.0
    concordant=discordant=0
    for left in range(len(items)):
        for right in range(left+1,len(items)):
            a,b=items[left],items[right]
            if (teacher_logits[a], a) > (teacher_logits[b], b): concordant += 1
            else: discordant += 1
    return (concordant-discordant)/max(1,concordant+discordant)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_domain(dataset: str, config: dict, output_root: Path) -> dict:
    device=torch.device("cuda:0"); torch.manual_seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"]))
    parent=json.loads((ROOT/config["inputs"]["phase4_parent_config"]).read_text()); prepared=prepare(dataset,parent,device)
    parent_checkpoint=ROOT/config["inputs"]["checkpoint_root"]/dataset/"C1/model.pt"
    expected_parent=config["inputs"]["parent_checkpoint_sha256"][dataset]
    if sha256(parent_checkpoint)!=expected_parent: raise ValueError("parent checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(parent_checkpoint,map_location=device),strict=True); prepared["model"].eval()
    graph,teacher,propagated,sequence_path,catalog_path,teacher_checkpoint=teacher_model(dataset,config,device)
    if tuple(graph.items)!=tuple(prepared["catalog"]): raise ValueError("teacher/GRAM catalog order mismatch")
    item_paths=dict(zip(prepared["catalog"],prepared["encoded_candidates"])); _,leaf_fractions=graph_prefix_inputs(item_paths,{item:0.0 for item in prepared["catalog"]})
    maximum_depth=max(map(len,item_paths.values()))-1
    train_users=read_users(ROOT/config["inputs"]["split_root"]/dataset/"train_users.txt")
    fit_users,calibration_users=stable_user_split(train_users,dataset,config["split"]["salt"],config["split"]["fit_fraction"])
    fit_pool=build_train_samples(prepared["sequences"],fit_users,prepared["item2input"],prepared["item2lexid"])
    cal_pool=build_train_samples(prepared["sequences"],calibration_users,prepared["item2input"],prepared["item2lexid"])
    fs=int(config["sample"]["fit_per_group"]); cs=int(config["sample"]["calibration_per_group"])
    fit_samples=select_stratified_samples(fit_pool,prepared["heads"],int(config["seed"]),f"{dataset}|tipa-fit",fs,fs)
    cal_samples=select_stratified_samples(cal_pool,prepared["heads"],int(config["seed"]),f"{dataset}|tipa-cal",cs,cs)
    records=[]; prefix_rows=[]
    for index,sample in enumerate(fit_samples,1):
        logits=transition_logits_for_sample(teacher,graph,sample,propagated)
        record=adapter_record(sample,prepared,item_paths,leaf_fractions,logits,maximum_depth,device,int(config["seed"]),dataset,config["sample"].get("prefix_sampling","random_all_depths"))
        if record is not None:
            records.append(record); prefix_rows.append({"dataset":dataset,"sample_key":record["sample_key"],"prefix":" ".join(map(str,record["prefix"])),
                                                        "legal_children":len(record["tokens"]),"teacher_mass_error":record["teacher_mass_error"]})
        if index%32==0: print(f"TIPA_P0_RECORDS dataset={dataset} records={index}/{len(fit_samples)}",flush=True)
    output=output_root/dataset; output.mkdir(parents=True,exist_ok=True)
    prefix_audit={"dataset":dataset,"attempted_records":len(fit_samples),"usable_branching_records":len(records),"null_or_single_child_records":len(fit_samples)-len(records),
                  "minimum_required":int(config["sample"]["minimum_prefix_records"]),"sampling_mode":config["sample"].get("prefix_sampling","random_all_depths")}
    write_json(output/"prefix_availability_audit.json",prefix_audit)
    if prefix_rows: write_csv(output/"per_prefix.csv",prefix_rows)
    print(f"TIPA_P0_PREFIX_AUDIT dataset={dataset} usable={len(records)} attempted={len(fit_samples)} minimum={config['sample']['minimum_prefix_records']}",flush=True)
    if len(records)<int(config["sample"]["minimum_prefix_records"]): raise ValueError("insufficient non-null prefix records")
    adapter,training,initial_max=train_adapter(records,config,device)
    arms={key:[] for key in ("A","B","C")}; rows=[]; identity_exact=False
    lengths=[len(sample["history_items"]) for sample in cal_samples]; length_median=float(np.median(lengths))
    teacher_margins=[]; cached=[]
    for sample in cal_samples:
        logits=transition_logits_for_sample(teacher,graph,sample,propagated); ordered=sorted(logits.values(),reverse=True)
        margin=ordered[0]-ordered[1]; teacher_margins.append(margin); cached.append((sample,logits,margin))
    margin_median=float(np.median(teacher_margins)); transition_sources=set(graph.transition_edges[0].tolist()); item_to_index={item:i for i,item in enumerate(graph.items)}
    mass_error=max(row["teacher_mass_error"] for row in records); start=time.perf_counter()
    for index,(sample,logits,teacher_margin) in enumerate(cached,1):
        prefix_scores,_=graph_prefix_inputs(item_paths,logits)
        baseline,_=generate_arm_items(sample,prepared,beam_size=50,length_penalty=1.0,device=device,processor=None)
        bproc=AdaptiveGraphPrefixLogitsProcessor(prefix_scores,leaf_fractions,alpha=float(config["decoding"]["B_alpha"]),maximum_depth=maximum_depth,adapter=None)
        bitems,_=generate_arm_items(sample,prepared,beam_size=50,length_penalty=1.0,device=device,processor=bproc)
        cproc=TIPAProcessor(prefix_scores,leaf_fractions,adapter,maximum_depth)
        citems,_=generate_arm_items(sample,prepared,beam_size=50,length_penalty=1.0,device=device,processor=cproc)
        if index==1:
            zero=PathAlignmentAdapter(int(config["adapter"]["hidden_size"]),float(config["adapter"]["bound"])).to(device)
            zproc=TIPAProcessor(prefix_scores,leaf_fractions,zero,maximum_depth)
            zitems,_=generate_arm_items(sample,prepared,beam_size=50,length_penalty=1.0,device=device,processor=zproc); identity_exact=zitems==baseline
            if not identity_exact: raise ValueError("zero-adapter identity failed")
        target=sample["positive_item"]; group="head" if target in prepared["heads"] else "tail"; last=item_to_index[sample["history_items"][-1]]
        common={"sample_key":sample["sample_key"],"target":target,"target_group":group,
                "history_group":"short" if len(sample["history_items"])<=length_median else "long",
                "teacher_margin_group":"low" if teacher_margin<=margin_median else "high",
                "transition_covered":int(last in transition_sources),"teacher_margin":teacher_margin,
                "teacher_target_rank":sorted(logits,key=lambda item:(-logits[item],item)).index(target)+1,
                "A_rank":baseline.index(target)+1 if target in baseline else "","B_rank":bitems.index(target)+1 if target in bitems else "",
                "C_rank":citems.index(target)+1 if target in citems else "","B_kendall":rank_agreement(bitems,logits),"C_kendall":rank_agreement(citems,logits),
                "C_null_rate":cproc.null_rows/max(1,cproc.null_rows+cproc.applied_rows),"C_max_abs_delta":cproc.max_abs_delta}
        rows.append(common)
        arms["A"].append(arm_metric_row(sample_key=sample["sample_key"],target=target,baseline_items=baseline,candidate_items=baseline,target_group=group,graph_covered=bool(last in transition_sources)))
        arms["B"].append(arm_metric_row(sample_key=sample["sample_key"],target=target,baseline_items=baseline,candidate_items=bitems,target_group=group,graph_covered=bool(last in transition_sources)))
        arms["C"].append(arm_metric_row(sample_key=sample["sample_key"],target=target,baseline_items=baseline,candidate_items=citems,target_group=group,graph_covered=bool(last in transition_sources)))
        if index%16==0: print(f"TIPA_P0_DECODE dataset={dataset} users={index}/{len(cached)}",flush=True)
    elapsed=time.perf_counter()-start; audit=audit_arm_rows(arms,("A","B","C"),len(cal_samples))
    arm_rows=[{"arm":arm,**row} for arm,values in arms.items() for row in values]; write_csv(output/"per_user_arms.csv",arm_rows); write_csv(output/"per_user.csv",rows)
    torch.save(adapter.state_dict(),output/"adapter.pt")
    exclusive=[row for row in rows if row["teacher_target_rank"]<=50 and row["A_rank"]==""]
    methods={arm:summarize_rows(values) for arm,values in arms.items()}; b_agree=float(np.mean([row["B_kendall"] for row in rows])); c_agree=float(np.mean([row["C_kendall"] for row in rows]))
    gate={"kendall_delta":c_agree-b_agree,"teacher_exclusive_users":len(exclusive),"B_realized_exclusive":sum(row["B_rank"]!="" for row in exclusive),
          "C_realized_exclusive":sum(row["C_rank"]!="" for row in exclusive),"C_recall10_delta":methods["C"]["overall"]["absolute_delta_Recall@10"],
          "C_ndcg10_delta":methods["C"]["overall"]["absolute_delta_NDCG@10"],"C_broad_harm":methods["C"]["overall"]["mean_broad_harm"],
          "C_tail_recall50_delta":methods["C"]["tail"]["absolute_delta_Recall@50"]}
    gate["passed"]=bool(gate["kendall_delta"]>=.10 and gate["C_realized_exclusive"]>=5 and gate["C_realized_exclusive"]>=gate["B_realized_exclusive"] and
                         not(gate["C_recall10_delta"]<0 and gate["C_ndcg10_delta"]<0) and (gate["C_recall10_delta"]>0 or gate["C_ndcg10_delta"]>0) and
                         gate["C_broad_harm"]<=.01 and gate["C_tail_recall50_delta"]>=-.005)
    result={"experiment_id":config["experiment_id"],"dataset":dataset,"status":"PASS","methods":methods,"mechanism":{"B_kendall":b_agree,"C_kendall":c_agree,**gate},
            "training":{"prefix_records":len(records),"prefix_availability":prefix_audit,"initial_identity_max_abs_delta":initial_max,"history":training},"audit":audit,
            "integrity":{"alpha_zero_identity_exact":identity_exact,"teacher_mass_max_error":mass_error,"fit_calibration_user_overlap":len(fit_users&calibration_users),
                         "gram_optimizer_steps":0,"teacher_optimizer_steps":0,"test_read":False,"sports_read":False,"external_development_read":False},
            "timing":{"decode_seconds":elapsed,"users_per_second":len(rows)/elapsed},"peak_allocated_mib":torch.cuda.max_memory_allocated(device)/1024**2,
            "peak_reserved_mib":torch.cuda.max_memory_reserved(device)/1024**2,"lineage":{"parent_checkpoint_sha256_before":expected_parent,"parent_checkpoint_sha256_after":sha256(parent_checkpoint),
                         "teacher_checkpoint_sha256":sha256(teacher_checkpoint),"user_sequence_sha256":sha256(sequence_path),"item_index_sha256":sha256(catalog_path)}}
    write_json(output/"summary.json",result); print(json.dumps({"dataset":dataset,"gate":gate},ensure_ascii=False),flush=True); return result


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--dataset",choices=("Toys","Beauty"),required=True); args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get("execution_enabled") is not True or config.get("decision_status")!="PREREGISTERED_FROZEN_READY_TO_RUN": raise ValueError("TIPA config not frozen/enabled")
    if not torch.cuda.is_available(): raise RuntimeError("TIPA-P0 requires CUDA")
    run_domain(args.dataset,config,args.output_root); return 0


if __name__ == "__main__":
    raise SystemExit(main())
