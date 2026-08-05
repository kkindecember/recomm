#!/usr/bin/env python3
"""Phase 6 GACR-v8: path-aware pointwise and listwise residual calibration."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gacr_p0 import relative_gain
from experiment.phase4.gacr_s0 import (BoundedResidualRanker, base_scores, finite_catalog_zscore,
    stable_ranking, target_free_union)
from experiment.phase4.gcdh_p0 import (ROOT, build_train_samples, collate, prepare, read_users,
    sha256, stable_sha, write_json)
from experiment.phase5.cet_rank_r0g import candidate_labels
from experiment.phase6.gacr_v2 import (build_validation_records, paired_bootstrap_candidate,
    rank_metrics, serializable_rows, validate_checkpoint_lineage)
from experiment.phase6.gacr_v6 import build_full_training_records
from experiment.phase6.gacr_v7 import assess_calibration_noninferiority, metric_aligned_pairwise_loss
from utils import generation_trie as gt
from experiment.phase4.gcdh_p0 import normalized_sequence


class ListwiseResidualRanker(nn.Module):
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 16, bound: float = 0.2):
        super().__init__()
        self.bound = float(bound)
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 2, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, hidden_dim))
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input(features).unsqueeze(0)
        attention, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attention
        x = x + self.ffn(self.norm2(x))
        return self.bound * torch.tanh(self.output(x).squeeze(0).squeeze(-1))


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return ((values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)).clamp(-10, 10)


def to_cpu_v8_record(record: dict) -> dict:
    """Detach the v8-specific feature interface from the frozen GRAM device."""
    copied = dict(record)
    for key in ("base", "features6", "features10"):
        value = record.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"v8 record missing tensor field: {key}")
        copied[key] = value.detach().cpu()
    return copied


@torch.no_grad()
def build_path_record(sample: dict, prepared: dict, config: dict, device: torch.device) -> dict:
    """Build the frozen union and true teacher-forced path features for one sample."""
    model = prepared["model"]
    probe = dict(sample)
    probe["output"] = prepared["item2lexid"][prepared["catalog"][0]]
    batch = collate(prepared["collator"], [probe])
    input_ids, attention = batch["item_text_ids"].to(device), batch["item_text_masks"].to(device)
    if "_v8_trie" not in prepared:
        prepared["_v8_trie"] = gt.Trie(prepared["encoded_candidates"])
        prepared["_v8_max_length"] = max(len(row) for row in prepared["encoded_candidates"])
    prediction = model.backbone.generate(input_ids=input_ids, attention_mask=attention,
        max_length=prepared["_v8_max_length"], prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(prepared["_v8_trie"]),
        num_beams=int(config["generator_top_k"]), num_return_sequences=int(config["generator_top_k"]),
        return_dict_in_generate=True, output_scores=True, length_penalty=1.0)
    gram = [prepared["sequence_to_item"].get(normalized_sequence(row.tolist())) for row in prediction["sequences"]]
    if any(item is None for item in gram) or len(set(gram)) != len(gram):
        raise ValueError("invalid v8 generator candidate mapping")
    model.backbone.encoder.n_passages = input_ids.size(1)
    flat_ids, flat_attention = input_ids.view(1, -1), attention.view(1, -1)
    hidden = model.backbone.encoder(input_ids=flat_ids, attention_mask=flat_attention, return_dict=True)[0]
    pooled = model.pool_coarse(hidden, attention, input_ids.shape[-1])[0]
    logits = model.catalog_head(pooled)
    for item in sample["history_items"]:
        if item in model.item_to_index: logits[model.item_to_index[item]] = -torch.inf
    catalog_items = [prepared["catalog"][i] for i in torch.topk(logits, k=int(config["catalog_top_k"])).indices.tolist()]
    union = target_free_union(gram, catalog_items)
    labels = candidate_labels(union, prepared, device)
    count = len(union)
    output = model.backbone(input_ids=None, attention_mask=flat_attention.expand(count, -1),
        encoder_outputs=(hidden.expand(count, -1, -1),), labels=labels, return_dict=True)
    log_probs = torch.log_softmax(output.logits.float(), dim=-1)
    valid = labels.ne(-100)
    gathered = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    path_logp = (gathered * valid).sum(1) / valid.sum(1).clamp_min(1)
    item_to_sequence = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    margins, entropies = [], []
    for row, item in enumerate(union):
        sequence = item_to_sequence[item]
        row_margins, row_entropy = [], []
        for depth in range(1, len(sequence)):
            allowed = prepared["_v8_trie"].get(sequence[:depth])
            if not allowed: raise ValueError("empty legal trie child set")
            legal = torch.log_softmax(log_probs[row, depth - 1, allowed], dim=0)
            pos = allowed.index(sequence[depth])
            other = torch.cat((legal[:pos], legal[pos + 1:]))
            row_margins.append(legal[pos] - other.max() if other.numel() else torch.zeros((), device=device))
            p = legal.exp(); row_entropy.append(-(p * legal).sum())
        margins.append(torch.stack(row_margins).min()); entropies.append(torch.stack(row_entropy).mean())
    path = torch.stack((path_logp, torch.stack(margins), torch.stack(entropies)))
    if not torch.isfinite(path).all(): raise ValueError("non-finite v8 path feature")
    gram_rank = {item: index + 1 for index, item in enumerate(gram)}
    catalog_rank = {item: index + 1 for index, item in enumerate(catalog_items)}
    sequence_scores = prediction.get("sequences_scores")
    if sequence_scores is None or sequence_scores.numel() != len(gram): raise ValueError("missing beam sequence scores")
    best = sequence_scores.max()
    beam_gap = torch.tensor([float(sequence_scores[gram_rank[item] - 1] - best) if item in gram_rank else 0.0 for item in union], device=device)
    finite_logits = logits[torch.isfinite(logits)]; mean, std = finite_logits.mean(), finite_logits.std(unbiased=False).clamp_min(1e-6)
    pooled_norm = torch.linalg.vector_norm(pooled).clamp_min(1e-6)
    legacy = []
    for item in union:
        idx = model.item_to_index[item]; weight = model.catalog_head.weight[idx]
        cosine = torch.dot(pooled, weight) / (pooled_norm * torch.linalg.vector_norm(weight).clamp_min(1e-6))
        legacy.append(torch.stack((finite_catalog_zscore(logits[idx], mean, std),
            torch.tensor(1 / gram_rank[item] if item in gram_rank else 0., device=device),
            torch.tensor(1 / catalog_rank[item] if item in catalog_rank else 0., device=device),
            torch.tensor(float(item in gram_rank), device=device), torch.tensor(float(item in catalog_rank), device=device), cosine)))
    features6 = torch.stack(legacy)
    features10 = torch.cat((features6, _zscore(path[0]).unsqueeze(1), _zscore(path[1]).unsqueeze(1),
        _zscore(path[2]).unsqueeze(1), _zscore(beam_gap).unsqueeze(1)), 1)
    target = sample["positive_item"]
    return {"sample_key": sample["sample_key"], "target_group": "head" if target in prepared["heads"] else "tail",
        "target_index": union.index(target) if target in union else None, "gram_rank": gram_rank.get(target),
        "base": base_scores(union, gram).to(device).detach(), "features6": features6.detach(), "features10": features10.detach()}


def build_records(dataset, config, p0_config, device, validation=False):
    if validation:
        prepared = prepare(dataset, p0_config, device); checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"
        before = sha256(checkpoint); prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True); prepared["model"].eval()
        train = read_users(ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt")
        from experiment.phase4.gacr_p0 import select_fresh_validation_users
        from experiment.phase4.gcdh_p0 import build_validation_samples
        excluded = train | read_users(ROOT / config["inputs"]["split_root"] / dataset / "validation_users.txt")
        prior=set()
        for salt in config["prior_validation_salts"]:
            users=set(select_fresh_validation_users(set(prepared["sequences"]), excluded, dataset, salt, int(config["validation_users_per_dataset"]))); prior |= users; excluded |= users
        users=select_fresh_validation_users(set(prepared["sequences"]), excluded, dataset, config["validation_salt"], int(config["validation_users_per_dataset"]))
        samples=build_validation_samples(prepared["sequences"], set(users), prepared["item2input"], prepared["item2lexid"])
        records=[to_cpu_v8_record(build_path_record(s, prepared, config, device)) for s in samples]
        meta={"users":len(records),"validation_user_sha256":stable_sha(set(users)),"gcdh_or_training_overlap":len(set(users)&train),"prior_gacr_p0_overlap":len(set(users)&prior),"parent_checkpoint_sha256_before":before,"parent_checkpoint_sha256_after":sha256(checkpoint)}
        del prepared; torch.cuda.empty_cache(); return meta, records
    prepared_meta, fit_samples, calibration_samples = build_full_training_records(dataset, config, p0_config, device)
    # Rebuild only samples needed for path scores; full-record helper is used solely to retain its frozen split metadata.
    del fit_samples, calibration_samples; torch.cuda.empty_cache()
    prepared = prepare(dataset, p0_config, device); checkpoint=ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1/model.pt"; prepared["model"].load_state_dict(torch.load(checkpoint,map_location=device),strict=True); prepared["model"].eval()
    from experiment.phase4.gacr_p0 import split_training_users
    from experiment.phase4.gacr_s0 import select_stratified_samples
    train=read_users(ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"); fit_users, cal_users=split_training_users(train,int(config["cohort_seed"]),dataset)
    fit=build_train_samples(prepared["sequences"],fit_users,prepared["item2input"],prepared["item2lexid"]); pool=build_train_samples(prepared["sequences"],cal_users,prepared["item2input"],prepared["item2lexid"])
    cal=select_stratified_samples(pool,prepared["heads"],int(config["cohort_seed"]),f"{dataset}|gacr-p0-calibration",128,128)
    records=[to_cpu_v8_record(build_path_record(s,prepared,config,device)) for s in fit]; calibration=[to_cpu_v8_record(build_path_record(s,prepared,config,device)) for s in cal]
    meta=prepared_meta | {"path_fit_records":len(records),"path_calibration_records":len(calibration)}; del prepared; torch.cuda.empty_cache(); return meta, records, calibration


def make_model(arm, device):
    return (ListwiseResidualRanker() if arm == "E" else BoundedResidualRanker(10 if arm == "D" else 6, 16, .2)).to(device)
def features(record, arm, device): return record["features10" if arm in ("D","E") else "features6"].to(device)
def score(model, record, arm, device): return record["base"].to(device) + model(features(record, arm, device))

def train(arm, fit, calibration, seed, device):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model=make_model(arm,device)
    identity=float(np.mean([stable_ranking(r["base"]) == stable_ranking(r["base"] + model(features(r,arm,device)).detach().cpu()) for r in fit+calibration]))
    groups={g:[r for r in fit if r["target_index"] is not None and r["target_group"]==g] for g in ("head","tail")}
    if any(not rows for rows in groups.values()): raise ValueError("empty v8 group")
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01); first=last=grad=None
    for step in range(30):
        opt.zero_grad(set_to_none=True); total=0.
        for g,rows in groups.items():
            for r in rows:
                value=metric_aligned_pairwise_loss(r["base"].to(device),model(features(r,arm,device)),int(r["target_index"]))
                if value is not None: (value/(2*len(rows))).backward(); total += float(value.detach())/(2*len(rows))
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.)); opt.step(); first=total if first is None else first; last=total
        print(f"GACR_V8_TRAIN arm={arm} seed={seed} step={step+1}/30 loss={total:.6f}",flush=True)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; groups_eval,rows=evaluate(calibration,state,arm,device)
    return {"state":state,"optimizer_steps":30,"first_loss":first,"last_loss":last,"last_gradient_norm":grad,"zero_residual_identity_rate":identity,"finite_checkpoint":all(torch.isfinite(v).all() for v in state.values()),"calibration":groups_eval},rows

@torch.no_grad()
def evaluate(records,state,arm,device):
    model=make_model(arm,device); model.load_state_dict(state); model.eval(); rows=[]
    for r in records:
        candidate_rank=None if r["target_index"] is None else stable_ranking(score(model,r,arm,device)).index(int(r["target_index"]))+1
        b10,bn,b50=rank_metrics(r["gram_rank"]); c10,cn,c50=rank_metrics(candidate_rank)
        rows.append({"sample_key":r["sample_key"],"target_group":r["target_group"],"baseline_rank":r["gram_rank"],"candidate_rank":candidate_rank,"union_covered":int(r["target_index"] is not None),"baseline_Recall@10":b10,"baseline_NDCG@10":bn,"baseline_Recall@50":b50,"candidate_Recall@10":c10,"candidate_NDCG@10":cn,"candidate_Recall@50":c50,"changed":int(candidate_rank!=r["gram_rank"]),"broad_harm":int(b10==1 and c10==0)})
    def summary(rows):
        covered=[r for r in rows if r["union_covered"]]
        return {"n":len(rows),**{f"{side}_{metric}":float(np.mean([r[f"{side}_{metric}"] for r in rows])) for side in ("baseline","candidate") for metric in ("Recall@10","NDCG@10","Recall@50")},"union_coverage":float(np.mean([r["union_covered"] for r in rows])),"changed_user_coverage":float(np.mean([r["changed"] for r in rows])),"changed_covered_user_coverage":float(np.mean([r["changed"] for r in covered])) if covered else 0.,"broad_harm_rate":float(np.mean([r["broad_harm"] for r in rows]))}
    return {"overall":summary(rows),"head":summary([r for r in rows if r["target_group"]=="head"]),"tail":summary([r for r in rows if r["target_group"]=="tail"])},rows


def integrity_is_valid(integrity: dict) -> bool:
    """Interpret negative read-status fields as required non-events, not failures."""
    return (integrity["all_fit_records_used"] and integrity["fit_calibration_user_disjoint"]
        and integrity["parent_checkpoint_sha_unchanged_during_training"]
        and integrity["backbone_optimizer_steps"] == 0
        and not integrity["test_data_read"] and not integrity["sports_data_read"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--validation-only",action="store_true"); p.add_argument("--checkpoint-root",type=Path); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("GACR-v8 requires CUDA")
    config=json.loads(a.config.read_text()); validate_checkpoint_lineage(config); p0=json.loads((ROOT/config["inputs"]["p0_config"]).read_text()); device=torch.device("cuda:0"); training={}; eligible={"D":[],"E":[]}
    if a.validation_only:
        if a.checkpoint_root is None: raise ValueError("--validation-only requires --checkpoint-root")
        recovery=config["validation_recovery"]
        if sha256(a.checkpoint_root/"summary.json") != recovery["parent_summary_sha256"]: raise ValueError("frozen parent summary mismatch")
        parent=json.loads((a.checkpoint_root/"summary.json").read_text()); training=parent["training"]; integrity=parent["integrity"]
        if not integrity_is_valid(integrity): raise ValueError("parent training integrity is not valid")
        for dataset in config["datasets"]:
            for seed in config["training_seeds"]:
                result=training[dataset]["arms"]["E"][str(seed)]
                path=a.checkpoint_root/dataset/f"E_seed{seed}.pt"
                if sha256(path) != result["residual_checkpoint_sha256"] or sha256(path) != recovery["expected_E_checkpoint_sha256"][dataset][str(seed)]: raise ValueError(f"frozen E checkpoint mismatch: {path}")
                eligible["E"].append(result["finite_checkpoint"] and result["calibration_noninferiority"]["eligible"])
    else:
        for dataset in config["datasets"]:
            meta,fit,cal=build_records(dataset,config,p0,device); arms={}
            for arm in ("C","D","E"):
                seeds={}
                for seed in config["training_seeds"]:
                    result,_=train(arm,fit,cal,int(seed),device); path=a.output_root/dataset/f"{arm}_seed{seed}.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save(result.pop("state"),path); result["residual_checkpoint_sha256"]=sha256(path); result["calibration_noninferiority"]=assess_calibration_noninferiority(result["calibration"],config); seeds[str(seed)]=result
                    if arm in eligible: eligible[arm].append(result["finite_checkpoint"] and result["calibration_noninferiority"]["eligible"])
                arms[arm]=seeds
            training[dataset]=meta|{"arms":arms}; del fit,cal; torch.cuda.empty_cache()
        integrity={"all_fit_records_used":True,"fit_calibration_user_disjoint":all(training[d]["fit_calibration_user_overlap"]==0 for d in config["datasets"]),"parent_checkpoint_sha_unchanged_during_training":all(training[d]["parent_checkpoint_sha256_before"]==training[d]["parent_checkpoint_sha256_after"] for d in config["datasets"]),"backbone_optimizer_steps":0,"test_data_read":False,"sports_data_read":False}
    qualified={arm:all(values) and integrity_is_valid(integrity) for arm,values in eligible.items()}
    if not any(qualified.values()): write_json(a.output_root/"summary.json",{"experiment_id":config["experiment_id"],"result_status":"STOPPED_BEFORE_FRESH_VALIDATION_CALIBRATION_GATE_FAILED","qualified_arms":qualified,"training":training,"validation":{},"integrity":integrity}); return 0
    validation={}
    for dataset in config["datasets"]:
        meta,records=build_records(dataset,config,p0,device,validation=True); seeds={}
        for seed in config["training_seeds"]:
            cell={}
            for arm in ("E",) if a.validation_only else ("D","E"):
                if not qualified[arm]: continue
                checkpoint_root=a.checkpoint_root if a.validation_only else a.output_root
                state=torch.load(checkpoint_root/dataset/f"{arm}_seed{seed}.pt",map_location="cpu"); groups,rows=evaluate(records,state,arm,device); path=a.output_root/dataset/f"{arm}_seed{seed}_per_user.csv"; serial=serializable_rows(rows)
                with path.open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(serial[0]));w.writeheader();w.writerows(serial)
                cell[arm]={"groups":groups,"gains":{"overall_ndcg10_relative_gain":relative_gain(groups["overall"]["baseline_NDCG@10"],groups["overall"]["candidate_NDCG@10"])},"bootstrap":{"overall_ndcg10_relative_gain_ci95":paired_bootstrap_candidate(rows,"NDCG@10",True,int(seed)+101)},"per_user_sha256":sha256(path)}
            seeds[str(seed)]=cell
        validation[dataset]=meta|{"seeds":seeds}; del records;torch.cuda.empty_cache()
    write_json(a.output_root/"summary.json",{"experiment_id":config["experiment_id"],"result_status":"RESULTS_READY_FOR_RESEARCHER_ANALYSIS","qualified_arms":qualified,"training":training,"validation":validation,"integrity":integrity})

if __name__=="__main__": main()
