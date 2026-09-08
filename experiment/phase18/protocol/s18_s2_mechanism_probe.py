#!/usr/bin/env python3
"""S18-2: immutable bounded four-arm probe, with a separate background supervisor."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase18.core.contracts import ROOT, load_json, metrics_from_ranks, sha256
from experiment.phase18.core.s2_contracts import (
    ARMS, choose_alpha, mechanism_gate, prefix_survival, shuffled_teacher_map, training_view,
)
from experiment.phase18.protocol.s18_s1_prepare import atomic_json, atomic_text, utc_now

CONFIG = ROOT / "experiment/phase18/config/s18_s2_mechanism_probe.json"
BASE = ROOT / "artifacts/phase18/s2_mechanism_probe"
STATUS = ROOT / "artifacts/phase18/status/s18_s2_mechanism_probe.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-2.attempts.jsonl"
REPORT = ROOT / "report/第十八阶段/Stage18_S2_机制探针执行报告.md"
PYTHON = "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python"
MODULE = "experiment.phase18.protocol.s18_s2_mechanism_probe"
SEALED = {"i1_i2_read": False, "d1_read": False, "d2_read": False,
          "official_validation_test_read": False, "sports_read": False,
          "automatic_retry": False, "automatic_s18_3": False}


def record(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def checked(ref):
    path = ROOT / ref["path"]
    if sha256(path) != ref["sha256"]:
        raise RuntimeError(f"frozen input hash mismatch: {path}")
    return path


def verify_inputs(config):
    for ref in config["prerequisites"].values():
        checked(ref)
    predecessor = load_json(checked(config["prerequisites"]["s1r_confirmation"]))
    if predecessor["decision"] != config["required_predecessor_decision"]:
        raise RuntimeError("S18-1R has not passed")
    if config["cohort"]["fold"] != "I0" or list(config["domains"]) != ["Toys", "Beauty"]:
        raise RuntimeError("only two-domain I0 is authorized")
    for domain in config["domains"].values():
        for name in ("parent", "item_head", "source_cohort"):
            checked(domain[name])
        for name, info in domain["dataset"]["files"].items():
            checked({"path": f'{domain["dataset"]["path"]}/{name}', "sha256": info["sha256"]})


def prepare():
    config = load_json(CONFIG)
    verify_inputs(config)
    out = BASE / "preflight"
    if out.exists():
        old = load_json(out / "manifest.json")
        if old["config_sha256"] != sha256(CONFIG):
            raise RuntimeError("preflight config changed; cannot overwrite")
        return old
    out.mkdir(parents=True)
    domains = {}
    for domain, info in config["domains"].items():
        users = checked(info["source_cohort"]).read_text().splitlines()[:config["cohort"]["probe_users"]]
        cohort_path = out / f"cohort_{domain}.txt"
        atomic_text(cohort_path, "\n".join(users) + "\n")
        if sha256(cohort_path) != info["probe_cohort_sha256"]:
            raise RuntimeError("probe cohort hash mismatch")
        # This is the S18-1 materialized I0 view, not any official split.
        sequences = {}
        with (ROOT / info["dataset"]["path"] / "user_sequence.txt").open() as f:
            for line in f:
                fields = line.split()
                if fields[0] in users:
                    sequences[fields[0]] = fields[1:]
        train_file = out / f"training_{domain}.jsonl"
        eval_file = out / f"evaluation_{domain}.jsonl"
        for user in users:
            seq = sequences[user]
            history, target = training_view(seq)
            append(train_file, {"user": user, "history": list(history), "target": target})
            append(eval_file, {"user": user, "history": seq[:-2], "target": seq[-2]})
        shuffle = shuffled_teacher_map(users, domain, config["seed"])
        shuffle_file = out / f"shuffled_teacher_{domain}.json"
        atomic_json(shuffle_file, shuffle)
        domains[domain] = {"cohort": record(cohort_path), "training": record(train_file),
                           "evaluation": record(eval_file), "shuffle": record(shuffle_file),
                           "users": len(users), "training_target_in_i0_visible": True}
    manifest = {"schema_version": "phase18.s18_2_preflight.v1", "status": "ENGINEERING_PREFLIGHT_PASS",
                "config_sha256": sha256(CONFIG), "created_at": utc_now(), "domains": domains, **SEALED}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def source_paths():
    paths = [CONFIG, Path(__file__), ROOT / "experiment/phase18/core/pcps_loss.py",
             ROOT / "experiment/phase18/core/s2_contracts.py",
             ROOT / "experiment/phase18/tests/test_s18_s2_contract.py",
             ROOT / "experiment/phase18/protocol/s18_s1_runtime.py",
             ROOT / "experiment/phase18/core/contracts.py",
             ROOT / "experiment/phase18/protocol/s18_s1_prepare.py",
             ROOT / "GRAM/prompt.txt",
             ROOT / "experiment/phase9/train_cf0_b2_item_head.py",
             ROOT / "plan/第十八阶段/GRAM_第十八阶段_S18-2机制探针执行补遗v0.1.md"]
    for directory in ("GRAM/src/model", "GRAM/src/data", "GRAM/src/processor", "GRAM/src/utils"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    return paths


def progress(out, phase, **fields):
    payload = {"phase": phase, "pid": os.getpid(), "updated_at": utc_now(), **SEALED, **fields}
    atomic_json(out / "progress.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


class Context:
    def __init__(self, domain, config):
        # Import the local pinned GRAM environment only in the GPU worker.
        from experiment.phase18.protocol import s18_s1_runtime as rt
        from experiment.phase18.core import pcps_loss as loss
        self.rt, self.loss, self.domain, self.config = rt, loss, domain, config
        self.torch, self.np = rt.torch, rt.np
        self.device = rt.torch.device("cuda:0")
        self.s1 = load_json(checked(config["prerequisites"]["s1_config"]))
        self.tokenizer = rt.AutoTokenizer.from_pretrained(self.s1["backbone"]["snapshot"], local_files_only=True)
        self.args = rt.gram_args(self.s1, domain, "I0")
        self.args.tokenizer = self.tokenizer
        self.dataset = rt.TestDatasetGRAM(self.args, rt.dataset_name_from_manifest(domain, "I0"),
                                         "sequential", None, self.tokenizer, mode="validation")
        self.collator = rt.CollatorGRAM(self.tokenizer, self.args, mode="train")
        self.item_paths = {raw: rt.identifier_tokens(self.tokenizer, lex)
                           for raw, lex in self.dataset.item2lexid.items()}
        self.path_item = {path: raw for raw, path in self.item_paths.items()}
        if len(self.path_item) != len(self.item_paths):
            raise RuntimeError("lexical path collision")
        self.trie = rt.gt.Trie([[0, *p] for p in self.item_paths.values()])
        self.max_length = max(map(len, self.item_paths.values())) + 1
        info = config["domains"][domain]
        index_name = f'item_generative_indexing_{self.s1["domains"][domain]["hierarchy"]}.txt'
        self.item_index, _, frequencies, _ = rt.read_numeric_fold_data(ROOT / info["dataset"]["path"], index_name)
        self.freq = rt.np.asarray([frequencies.get(i, 0) for i in range(1, len(self.item_index) + 1)], dtype=rt.np.float64)
        self.q1 = info["q1"]
        manifest = load_json(BASE / "preflight/manifest.json")["domains"][domain]
        self.users = checked(manifest["cohort"]).read_text().splitlines()
        self.train = self.read_rows(checked(manifest["training"]))
        self.evaluation = self.read_rows(checked(manifest["evaluation"]))
        self.shuffle = load_json(checked(manifest["shuffle"]))
        self.teacher = None

    @staticmethod
    def read_rows(path):
        with path.open() as f:
            return {r["user"]: r for r in (json.loads(line) for line in f)}

    def batch(self, row):
        history = row["history"][-self.s1["parent_training"]["max_history"]:]
        # Keep the native TestDatasetGRAM passage/history order exactly.
        sample = {"user_id": row["user"], "output": self.dataset.item2lexid[row["target"]],
                  "input": ["What would user purchase after " + self.args.his_sep.join(
                      self.dataset.item2lexid[i] for i in history[::-1]) + " ?"] +
                      [self.dataset.item2input[i] for i in history[::-1]],
                  "history_item_ids": [self.item_index[i] for i in history[::-1]],
                  "target_item_id": self.item_index[row["target"]]}
        batch = self.collator([sample])
        actual = tuple(int(i) for i in batch["target_ids"][0] if i != -100)
        if actual != self.item_paths[row["target"]]:
            raise RuntimeError("teacher forcing labels differ from legal lexical path")
        return batch

    def parent(self):
        model = self.rt.load_parent(self.s1, checked(self.config["domains"][self.domain]["parent"]), self.device)
        self.rt.set_cross_attention_cache(model, self.config["generation"]["cross_attention_cache"])
        return model

    def load_teacher(self):
        if self.teacher is None:
            checkpoint = self.torch.load(checked(self.config["domains"][self.domain]["item_head"]), map_location="cpu")
            mc = dict(checkpoint["model_config"])
            mc["temperature"] = mc.pop("temperature_initial")
            self.teacher = self.rt.CF0B2ItemHead(**mc).to(self.device)
            self.teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.teacher.eval()

    def cf(self, row):
        self.load_teacher()
        values = self.rt.item_head_scores(self.teacher, [self.item_index[i] for i in row["history"]], 20, self.device)
        if not self.np.isfinite(values).all():
            raise FloatingPointError("nonfinite CF teacher")
        return values

    def generate(self, model, row, width):
        model.eval()
        output, paths, raw, normalized, active = self.rt.generate_one(
            model, self.batch(row), self.trie, self.max_length, width, self.device)
        items = [self.path_item.get(p) for p in paths]
        if None in items or len(set(items)) != width or not self.np.isfinite(raw).all() or not self.np.isfinite(normalized).all():
            raise RuntimeError("illegal, duplicate or nonfinite generated beam")
        del output
        return {"items": items, "scores": normalized.tolist(), "active": active}

    def forward(self, model, row, arm="C0_CONT", alpha=0.0, cache=None):
        batch = self.batch(row)
        output = model(input_ids=batch["item_text_ids"].to(self.device),
                       attention_mask=batch["item_text_masks"].to(self.device),
                       history_item_ids=batch["history_item_ids"].to(self.device),
                       history_item_mask=batch["history_item_mask"].to(self.device),
                       labels=batch["target_ids"].to(self.device), use_cache=False, return_dict=True)
        if alpha == 0 or arm == "C0_CONT":
            total, components = self.loss.combine_loss(output.loss, alpha, arm=arm)
            return total, components, output.logits, None
        target = self.item_paths[row["target"]]
        positions = self.torch.arange(len(target), device=self.device)
        tokens = self.torch.tensor(target, device=self.device)
        target_score = output.logits[0].float().log_softmax(-1)[positions, tokens].sum() / len(target)
        paths = [self.item_paths[item] for item in cache["path_items"]]
        scores = [target_score]
        if paths:
            labels = self.torch.full((len(paths), max(map(len, paths))), -100, dtype=self.torch.long, device=self.device)
            for i, path in enumerate(paths):
                labels[i, :len(path)] = self.torch.tensor(path, device=self.device)
            hidden = output.encoder_last_hidden_state
            negatives = model(encoder_outputs=self.rt.BaseModelOutput(last_hidden_state=hidden.repeat(len(paths), 1, 1)),
                              attention_mask=batch["item_text_masks"].to(self.device).reshape(1, -1).repeat(len(paths), 1),
                              labels=labels, use_cache=False, return_dict=True)
            logp = negatives.logits.float().log_softmax(-1)
            valid = labels.ne(-100)
            selected = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            scores.extend((selected * valid).sum(-1) / valid.sum(-1))
        scores = self.torch.stack(scores)
        total, components = self.loss.combine_loss(output.loss, alpha, output.logits[0], scores, cache, arm)
        if not bool(self.torch.isfinite(total)) or not bool(self.torch.isfinite(scores).all()):
            raise FloatingPointError("nonfinite PCPS objective")
        return total, components, output.logits, scores


def make_cache(ctx, model, row, *, training):
    beam50 = ctx.generate(model, row, 50)
    beam200 = ctx.generate(model, row, 200)
    wrong = [item for item in beam200["items"] if item != row["target"]][:ctx.config["loss"]["path_negative_cap"]]
    forced = ctx.rt.teacher_force_paths(model, ctx.batch(row),
                                       [ctx.item_paths[i] for i in [row["target"], *wrong]], ctx.device)
    if not all(r["finite"] for r in forced):
        raise FloatingPointError("nonfinite teacher-forced parent")
    parent_scores = [r["normalized_log_probability"] for r in forced[1:]]
    cf = ctx.cf(row)
    positions50 = [ctx.item_index[i] - 1 for i in beam50["items"]]
    _, reliability = ctx.loss.pcrf_scores(beam50["scores"], cf[positions50], ctx.freq[positions50], ctx.q1)
    result = {"user": row["user"], "target": row["target"], "path_items": wrong,
              "parent_path_scores": parent_scores,
              "parent_weights": ctx.loss.path_weights(parent_scores), "reliability": reliability,
              "parent_prefix_survival": prefix_survival(beam50["active"], ctx.item_paths[row["target"]]),
              "beam50_items": beam50["items"], "beam50_scores": beam50["scores"],
              "beam200_items": beam200["items"], "beam200_scores": beam200["scores"], "arms": {}}
    if training:
        shuffled_cf = ctx.cf(ctx.train[ctx.shuffle[row["user"]]])
        for arm in ARMS[1:]:
            values = None if arm == "A0_LEGAL_GENERIC" else (cf if arm == "M0_PCPS" else shuffled_cf)
            nodes = ctx.loss.mine_prefix_nodes(ctx.item_paths[row["target"]], ctx.item_paths,
                        beam200["items"], beam200["scores"], None if values is None else (values, reliability),
                        None if values is None else ctx.freq, ctx.item_index, ctx.config["loss"]["negative_k_per_node"])
            if values is None:
                weights = result["parent_weights"]
            else:
                corrected = ctx.loss.zscore(ctx.loss.zscore(values) - 0.5 * ctx.loss.zscore(ctx.np.log1p(ctx.freq)))
                weights = ctx.loss.path_weights(parent_scores, corrected[ctx.item_index[row["target"]] - 1],
                           [corrected[ctx.item_index[i] - 1] for i in wrong], reliability)
            result["arms"][arm] = {"nodes": nodes, "path_items": wrong, "weights": weights}
    return result


def cache_users(ctx, model, users, out, existing=None):
    cache = {} if existing is None else existing
    model.eval()
    started = time.monotonic()
    for ordinal, user in enumerate(users, 1):
        if user in cache:
            continue
        row = make_cache(ctx, model, ctx.train[user], training=True)
        append(out / "training_cache.jsonl", row)
        cache[user] = row
        if ordinal == 1 or ordinal % 10 == 0:
            progress(out.parent, "TRAIN_NEGATIVE_CACHE", domain=ctx.domain, users_done=ordinal,
                     users_total=len(users), wall_seconds=time.monotonic() - started)
    return cache


def grad_norm(model):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))


def calibrate(ctx, model, cache, out):
    users = ctx.users[:ctx.config["cohort"]["calibration_users"]]
    norms = {}
    model.eval()
    started = time.monotonic()
    for component in ("ce", "auxiliary"):
        model.zero_grad(set_to_none=True)
        for index, user in enumerate(users, 1):
            _, parts, _, _ = ctx.forward(model, ctx.train[user], "M0_PCPS", 1.0, cache[user]["arms"]["M0_PCPS"])
            (parts[component] / len(users)).backward()
            if index % 10 == 0:
                progress(out.parent, "GRADIENT_CALIBRATION", domain=ctx.domain, component=component,
                         users_done=index, users_total=len(users))
        norms[component] = grad_norm(model)
    model.zero_grad(set_to_none=True)
    if not all(math.isfinite(v) and v > 0 for v in norms.values()):
        raise FloatingPointError("invalid calibration gradient norms")
    result = {"users": len(users), "norms": norms, "unweighted_ratio": norms["auxiliary"] / norms["ce"],
              "wall_seconds": time.monotonic() - started, "accuracy_used": False}
    atomic_json(out / "gradient_calibration.json", result)
    return result


def train_arm(ctx, cache, users, stage, arm, alpha, out):
    torch = ctx.torch
    ctx.rt.set_seed(ctx.config["seed"])
    model = ctx.parent()
    optcfg = ctx.config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optcfg["learning_rate"],
                                 weight_decay=optcfg["weight_decay"], eps=optcfg["adam_eps"])
    epochs = optcfg[f"{stage}_epochs"]
    accum = optcfg["gradient_accumulation_users"]
    steps = math.ceil(len(users) / accum) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: ctx.rt.scheduler_lambda(
        n, steps, int(steps * optcfg["warmup_ratio"])))
    initial_ce = train_ce(ctx, model, users) if stage == "overfit" else None
    history = []
    started = time.monotonic()
    update_count = 0
    order_digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        order = list(users)
        random.Random(ctx.config["seed"] + epoch).shuffle(order)
        model.train()
        means = {}
        optimizer.zero_grad(set_to_none=True)
        for index, user in enumerate(order):
            ctx.rt.set_seed(ctx.config["seed"] + epoch * 100000 + ctx.users.index(user))
            order_digest.update(f"{epoch}|{user}\n".encode())
            actual_alpha = 0.0 if arm == "C0_CONT" else alpha
            row_cache = None if arm == "C0_CONT" else cache[user]["arms"][arm]
            total, parts, _, _ = ctx.forward(model, ctx.train[user], arm, actual_alpha, row_cache)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("nonfinite training loss")
            group_start = (index // accum) * accum
            group_size = min(accum, len(order) - group_start)
            (total / group_size).backward()
            for key, value in parts.items():
                means[key] = means.get(key, 0.0) + float(value.detach())
            if (index + 1) % accum == 0 or index + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optcfg["max_grad_norm"])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("nonfinite optimizer gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            if (index + 1) % 25 == 0:
                progress(out.parents[2], "TRAINING", domain=ctx.domain, stage=stage, arm=arm,
                         epoch=epoch + 1, epochs=epochs, users_done=index + 1, users_total=len(order))
        history.append({"epoch": epoch + 1, **{k: v / len(order) for k, v in means.items()}})
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "final.pt"
    torch.save({"model_state_dict": model.state_dict(), "arm": arm, "stage": stage,
                "alpha": alpha, "config_sha256": sha256(CONFIG), "fixed_final_epoch": epochs}, checkpoint)
    result = {"arm": arm, "stage": stage, "history": history, "samples": len(users) * epochs,
              "optimizer_steps": update_count, "batch_order_sha256": order_digest.hexdigest(),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "initial_train_ce": initial_ce,
              "final_train_ce": train_ce(ctx, model, users) if stage == "overfit" else None,
              "wall_seconds": time.monotonic() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "checkpoint": record(checkpoint)}
    atomic_json(out / "training.json", result)
    del optimizer, scheduler
    return model, result


def train_ce(ctx, model, users):
    model.eval()
    with ctx.torch.no_grad():
        return sum(float(ctx.forward(model, ctx.train[u])[0]) for u in users) / len(users)


def evaluate(ctx, model, anchors, out):
    model.eval()
    rows = []
    path = out / "per_user.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user", "rank", "pcrf_rank", "prefix_survival",
                       "path_margin", "target", "candidates", "scores"], delimiter="\t")
        writer.writeheader()
        for index, user in enumerate(ctx.users, 1):
            row, anchor = ctx.evaluation[user], anchors[user]
            beam = ctx.generate(model, row, 50)
            cf = ctx.cf(row)
            positions = [ctx.item_index[i] - 1 for i in beam["items"]]
            joint, _ = ctx.loss.pcrf_scores(beam["scores"], cf[positions], ctx.freq[positions], ctx.q1)
            pcrf_items = [beam["items"][i] for i in sorted(range(50), key=lambda i: (-float(joint[i]), i))]
            forced = ctx.rt.teacher_force_paths(model, ctx.batch(row),
                       [ctx.item_paths[i] for i in [row["target"], *anchor["path_items"]]], ctx.device)
            if not all(r["finite"] for r in forced):
                raise FloatingPointError("nonfinite evaluation full-path score")
            margin = forced[0]["normalized_log_probability"] - sum(
                w * r["normalized_log_probability"] for w, r in zip(anchor["parent_weights"], forced[1:]))
            result = {"user": user, "target": row["target"],
                      "rank": beam["items"].index(row["target"]) + 1 if row["target"] in beam["items"] else 51,
                      "pcrf_rank": pcrf_items.index(row["target"]) + 1 if row["target"] in pcrf_items else 51,
                      "prefix_survival": prefix_survival(beam["active"], ctx.item_paths[row["target"]]),
                      "path_margin": margin, "candidates": "||".join(beam["items"]),
                      "scores": "||".join(map(str, beam["scores"]))}
            writer.writerow(result)
            rows.append(result)
            if index % 25 == 0:
                handle.flush()
                progress(out.parents[2], "PROBE_EVALUATION", domain=ctx.domain, arm=out.name,
                         users_done=index, users_total=len(ctx.users))
    result = {**metrics_from_ranks(r["rank"] for r in rows),
              "prefix_survival": sum(r["prefix_survival"] for r in rows) / len(rows),
              "path_margin": sum(r["path_margin"] for r in rows) / len(rows),
              "pcrf": metrics_from_ranks(r["pcrf_rank"] for r in rows),
              "finite_fraction": 1.0, "generated_path_legality": 1.0, "per_user": record(path)}
    atomic_json(out / "evaluation.json", result)
    return result


def scientific_worker(config, out):
    calibrations, contexts, caches = {}, {}, {}
    for domain in config["domains"]:
        progress(out, "LOAD_CONTEXT", domain=domain)
        ctx = Context(domain, config)
        contexts[domain] = ctx
        unit = out / domain
        unit.mkdir()
        model = ctx.parent().eval()
        caches[domain] = cache_users(ctx, model, ctx.users[:100], unit)
        calibrations[domain] = calibrate(ctx, model, caches[domain], unit)
        del model
        if ctx.teacher is not None:
            ctx.teacher.cpu()
            ctx.teacher = None
        ctx.torch.cuda.empty_cache()
    ratios = {d: r["unweighted_ratio"] for d, r in calibrations.items()}
    alpha = choose_alpha(ratios)
    frozen = {"alpha": alpha, "calibrations": calibrations, "config_sha256": sha256(CONFIG),
              "selection_used_accuracy": False, "frozen_at": utc_now(),
              "weighted_ratios": {d: {str(a): a * r for a in (0.1, 0.3)} for d, r in ratios.items()}}
    atomic_json(out / "frozen_alpha.json", frozen)
    if alpha is None:
        return {"decision": "GRADIENT_CALIBRATION_FAILED", "alpha_calibration": frozen,
                "treatment_training_started": False}
    progress(out, "ALPHA_FROZEN", alpha=alpha, ratios=ratios)
    domain_results = {}
    # The entire 100-user engineering stage precedes the 1k probe.
    for domain, ctx in contexts.items():
        overfit = {}
        for arm in ARMS:
            model, result = train_arm(ctx, caches[domain], ctx.users[:100], "overfit", arm, alpha,
                                      out / domain / "overfit" / arm)
            overfit[arm] = result
            del model
            ctx.torch.cuda.empty_cache()
        fields = ("parameter_count", "samples", "optimizer_steps", "batch_order_sha256")
        matched = all(len({overfit[a][key] for a in ARMS}) == 1 for key in fields)
        loss_decreased = all(r["final_train_ce"] < r["initial_train_ce"] for r in overfit.values())
        domain_results[domain] = {"overfit": overfit, "overfit_engineering_pass": matched and loss_decreased}
        atomic_json(out / domain / "overfit_summary.json", domain_results[domain])
        if not matched or not loss_decreased:
            return {"decision": "FAILED_ENGINEERING_GATE", "alpha": alpha, "domains": domain_results}
    for domain, ctx in contexts.items():
        unit = out / domain
        parent = ctx.parent().eval()
        cache_users(ctx, parent, ctx.users, unit, caches[domain])
        cache_ref = record(unit / "training_cache.jsonl")
        anchors = {}
        for index, user in enumerate(ctx.users, 1):
            # Held-out labels appear only in this evaluation-only anchor artifact.
            anchor = make_cache(ctx, parent, ctx.evaluation[user], training=False)
            anchors[user] = anchor
            append(unit / "evaluation_anchors.jsonl", anchor)
            if index % 10 == 0:
                progress(out, "EVALUATION_ANCHOR_CACHE", domain=domain, users_done=index, users_total=len(ctx.users))
        del parent
        evaluations, training = {}, {}
        for arm in ARMS:
            arm_out = unit / "probe" / arm
            model, trained = train_arm(ctx, caches[domain], ctx.users, "probe", arm, alpha, arm_out)
            training[arm] = trained
            evaluations[arm] = evaluate(ctx, model, anchors, arm_out)
            del model
            ctx.torch.cuda.empty_cache()
        fields = ("parameter_count", "samples", "optimizer_steps", "batch_order_sha256")
        if not all(len({training[a][key] for a in ARMS}) == 1 for key in fields):
            raise RuntimeError("probe arm budgets differ")
        gate = mechanism_gate(evaluations, config["gates"])
        domain_results[domain].update({"probe": evaluations, "training": training, "gate": gate,
                                      "training_cache": cache_ref,
                                      "evaluation_anchors": record(unit / "evaluation_anchors.jsonl")})
        atomic_json(unit / "summary.json", domain_results[domain])
        if gate["decision"] != "MECHANISM_PASS":
            return {"decision": gate["decision"], "alpha": alpha, "domains": domain_results,
                    "stopped_at_domain": domain, "treatment_training_started": True}
        if ctx.teacher is not None:
            ctx.teacher.cpu()
            ctx.teacher = None
    return {"decision": "MECHANISM_PASS", "alpha": alpha, "domains": domain_results,
            "treatment_training_started": True, "accuracy_success_claim": False}


def smoke_worker(config, out):
    reports = {}
    for domain in config["domains"]:
        progress(out, "SMOKE_LOAD", domain=domain)
        ctx = Context(domain, config)
        torch = ctx.torch
        ctx.rt.set_seed(config["seed"])
        model = ctx.parent().eval()
        # Longest selected visible history is an outcome-independent memory stress case.
        user = max(ctx.users, key=lambda u: (min(len(ctx.train[u]["history"]), 20), u))
        row = ctx.train[user]
        # Native prompt construction must match the established held-out dataset.
        indices = {u: i for i, u in enumerate(ctx.dataset.data["user_id"])}
        native = ctx.collator([ctx.dataset[indices[user]]])
        rebuilt = ctx.batch(ctx.evaluation[user])
        input_identity = all(torch.equal(native[k], rebuilt[k]) for k in
             ("item_text_ids", "item_text_masks", "history_item_ids", "history_item_mask", "target_ids"))
        if not input_identity:
            raise RuntimeError("probe input differs from native GRAM")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        cache = make_cache(ctx, model, row, training=True)
        cache_seconds = time.monotonic() - started
        model.eval()
        with torch.no_grad():
            loss0, _, logits0, _ = ctx.forward(model, row)
            loss_identity, _, logits_identity, _ = ctx.forward(model, row, "M0_PCPS", 0.0, object())
            loss_delta = float((loss0 - loss_identity).abs())
            logits_delta = float((logits0 - logits_identity).abs().max())
        del logits0, logits_identity
        # Same one-step continuation from the same snapshot must yield the exact beam50.
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        beams = []
        for arm in ("C0_CONT", "M0_PCPS"):
            model.load_state_dict(initial, strict=True)
            model.train()
            ctx.rt.set_seed(config["seed"])
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = ctx.forward(model, row, arm, 0.0, object())
            loss.backward()
            optimizer.step()
            beams.append(ctx.generate(model, row, 50)["items"])
            del optimizer
        model.load_state_dict(initial, strict=True)
        del initial
        model.eval()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        total, parts, logits, scores = ctx.forward(model, row, "M0_PCPS", 1.0, cache["arms"]["M0_PCPS"])
        prefix_grad = torch.autograd.grad(parts["prefix"], logits, retain_graph=True)[0]
        path_grad = torch.autograd.grad(parts["path"], scores, retain_graph=True)[0]
        direction_ok = bool(path_grad[0] < 0 and (path_grad[1:] > 0).all())
        for node in cache["arms"]["M0_PCPS"]["nodes"]:
            d = node["depth"]
            direction_ok = direction_ok and bool(prefix_grad[0, d, node["target_child"]] < 0)
            direction_ok = direction_ok and bool((prefix_grad[0, d, node["negative_children"]] > 0).all())
        total.backward()
        gradient_norm = grad_norm(model)
        torch.cuda.synchronize()
        step_seconds = time.monotonic() - started
        passed = (loss_delta <= 1e-6 and logits_delta <= 1e-6 and beams[0] == beams[1] and
                  direction_ok and gradient_norm > 0 and math.isfinite(gradient_norm) and
                  float(parts["prefix"]) > 0 and float(parts["path"]) > 0)
        reports[domain] = {"passed": passed, "input_identity": input_identity, "stress_user": user,
              "history_length": min(len(row["history"]), 20), "loss_identity_delta": loss_delta,
              "logits_identity_max_delta": logits_delta, "beam50_after_alpha0_update_exact": beams[0] == beams[1],
              "prefix_loss": float(parts["prefix"].detach()), "path_loss": float(parts["path"].detach()),
              "gradient_direction_ok": direction_ok, "gradient_norm": gradient_norm,
              "training_cache_seconds_per_user": cache_seconds, "aux_forward_backward_seconds": step_seconds,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "generation_legal": True, "scores_finite": True}
        atomic_json(out / f"{domain}.json", reports[domain])
        del model, ctx, logits, scores, parts, total, prefix_grad, path_grad
        torch.cuda.empty_cache()
        if not passed:
            return {"decision": "FAILED_ENGINEERING_GATE", "domains": reports}
    # Conservative estimate includes train and evaluation anchor generation, four arms,
    # calibration and five-epoch overfit. It is a smoke-based estimate, not a promise.
    seconds = sum(2 * 1000 * r["training_cache_seconds_per_user"] +
                  (4 * 1500 + 2 * 100) * r["aux_forward_backward_seconds"] +
                  4 * 1000 * r["training_cache_seconds_per_user"] * 0.5 for r in reports.values())
    return {"decision": "ENGINEERING_PASS", "domains": reports, "estimated_full_wall_hours": seconds / 3600,
            "scientific_result_eligible": False}


def write_report(summary):
    lines = ["# Stage18 S18-2 机制探针执行报告", "", "## Material Passport", "",
             "- Origin Skill：`academic-research-suite / experiment-agent`",
             f'- Status：`{summary["status"]}`', f'- Decision：`{summary["decision"]}`',
             f'- Config SHA256：`{summary["config_sha256"]}`',
             "- Scope：Toys/Beauty I0，四臂 matched continuation；I1/I2 与所有外部保护数据保持封存。", ""]
    if "alpha_calibration" in summary:
        lines += ["## 梯度校准", "", "| Domain | CE gradient norm | PCPS gradient norm | ratio |", "|---|---:|---:|---:|"]
        for d, r in summary["alpha_calibration"]["calibrations"].items():
            lines.append(f'| {d} | {r["norms"]["ce"]:.6g} | {r["norms"]["auxiliary"]:.6g} | {r["unweighted_ratio"]:.6g} |')
    for domain, result in summary.get("domains", {}).items():
        if "gate" in result:
            lines += ["", f'## {domain}', "", f'Gate：`{result["gate"]["decision"]}`', "",
                      "| Arm | Hit@50 | NDCG@10 | Prefix survival | Path margin |", "|---|---:|---:|---:|---:|"]
            for arm, metrics in result["probe"].items():
                lines.append(f'| {arm} | {metrics["Hit@50"]:.6f} | {metrics["NDCG@10"]:.6f} | {metrics["prefix_survival"]:.6f} | {metrics["path_margin"]:.6f} |')
    lines += ["", "本步骤仅用于工程和机制淘汰，不构成 accuracy success；不自动重试或启动 S18-3。", ""]
    atomic_text(REPORT, "\n".join(lines))


def worker(args):
    out = BASE / args.run_name
    manifest = load_json(out / "run_manifest.json")
    for ref in manifest["sources"]:
        checked(ref)
    checked(manifest["preflight"])
    config = load_json(CONFIG)
    verify_inputs(config)
    from experiment.phase18.protocol import s18_s1_runtime as rt
    rt.torch.set_num_threads(4)
    if rt.torch.cuda.device_count() != 1:
        raise RuntimeError("S18-2 requires exactly one visible GPU")
    started = time.monotonic()
    result = smoke_worker(config, out) if args.kind == "smoke" else scientific_worker(config, out)
    summary = {"schema_version": "phase18.s18_2_summary.v1", "status": "COMPLETED", **result,
               "run_name": args.run_name, "physical_gpu": args.gpu, "kind": args.kind,
               "config_sha256": sha256(CONFIG), "input_manifest_sha256": sha256(out / "run_manifest.json"),
               "wall_seconds": time.monotonic() - started, "completed_at": utc_now(), **SEALED}
    atomic_json(out / "summary.json", summary)
    if args.kind == "formal":
        atomic_json(BASE / "summary.json", summary)
        write_report(summary)
    progress(out, "COMPLETED", decision=result["decision"])
    return 0


def supervisor(args):
    out = BASE / args.run_name
    config = load_json(CONFIG)
    status_path = STATUS if args.kind == "formal" else STATUS.with_name(f"s18_s2_{args.run_name}.status.json")
    started = time.monotonic()
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONUNBUFFERED="1")
    command = [PYTHON, "-m", MODULE, "worker", "--kind", args.kind, "--run-name", args.run_name, "--gpu", str(args.gpu)]
    with (out / "worker.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            rc = process.poll()
            elapsed = time.monotonic() - started
            timed_out = elapsed > config["runtime"]["hard_timeout_seconds"]
            if timed_out and rc is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    rc = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    rc = process.wait()
            progress_file = out / "progress.json"
            details = load_json(progress_file) if progress_file.exists() else {}
            summary_path = out / "summary.json"
            completed = rc == 0 and summary_path.exists()
            state = "RUNNING" if rc is None else "COMPLETED" if completed else "INFRASTRUCTURE_FAILED"
            status = {"experiment_id": "s18_s2_mechanism_probe", "run_name": args.run_name,
                      "kind": args.kind, "state": state, "physical_gpu": args.gpu, "pid": process.pid,
                      "supervisor_pid": os.getpid(), "process_alive": rc is None,
                      "heartbeat_at": utc_now(), "elapsed_seconds": elapsed,
                      "hard_timeout_seconds": config["runtime"]["hard_timeout_seconds"],
                      "exit_code": rc, "timed_out": timed_out, "progress": details,
                      "status_path": str(status_path.relative_to(ROOT)), **SEALED}
            if completed:
                status["decision"] = load_json(summary_path)["decision"]
            atomic_json(status_path, status)
            if rc is not None:
                if args.kind == "formal":
                    append(LEDGER, {"event": "FINISHED", "at": utc_now(), "run_name": args.run_name,
                                    "state": state, "exit_code": rc, "decision": status.get("decision")})
                if not completed:
                    failure = {"status": state, "decision": "INFRASTRUCTURE_FAILED", "exit_code": rc,
                               "timed_out": timed_out, "config_sha256": sha256(CONFIG), **SEALED}
                    atomic_json(out / "failure.json", failure)
                    if args.kind == "formal":
                        atomic_json(BASE / "summary.json", failure)
                        write_report(failure)
                return 0 if completed else 1
            time.sleep(config["runtime"]["heartbeat_seconds"])


def launch(args):
    from experiment.phase17.core.run_manager import launch_background_tmux, wait_for_tmux_startup
    config = load_json(CONFIG)
    verify_inputs(config)
    if args.gpu in config["runtime"]["excluded_physical_gpus"]:
        raise RuntimeError("GPU is excluded by the Stage18 runtime policy")
    preflight = BASE / "preflight/manifest.json"
    if load_json(preflight)["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("preflight is stale")
    if args.kind == "formal":
        smoke = load_json(BASE / args.smoke_run / "summary.json")
        if smoke["decision"] != "ENGINEERING_PASS" or smoke["config_sha256"] != sha256(CONFIG):
            raise RuntimeError("current config must pass GPU engineering smoke first")
        smoke_manifest = load_json(BASE / args.smoke_run / "run_manifest.json")
        for ref in smoke_manifest["sources"]:
            checked(ref)
        if LEDGER.exists() or (BASE / "summary.json").exists():
            raise RuntimeError("scientific attempt already exists; no automatic retry")
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    free = dict((int(row.split(',')[0]), int(row.split(',')[1])) for row in snapshot.splitlines())
    required = 12000
    if args.kind == "formal":
        required = math.ceil(max(r["peak_reserved_mib"] for r in smoke["domains"].values()) + 1024)
    if free[args.gpu] < required:
        raise RuntimeError(f"GPU{args.gpu}: {free[args.gpu]} MiB free, measured launch requirement {required} MiB")
    out = BASE / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    session = f"s18_s2_{args.run_name.replace('-', '_')}"
    refs = [record(p) for p in source_paths()]
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for index, ref in enumerate(refs):
        destination = snapshots / f'{index:03d}_{Path(ref["path"]).name}'
        destination.write_bytes((ROOT / ref["path"]).read_bytes())
    manifest = {"schema_version": "phase18.s18_2_run_manifest.v1", "created_at": utc_now(),
                "kind": args.kind, "run_name": args.run_name, "config_sha256": sha256(CONFIG),
                "preflight": record(preflight), "sources": refs,
                "physical_gpu": args.gpu, "free_mib_at_launch": free[args.gpu],
                "required_free_mib": required, "tmux_session": session,
                "authorization": config["authorization"], **SEALED}
    atomic_json(out / "run_manifest.json", manifest)
    if args.kind == "formal":
        append(LEDGER, {"event": "STARTED", "at": utc_now(), "run_name": args.run_name,
                        "config_sha256": sha256(CONFIG), "manifest": record(out / "run_manifest.json")})
    argv = [PYTHON, "-m", MODULE, "supervisor", "--kind", args.kind, "--run-name", args.run_name, "--gpu", str(args.gpu)]
    launch_background_tmux(experiment_id="s18_s2_mechanism_probe", argv=argv, cwd=ROOT,
                           tmux_session=session, startup_log_path=out / "master.log")
    if not wait_for_tmux_startup(session):
        raise RuntimeError(f"background startup failed; inspect {out / 'master.log'}")
    return {"status": "STARTED", "tmux_session": session, "run_directory": str(out), "physical_gpu": args.gpu}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "launch", "supervisor", "worker"])
    parser.add_argument("--kind", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--run-name", default="run-0001")
    parser.add_argument("--smoke-run", default="smoke-0001")
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_name):
        raise ValueError("invalid run name")
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False))
        return 0
    if args.command == "launch":
        print(json.dumps(launch(args), ensure_ascii=False))
        return 0
    if args.command == "supervisor":
        return supervisor(args)
    return worker(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
