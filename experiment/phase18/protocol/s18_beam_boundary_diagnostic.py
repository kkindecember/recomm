"""Frozen-parent beam boundary and continuation-gradient diagnostic. No optimizer."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from experiment.phase18.protocol import s18_s2_mechanism_probe as s2
from experiment.phase18.core.contracts import ROOT, load_json, sha256
from experiment.phase18.protocol.s18_s1_prepare import atomic_json, atomic_text, utc_now

BASE = ROOT / "artifacts/phase18/beam_boundary_diagnostic"
CONFIG = ROOT / "experiment/phase18/config/s18_beam_boundary_diagnostic.json"
MODULE = "experiment.phase18.protocol.s18_beam_boundary_diagnostic"
STATUS = ROOT / "artifacts/phase18/status/s18_beam_boundary_diagnostic.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/beam_boundary_diagnostic.attempts.jsonl"
REPORT = ROOT / "report/第十八阶段/Stage18_真实Beam边界诊断执行报告.md"
ADDENDUM = ROOT / "plan/第十八阶段/GRAM_真实Beam边界诊断_执行补遗v0.1.md"
ARMS = s2.ARMS
SEALED = dict(s2.SEALED, optimizer_steps=0, new_checkpoints=0, accuracy_promotion=False)


def ref(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path), "sha256": sha256(path)}


def verify():
    config = load_json(CONFIG)
    old = load_json(s2.checked(config["s2_config"]))
    s2.verify_inputs(old)
    previous = load_json(s2.checked(config["s2_run_manifest"]))
    for source in previous["sources"]:
        s2.checked(source)
    for source in config["inputs"]:
        s2.checked(source)
    if (config["users_per_domain"], config["beam"], config["alpha"]) != (100, 50, 0.1):
        raise RuntimeError("diagnostic protocol changed")
    return config, old


def sources():
    import transformers.generation.utils as gen
    import transformers.generation.beam_search as beam
    from experiment.phase18.core import beam_boundary_trace
    return list(dict.fromkeys(s2.source_paths() + [CONFIG, Path(__file__), ADDENDUM,
                    ROOT / "experiment/phase18/run_stage18_beam_boundary_diagnostic.sh",
                    ROOT / "experiment/phase18/tests/test_s18_beam_boundary.py",
                    Path(beam_boundary_trace.__file__), Path(gen.__file__), Path(beam.__file__)]))


def manifest(out, kind, gpu):
    config, _ = verify()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    refs = []
    for i, path in enumerate(sources()):
        refs.append(ref(path))
        (snapshot / f"{i:03d}_{path.name}").write_bytes(path.read_bytes())
    data = {"created_at": utc_now(), "kind": kind, "gpu": gpu, "sources": refs,
            "config_sha256": sha256(CONFIG), "inputs": config["inputs"], **SEALED}
    atomic_json(out / "run_manifest.json", data)
    return data


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_equal(a, b, torch):
    if set(a) != set(b):
        raise RuntimeError("native/S18-2 batch keys differ")
    for key in a:
        equal = torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else a[key] == b[key]
        if not equal:
            raise RuntimeError(f"native/S18-2 input differs: {key}")


def audit_inputs(ctx, users, out):
    native = ctx.rt.MultiTaskDatasetGRAM(ctx.args, ctx.rt.dataset_name_from_manifest(ctx.domain, "I0"),
                                        "train", None, ctx.tokenizer)
    last = {s["user_id"]: i for i, s in enumerate(native.data_samples) if s["user_id"] in users}
    rows = []
    for user in users:
        row, i = ctx.train[user], last[user]
        expected = native.data_samples[i]
        if expected["target"] != row["target"]:
            raise RuntimeError("S18-2 target is not the final native training transition")
        batch_equal(ctx.batch(row), ctx.collator([native[i]]), ctx.torch)
        if ctx.domain == "Toys":
            evaluation = ctx.evaluation[user]
            if evaluation["history"] != row["history"] + [row["target"]]:
                raise RuntimeError("evaluation is not the next offset")
        rows.append({"user": user, "native_index": i, "target": row["target"], "exact_batch_identity": True})
    if ctx.shuffle != s2.shuffled_teacher_map(ctx.users, ctx.domain, ctx.config["seed"]):
        raise RuntimeError("frozen teacher shuffle differs")
    summary = {"users": len(users), "rows": rows, "native_all_transition_count": len(native),
               "s2_last_transition_count": len(ctx.users), "parent_effective_batch": 128,
               "s2_effective_batch": 16, "optimizer_scheduler_reset_in_s2": True,
               "native_batch_identity": True, "shuffle_identity": True}
    atomic_json(out / "input_audit.json", summary)
    del native
    gc.collect()
    return summary


def load_cache(ctx, users, evaluation=False):
    filename = "evaluation_anchors.jsonl" if evaluation else "training_cache.jsonl"
    all_rows = s2.Context.read_rows(s2.BASE / "run-0001" / ctx.domain / filename)
    selected = {}
    for user in users:
        cache = all_rows[user]
        row = (ctx.evaluation if evaluation else ctx.train)[user]
        if cache["target"] != row["target"]:
            raise RuntimeError("negative cache target mismatch")
        if any(i == row["target"] or i not in ctx.item_paths for i in cache["path_items"]):
            raise RuntimeError("illegal path negative")
        target = ctx.item_paths[row["target"]]
        for info in cache["arms"].values():
            if info["path_items"] != cache["path_items"]:
                raise RuntimeError("shared path negatives changed")
            for node in info["nodes"]:
                d = node["depth"]
                children = set()
                for item in node["negative_items"]:
                    p = ctx.item_paths[item]
                    if len(p) <= d or p[:d] != target[:d] or p[d] == target[d]:
                        raise RuntimeError("illegal prefix negative")
                    children.add(p[d])
                if children != set(node["negative_children"]):
                    raise RuntimeError("prefix negative identity mismatch")
        selected[user] = cache
    return selected


def generate(ctx, model, row, traced=True):
    from transformers.generation.logits_process import LogitsProcessorList
    from experiment.phase18.core.beam_boundary_trace import BoundaryTrace
    from contextlib import nullcontext
    batch = ctx.batch(row)
    trace = BoundaryTrace(ctx.item_paths[row["target"]]) if traced else None
    kwargs = {"logits_processor": LogitsProcessorList([trace])} if traced else {}
    with ctx.torch.no_grad(), trace.installed() if traced else nullcontext():
        output = model.generate(input_ids=batch["item_text_ids"].to(ctx.device),
                   attention_mask=batch["item_text_masks"].to(ctx.device),
                   history_item_ids=batch["history_item_ids"].to(ctx.device),
                   history_item_mask=batch["history_item_mask"].to(ctx.device),
                   max_length=ctx.max_length, prefix_allowed_tokens_fn=lambda b, p: ctx.trie.get(p.tolist()),
                   num_beams=50, num_return_sequences=50, output_scores=True,
                   return_dict_in_generate=True, length_penalty=1.0, use_cache=True, **kwargs)
    paths = [ctx.rt.trim_generated_path(p) for p in output.sequences]
    items = [ctx.path_item.get(p) for p in paths]
    if None in items or len(set(items)) != 50 or not bool(ctx.torch.isfinite(output.sequences_scores).all()):
        raise RuntimeError("illegal final native beam")
    result = {"items": items, "sequences": output.sequences.cpu().tolist(),
              "scores": output.sequences_scores.cpu().tolist()}
    if trace is not None and (trace.final["sequences"] != result["sequences"] or
                              trace.final["sequence_scores"] != result["scores"]):
        raise RuntimeError("final scorer result differs from native generate output")
    return result, trace


def prefix_scores(ctx, model, row, paths):
    """Differentiable full-vocabulary cumulative scores, no EOS added to prefixes."""
    torch, batch = ctx.torch, ctx.batch(row)
    labels = torch.full((len(paths), max(map(len, paths))), -100, dtype=torch.long, device=ctx.device)
    for i, path in enumerate(paths):
        labels[i, :len(path)] = torch.tensor(path, device=ctx.device)
    n = len(paths)
    output = model(input_ids=batch["item_text_ids"].to(ctx.device).repeat(n, 1, 1),
                   attention_mask=batch["item_text_masks"].to(ctx.device).repeat(n, 1, 1),
                   history_item_ids=batch["history_item_ids"].to(ctx.device).repeat(n, 1),
                   history_item_mask=batch["history_item_mask"].to(ctx.device).repeat(n, 1),
                   labels=labels, use_cache=False, return_dict=True)
    logp = output.logits.float().log_softmax(-1)
    selected = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (selected * labels.ne(-100)).sum(-1)


def gradient_vector(model, torch):
    vector = torch.cat([p.grad.detach().cpu().reshape(-1) if p.grad is not None else
                        torch.zeros(p.numel(), dtype=p.dtype) for p in model.parameters()])
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError("nonfinite gradient; MEASUREMENT_UNRESOLVED")
    return vector


def dot(a, b):
    # Double-precision reductions in bounded CPU chunks.
    return sum(float(a[i:i + 1000000].double().dot(b[i:i + 1000000].double()))
               for i in range(0, a.numel(), 1000000))


def mean_gradients(ctx, model, users, cache, out):
    gradients, records = {}, {}
    for arm in ARMS:
        model.zero_grad(set_to_none=True)
        mean_loss = 0.
        for i, user in enumerate(users, 1):
            loss, _, _, _ = ctx.forward(model, ctx.train[user], arm, .1,
                                       None if arm == ARMS[0] else cache[user]["arms"][arm])
            mean_loss += float(loss.detach()) / len(users)
            (loss / len(users)).backward()
            del loss
            if i == 1 or i % 10 == 0:
                s2.progress(out.parent, "MEAN_TRAIN_GRADIENT", domain=ctx.domain, arm=arm,
                            users_done=i, users_total=len(users), **SEALED)
        g = gradient_vector(model, ctx.torch)
        gradients[arm] = g
        records[arm] = {"mean_loss": mean_loss, "gradient_norm": math.sqrt(dot(g, g)),
                        "gradient_sha256": hashlib.sha256(g.numpy().tobytes()).hexdigest(),
                        "gradient_elements": g.numel()}
    model.zero_grad(set_to_none=True)
    for arm in ARMS[1:]:
        gradients[arm + "_AUX_INCREMENT"] = gradients[arm] - gradients[ARMS[0]]
    records["auxiliary_increment_norms"] = {arm: math.sqrt(dot(gradients[arm], gradients[arm]))
                                             for arm in gradients if arm.endswith("_AUX_INCREMENT")}
    atomic_json(out / "mean_training_gradients.json", records)
    return gradients, records


def scoring_record(event, values):
    target, boundary = map(float, values)
    rtarget, rboundary = target - event["target_score"], boundary - event["boundary_score"]
    margin = target - boundary
    budget = abs(rtarget) + abs(rboundary)
    sign_match = (margin > 0) - (margin < 0) == (event["native_margin"] > 0) - (event["native_margin"] < 0)
    return {"tf_target": target, "tf_boundary": boundary, "tf_margin": margin,
            "target_residual": rtarget, "boundary_residual": rboundary,
            "margin_residual": margin - event["native_margin"], "sign_match": sign_match,
            "near_tie_or_sign_uncertain": not sign_match or abs(event["native_margin"]) <= budget or event["global_ties"] > 1,
            "residual_status": "EXACT" if rtarget == 0 and rboundary == 0 else "UNEXPLAINED_NONZERO_RESIDUAL"}


def trace_users(ctx, model, users, cache, out, split, gradients=None):
    from experiment.phase18.core.beam_boundary_trace import first_drop_summary
    summaries = []
    data = ctx.train if split == "training" else ctx.evaluation
    norms = {k: math.sqrt(dot(g, g)) for k, g in (gradients or {}).items()}
    for i, user in enumerate(users, 1):
        row = data[user]
        result, trace = generate(ctx, model, row)
        event = first_drop_summary(trace, ctx.item_paths, cache[user]["beam50_items"], cache[user])
        summary = {"user": user, "target": row["target"], "split": split, "event": event,
                   "native_hit50": row["target"] in result["items"],
                   "cached_final_beam_order_equal": result["items"] == cache[user]["beam50_items"]}
        if event["event"] and event["boundary_prefix"] is not None:
            model.zero_grad(set_to_none=True)
            with ctx.torch.set_grad_enabled(gradients is not None):
                values = prefix_scores(ctx, model, row, [event["target_prefix"], event["boundary_prefix"]])
                if not bool(ctx.torch.isfinite(values).all()):
                    raise FloatingPointError("nonfinite margin; MEASUREMENT_UNRESOLVED")
                summary["score_audit"] = scoring_record(event, values.detach().cpu().tolist())
                if gradients is not None:
                    (values[0] - values[1]).backward()
            del values
            if gradients is not None:
                gmargin = gradient_vector(model, ctx.torch)
                summary["directions_exploratory"] = {}
                for arm, g in gradients.items():
                    change = -dot(gmargin, g)
                    if not math.isfinite(change):
                        raise FloatingPointError("nonfinite direction; MEASUREMENT_UNRESOLVED")
                    summary["directions_exploratory"][arm] = {"raw": change,
                        "unit_norm": change / norms[arm] if norms[arm] else None}
                del gmargin
                model.zero_grad(set_to_none=True)
        s2.append(out / f"{split}_traces.jsonl", {"user": user, "target": row["target"], **trace.payload()})
        s2.append(out / f"{split}_events.jsonl", summary)
        summaries.append(summary)
        if i == 1 or i % 10 == 0:
            s2.progress(out.parent, "NATIVE_BOUNDARY_TRACE", domain=ctx.domain, split=split,
                        users_done=i, users_total=len(users), **SEALED)
    return summaries


def aggregate(rows):
    events = [r["event"] for r in rows if r["event"]["event"]]
    valid = [e for e in events if e["boundary_prefix"] is not None]
    audits = [r["score_audit"] for r in rows if "score_audit" in r]
    total = sum(e["competitor_prefixes"] for e in events)
    dirs = [r for r in rows if "directions_exploratory" in r]
    result = {"users": len(rows), "first_drop_events": len(events), "complete_boundary_events": len(valid),
              "evidence_status": "DESCRIPTIVE_SUFFICIENT_COUNT" if len(valid) >= 30 else "INSUFFICIENT_EVENTS",
              "local_top50_but_pruned": sum(e["local_rank"] <= 50 for e in events),
              "single_child_but_pruned": sum(e["branch_degree"] == 1 for e in events),
              "boundary_cross_parent": sum(e["boundary_cross_parent"] is True for e in valid),
              "target_tie_events": sum(e["global_ties"] > 1 for e in events),
              "eos_at_drop_events": sum(e["eos_offered_count"] > 0 for e in events),
              "proxy_covered_prefixes": sum(e["proxy_overlap_prefix_count"] for e in events),
              "competitor_prefixes_total": total,
              "proxy_prefix_coverage": sum(e["proxy_overlap_prefix_count"] for e in events) / total if total else None,
              "unexplained_nonzero_score_residuals": sum(a["residual_status"] != "EXACT" for a in audits),
              "max_absolute_individual_score_residual": max([abs(a[k]) for a in audits for k in ("target_residual", "boundary_residual")] or [0.]),
              "near_tie_or_sign_uncertain": sum(a["near_tie_or_sign_uncertain"] for a in audits),
              "direction_event_count": len(dirs), "directions_exploratory": {}}
    for arm in dirs[0]["directions_exploratory"] if dirs else []:
        values = [r["directions_exploratory"][arm]["raw"] for r in dirs]
        unit = [r["directions_exploratory"][arm]["unit_norm"] for r in dirs]
        result["directions_exploratory"][arm] = {"count": len(values), "mean_raw": sum(values) / len(values),
                     "positive_count": sum(v > 0 for v in values), "negative_count": sum(v < 0 for v in values),
                     "mean_unit_norm": sum(unit) / len(unit) if None not in unit else None}
    return result


def smoke(ctx, model, users, cache, out):
    # Longest encoded batch across the fixed training set; deterministic tie break by cohort order.
    user = max(users, key=lambda u: ctx.batch(ctx.train[u])["item_text_ids"].numel())
    row = ctx.train[user]
    before = state_hash(model)
    ctx.torch.cuda.reset_peak_memory_stats()
    baseline, _ = generate(ctx, model, row, traced=False)
    traced, trace = generate(ctx, model, row)
    if baseline != traced:
        raise RuntimeError("passive tracer changes native beam order or scores")
    native, _, _, _, _ = ctx.rt.generate_one(model, ctx.batch(row), ctx.trie, ctx.max_length, 50, ctx.device)
    if native.sequences.cpu().tolist() != traced["sequences"] or native.sequences_scores.cpu().tolist() != traced["scores"]:
        raise RuntimeError("diagnostic generation differs from S18 native generation helper")
    del native
    # Alpha-zero contract on real parameters, followed by full four-objective backward memory smoke.
    zero_grads = []
    for arm in (ARMS[0], ARMS[2]):
        model.zero_grad(set_to_none=True)
        loss, _, _, _ = ctx.forward(model, row, arm, 0, cache=None)
        loss.backward()
        zero_grads.append(gradient_vector(model, ctx.torch))
    if not ctx.torch.equal(*zero_grads):
        raise RuntimeError("alpha-zero parameter gradients differ")
    del zero_grads
    gradients, norms = mean_gradients(ctx, model, [user], cache, out)
    # Two equal-depth legal prefixes exercise differentiable scoring regardless of a target drop.
    paths = [ctx.item_paths[i] for i in traced["items"][:2]]
    depth = min(map(len, paths)) - 1
    model.zero_grad(set_to_none=True)
    scores = prefix_scores(ctx, model, row, [p[:depth] for p in paths])
    (scores[0] - scores[1]).backward()
    g = gradient_vector(model, ctx.torch)
    if not all(math.isfinite(dot(g, v)) for v in gradients.values()):
        raise FloatingPointError("nonfinite smoke derivative")
    model.zero_grad(set_to_none=True)
    after = state_hash(model)
    if before != after:
        raise RuntimeError("smoke changed frozen weights")
    s2.append(out / "smoke_trace.jsonl", {"user": user, **trace.payload()})
    return {"user": user, "tracer_identity": True, "native_helper_identity": True,
            "alpha_zero_gradient_identity": True, "parameters_sha256_before": before,
            "parameters_sha256_after": after, "peak_reserved_mib": ctx.torch.cuda.max_memory_reserved() / 2**20,
            "peak_allocated_mib": ctx.torch.cuda.max_memory_allocated() / 2**20, "gradient_smoke": norms}


def worker(args):
    config, old = verify()
    out = BASE / args.run_name
    for source in load_json(out / "run_manifest.json")["sources"]:
        s2.checked(source)
    from experiment.phase18.protocol import s18_s1_runtime as rt
    rt.torch.set_num_threads(4)
    rt.set_seed(config["seed"])
    if rt.torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU required")
    started, domains = time.monotonic(), {}
    for domain in ("Toys", "Beauty"):
        destination = out / domain
        destination.mkdir()
        ctx = s2.Context(domain, old)
        users = ctx.users[:config["users_per_domain"]]
        atomic_json(destination / "selected_inputs.json", {"users": users, "training": [ctx.train[u] for u in users],
                    "evaluation": [ctx.evaluation[u] for u in users] if domain == "Toys" else [], **SEALED})
        audit_inputs(ctx, users, destination)
        cache = load_cache(ctx, users)
        model = ctx.parent().eval()
        before = state_hash(model)
        if args.kind == "smoke":
            domains[domain] = smoke(ctx, model, users, cache, destination)
        else:
            training = trace_users(ctx, model, users, cache, destination, "training")
            gradients, gradient_summary = mean_gradients(ctx, model, users, cache, destination)
            domains[domain] = {"training": aggregate(training), "mean_gradients": gradient_summary}
            if domain == "Toys":
                evaluation = trace_users(ctx, model, users, load_cache(ctx, users, evaluation=True),
                                         destination, "evaluation", gradients)
                domains[domain]["evaluation"] = aggregate(evaluation)
            after = state_hash(model)
            if before != after:
                raise RuntimeError("frozen model changed during diagnostic")
            domains[domain].update(parameters_sha256_before=before, parameters_sha256_after=after)
            del gradients
        atomic_json(destination / "summary.json", domains[domain])
        del model, ctx
        gc.collect()
        rt.torch.cuda.empty_cache()
    verify()
    result = {"decision": "ENGINEERING_PASS" if args.kind == "smoke" else "DESCRIPTIVE_DIAGNOSTIC_COMPLETED",
              "kind": args.kind, "domains": domains, "config_sha256": sha256(CONFIG),
              "completed_at": utc_now(), "elapsed_seconds": time.monotonic() - started, **SEALED}
    if args.kind == "formal":
        unresolved = any(d[s]["unexplained_nonzero_score_residuals"] for d in domains.values()
                         for s in ("training", "evaluation") if s in d)
        result["gradient_measurement_status"] = "MEASUREMENT_UNRESOLVED" if unresolved else "SCORE_IDENTITY_VERIFIED"
        result["training_authorized_by_result"] = False
        result["s18_2_decision_unchanged"] = "FAILED_SCIENTIFIC_GATE"
    atomic_json(out / "summary.json", result)
    if args.kind == "formal":
        write_report(result, out)
    s2.progress(out, "COMPLETED", decision=result["decision"], **SEALED)
    return 0


def write_report(result, out):
    lines = ["# 第十八阶段：真实 Beam 边界诊断", "", f"完成时间：{result.get('completed_at', utc_now())}。",
             "", f"执行状态：`{result['decision']}`。S18-2 仍为 `FAILED_SCIENTIFIC_GATE`。",
             "本轮冻结模型，只计算轨迹和梯度，optimizer step=0；没有新训练或 accuracy 晋级。", ""]
    if "domains" in result:
        lines += ["| 域/输入 | 用户 | 首次非 EOS drop | 局部 top50 仍掉出 | 跨父边界 | 旧 sibling proxy 覆盖 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for domain, d in result["domains"].items():
            for split in ("training", "evaluation"):
                if split in d:
                    a = d[split]
                    coverage = a["proxy_prefix_coverage"]
                    lines.append(f"| {domain}/{split} | {a['users']} | {a['first_drop_events']} | "
                                 f"{a['local_top50_but_pruned']} | {a['boundary_cross_parent']} | "
                                 f"{coverage:.4f} |" if coverage is not None else f"| {domain}/{split} | 无可比事件 |")
        lines += ["", f"梯度测量状态：`{result['gradient_measurement_status']}`。",
                  "非零 teacher-forcing/native 残差在解释前保留为测量未决；方向统计仅作探索，不能据此启动训练。",
                  "Toys evaluation 为同一用户下一 offset，未使用新的确认集。Beauty 只有 training 诊断。"]
    lines += ["", f"完整结果与逐步轨迹：[{out.name}](../../{out.relative_to(ROOT)})。", ""]
    atomic_text(REPORT, "\n".join(lines))


def resources(gpu, required):
    if gpu in (1, 4):
        raise RuntimeError("GPU1/GPU4 excluded")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    devices = {int(r.split(',')[0]): [x.strip() for x in r.split(',')[1:]] for r in raw.splitlines()}
    uuid, free = devices[gpu]
    if int(free) < required:
        raise RuntimeError(f"GPU{gpu} free {free} MiB < {required} MiB required")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    pids = [r.split(',')[1].strip() for r in processes.splitlines() if r.startswith(uuid)]
    commands = subprocess.check_output(["ps", "-o", "pid,user,args", "-p", ','.join(pids)], text=True) if pids else ""
    if re.search(r"occupancy|postrun_guard|runtime_guard|reservation", commands, flags=re.I):
        raise RuntimeError("candidate GPU contains a guard/reservation process")
    return {"gpu": gpu, "free_mib": int(free), "required_mib": required,
            "gpu_snapshot": raw, "selected_gpu_processes": commands}


def supervisor(args):
    out = BASE / args.run_name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    with (out / "worker.log").open("a") as log:
        p = subprocess.Popen([s2.PYTHON, "-m", MODULE, "worker", "--run-name", args.run_name, "--kind", args.kind,
                              "--gpu", str(args.gpu)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        while True:
            elapsed, rc = time.monotonic() - started, p.poll()
            timeout = elapsed > (600 if args.kind == "smoke" else 7200)
            if timeout and rc is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    rc = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    rc = p.wait()
            completed = rc == 0 and (out / "summary.json").exists()
            status = {"state": "RUNNING" if rc is None else "COMPLETED" if completed else "INFRASTRUCTURE_FAILED",
                      "run_name": args.run_name, "kind": args.kind, "physical_gpu": args.gpu,
                      "pid": p.pid, "supervisor_pid": os.getpid(), "heartbeat_at": utc_now(),
                      "elapsed_seconds": elapsed, "exit_code": rc, "timed_out": timeout,
                      "progress": load_json(out / "progress.json") if (out / "progress.json").exists() else {}, **SEALED}
            atomic_json(out / "status.json", status)
            if args.kind == "formal":
                atomic_json(STATUS, status)
            if rc is not None:
                if not completed:
                    failure = dict(status, decision="INFRASTRUCTURE_FAILED", gradient_measurement_status="MEASUREMENT_UNRESOLVED")
                    atomic_json(out / "failure.json", failure)
                    if args.kind == "formal":
                        write_report(failure, out)
                if args.kind == "formal":
                    s2.append(LEDGER, {"event": "FINISHED", **status})
                return 0 if completed else 1
            time.sleep(5 if args.kind == "smoke" else 30)


def launch(args):
    from experiment.phase17.core.run_manager import launch_background_tmux, wait_for_tmux_startup
    smoke_out = BASE / args.smoke_run
    result = load_json(smoke_out / "summary.json")
    if result["decision"] != "ENGINEERING_PASS" or result["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("current config has not passed smoke")
    for source in load_json(smoke_out / "run_manifest.json")["sources"]:
        s2.checked(source)
    if LEDGER.exists():
        raise RuntimeError("diagnostic attempt exists; automatic retry forbidden")
    resource = resources(args.gpu, math.ceil(max(d["peak_reserved_mib"] for d in result["domains"].values()) + 1024))
    out = BASE / args.run_name
    manifest(out, "formal", args.gpu)
    atomic_json(out / "launch_resources.json", resource)
    session = "s18_boundary_" + args.run_name.replace('-', '_')
    s2.append(LEDGER, {"event": "STARTED", "at": utc_now(), "manifest": ref(out / "run_manifest.json"),
                       "tmux_session": session, "gpu": args.gpu, **SEALED})
    launch_background_tmux(experiment_id="s18_beam_boundary_diagnostic", cwd=ROOT, tmux_session=session,
        startup_log_path=out / "master.log", argv=[s2.PYTHON, "-m", MODULE, "supervisor", "--run-name", args.run_name,
                                                "--kind", "formal", "--gpu", str(args.gpu)])
    if not wait_for_tmux_startup(session):
        raise RuntimeError("background startup failed; inspect master.log")
    print(json.dumps({"state": "STARTED", "tmux_session": session, "run_directory": str(out), **resource}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "launch", "supervisor", "worker"])
    parser.add_argument("--kind", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--run-name", default="run-0001")
    parser.add_argument("--smoke-run", default="smoke-0001")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.run_name):
        raise ValueError("invalid run name")
    if args.command == "smoke":
        args.kind = "smoke"
        resource = resources(args.gpu, 12000)
        out = BASE / args.run_name
        manifest(out, "smoke", args.gpu)
        atomic_json(out / "launch_resources.json", resource)
        return supervisor(args)
    return {"worker": worker, "supervisor": supervisor, "launch": launch}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
