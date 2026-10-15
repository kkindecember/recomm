"""Inventory research inputs and count train-only support for a proposed gate.

No fitting, generation, test data, or modification of existing artifacts.
An indexed file is not a claim of line-by-line human/assistant review.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inventory():
    records = []
    for folder in ("plan", "report", "experiment", "GRAM/src", "third_party/LIME-Rec"):
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.suffix not in (".md", ".py", ".sh", ".yaml"):
                continue
            if any(part in (".git", "__pycache__", "output", "data") for part in path.parts):
                continue
            if path.name == "快速成稿定位与最小方法改造建议_20261010.md":
                continue
            raw = path.read_bytes()
            content = raw.decode("utf-8", errors="replace")
            entry = dict(path=str(path.relative_to(ROOT)), bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest(), lines=len(content.splitlines()),
                         access="indexed_not_line_by_line_reviewed")
            if path.suffix == ".md":
                entry["headings"] = [s for s in content.splitlines() if s.startswith("#")]
            if path.suffix == ".py":
                try:
                    tree = ast.parse(content)
                    entry["top_level_symbols"] = [n.name for n in tree.body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                except SyntaxError as err:
                    entry["parse_note"] = str(err)
            records.append(entry)
    dump("input_inventory.json", records)
    return dict(indexed_files=len(records), markdown_by_root=dict(Counter(
        r["path"].split("/")[0] for r in records if r["path"].endswith(".md"))))


def training_support(domain):
    base = ROOT / "artifacts/phase20/lime_clean_benefit_20260922_v1" / domain
    samples = json.loads((base / "samples.json").read_text())
    experts = np.load(base / "experts.npy", mmap_mode="r")
    fused = np.load(base / "lime.npy", mmap_mode="r")
    fit = [j for j, s in enumerate(samples) if s["role"] == "ranker_fit"]
    tally = Counter()
    sizes = []
    sources = []
    for name in ("samples.json", "experts.npy", "lime.npy"):
        p = base / name
        sources.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                            mtime_ns=p.stat().st_mtime_ns))
    for j in fit:
        sample = samples[j]
        matrix = np.concatenate((np.asarray(experts[j]), np.asarray(fused[j])[None]), axis=0).copy()
        matrix[:, np.asarray(sample["history"], dtype=int) - 1] = -np.inf
        # IDs break ties deterministically, matching the project's ranking convention.
        ids = np.arange(matrix.shape[1])
        tops = [np.lexsort((ids, -row))[:50] for row in matrix]
        support = set(np.concatenate(tops).tolist())
        # The support set has already been fixed without the target.
        target = int(sample["target"]) - 1
        tally["events"] += 1
        tally["target_in_support"] += target in support
        tally["target_in_fused_top50"] += target in set(tops[-1].tolist())
        tally["target_in_history_filter"] += target + 1 in sample["history"]
        for e, name in enumerate(("sasrec", "itemcf", "text")):
            tally["target_in_" + name + "_top50"] += target in set(tops[e].tolist())
        sizes.append(len(support))
    for source in sources:
        p = ROOT / source["path"]
        assert p.stat().st_size == source["bytes"] and p.stat().st_mtime_ns == source["mtime_ns"]
    return dict(**tally, support_mean=float(np.mean(sizes)), support_min=min(sizes),
                support_max=max(sizes), coverage=tally["target_in_support"] / len(fit),
                support_definition="Union of each expert top50 and original fused top50; recent history filtered",
                sources=sources)


def main():
    start = time.monotonic()
    indexed = inventory()
    support = {domain: training_support(domain) for domain in ("Toys", "Beauty")}
    source = ROOT / "artifacts/phase21/full_validation_20260924_v1/summary.json"
    summary = json.loads(source.read_text())
    margins = {}
    for domain, values in summary["domains"].items():
        m = values["metrics"]
        margins[domain] = dict(
            gram_N10=m["GRAM"]["N10"], late_N10=m["late"]["N10"], guided_N10=m["guided"]["N10"],
            relative_gain_over_gram=m["guided"]["N10"] / m["GRAM"]["N10"] - 1,
            relative_gain_over_late=m["guided"]["N10"] / m["late"]["N10"] - 1,
            relative_loss_from_guided_at_late_parity=1 - m["late"]["N10"] / m["guided"]["N10"])
    result = dict(state="COMPLETED", date="2026-10-10", indexed=indexed,
                  training_support=support, existing_validation_margins=margins,
                  summary_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope="Inventory and cached training coverage only; no new method evaluated",
                  fitting_steps=0, generation_calls=0, test_read=False,
                  elapsed_seconds=time.monotonic()-start)
    dump("audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
