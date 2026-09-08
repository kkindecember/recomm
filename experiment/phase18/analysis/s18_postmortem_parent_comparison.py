"""Compare all completed arms with their frozen parent using existing I0 rows."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/phase18/s2_mechanism_probe/run-0001/Toys"
OUT = ROOT / "artifacts/phase18/postmortem_research/run-0001/parent_comparison.json"


def main():
    if OUT.exists():
        raise RuntimeError("preserve existing retrospective comparison")
    paths = [BASE / "evaluation_anchors.jsonl"]
    anchors = [json.loads(line) for line in paths[0].read_text().splitlines()]
    parent = {r["user"]: dict(user=r["user"], target=r["target"],
              rank=r["beam50_items"].index(r["target"])+1 if r["target"] in r["beam50_items"] else 51,
              prefix_survival=r["parent_prefix_survival"]) for r in anchors}
    arms = {"PARENT_FROZEN": parent}
    for arm in ("C0_CONT", "A0_LEGAL_GENERIC", "M0_PCPS", "S0_SHUFFLED_CF"):
        path = BASE / "probe" / arm / "per_user.tsv"
        paths.append(path)
        rows = list(csv.DictReader(path.open(), delimiter="\t"))
        assert len(rows) == len({r["user"] for r in rows}) == len(parent) == 1000
        arms[arm] = {r["user"]: {**r, "rank": int(r["rank"]),
                    "prefix_survival": float(r["prefix_survival"])} for r in rows}
        assert set(arms[arm]) == set(parent)
        assert all(r["target"] == parent[u]["target"] for u, r in arms[arm].items())
    result = {"verification_status": "ANALYZED", "posthoc": True,
              "scope": "same 1000 already evaluated Toys I0 users",
              "canonical_s18_2_decision_changed": False,
              "metrics": {}, "paired_vs_parent": {}}
    for arm, rs in arms.items():
        result["metrics"][arm] = {
            "Hit@50": sum(r["rank"] <= 50 for r in rs.values())/1000,
            "Hit@10": sum(r["rank"] <= 10 for r in rs.values())/1000,
            "NDCG@10": sum(1/math.log2(r["rank"]+1) if r["rank"] <= 10 else 0 for r in rs.values())/1000,
            "prefix_survival": sum(r["prefix_survival"] for r in rs.values())/1000}
        diffs = [int(rs[u]["rank"] <= 50)-int(p["rank"] <= 50) for u,p in parent.items()]
        result["paired_vs_parent"][arm] = {"hit50_gain": sum(d > 0 for d in diffs),
                "hit50_loss": sum(d < 0 for d in diffs), "hit50_tie": sum(d == 0 for d in diffs),
                "deltas": {k: v-result["metrics"]["PARENT_FROZEN"][k]
                           for k,v in result["metrics"][arm].items()}}
    result["input_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "input_sha256"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
