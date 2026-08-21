"""Train an item-disjoint R2 teacher and freeze its score calibration.

Only the student-readable train-prefix artifact from Stage 14-1a is accepted;
the held pseudo-cold ground-truth directory is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUserProjector(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values + self.residual_scale * self.net(values), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sequences", required=True)
    parser.add_argument("--pseudo-cold-items", required=True)
    parser.add_argument("--real-cold-items", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--calibration-modulus", type=int, default=10)
    parser.add_argument(
        "--score-temperatures", default="0.03,0.05,0.07,0.10,0.20"
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def read_train_sequences(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            row = json.loads(raw)
            if set(row) != {"user_id", "train_items"}:
                raise ValueError(f"{path}:{line_no}: unexpected schema")
            rows.append((str(row["user_id"]), [str(item) for item in row["train_items"]]))
    if not rows:
        raise ValueError("No student-readable train sequences")
    return rows


def recency_weighted_history(
    indices: list[int], embeddings: torch.Tensor, decay: float
) -> torch.Tensor:
    if not indices:
        raise ValueError("History cannot be empty")
    values = embeddings[torch.tensor(indices, dtype=torch.long)]
    ages = torch.arange(len(indices) - 1, -1, -1, dtype=values.dtype)
    weights = decay**ages
    return F.normalize((values * weights[:, None]).sum(0) / weights.sum(), dim=0)


def fold(seed: int, user: str, position: int, modulus: int) -> int:
    value = hashlib.sha256(f"{seed}:{user}:{position}".encode()).digest()
    return int.from_bytes(value[:8], "big") % modulus


def build_examples(
    sequences: list[tuple[str, list[str]]],
    item_to_idx: dict[str, int],
    embeddings: torch.Tensor,
    forbidden: set[str],
    max_history: int,
    decay: float,
    seed: int,
    modulus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    train_x: list[torch.Tensor] = []
    train_y: list[int] = []
    calibration_x: list[torch.Tensor] = []
    calibration_y: list[int] = []
    unique_targets: set[str] = set()
    for user, items in sequences:
        if any(item in forbidden for item in items):
            raise RuntimeError(f"Forbidden item in student-readable sequence for {user}")
        for position in range(1, len(items)):
            target = items[position]
            history = items[max(0, position - max_history):position]
            vector = recency_weighted_history(
                [item_to_idx[item] for item in history], embeddings, decay
            )
            unique_targets.add(target)
            if fold(seed, user, position, modulus) == 0:
                calibration_x.append(vector)
                calibration_y.append(item_to_idx[target])
            else:
                train_x.append(vector)
                train_y.append(item_to_idx[target])
    if not train_x or not calibration_x:
        raise ValueError("Train/calibration split is empty")
    return (
        torch.stack(train_x),
        torch.tensor(train_y, dtype=torch.long),
        torch.stack(calibration_x),
        torch.tensor(calibration_y, dtype=torch.long),
        {
            "n_train": len(train_x),
            "n_calibration": len(calibration_x),
            "n_unique_targets": len(unique_targets),
            "forbidden_target_count": 0,
            "forbidden_history_count": 0,
        },
    )


def multi_positive_loss(
    users: torch.Tensor,
    targets: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = users @ targets.T / temperature
    positive = target_ids[:, None].eq(target_ids[None, :])
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    return -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()


def train_teacher(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    embeddings: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ResidualUserProjector, list[dict]]:
    model = ResidualUserProjector(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        count = 0
        model.train()
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset:offset + args.batch_size]
            x = train_x[rows].to(device)
            y = train_y[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            users = model(x)
            loss = multi_positive_loss(
                users, embeddings[y].to(device), y, args.contrastive_temperature
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
            count += len(rows)
        record = {
            "epoch": epoch,
            "loss": total / count,
            "residual_scale": float(model.residual_scale.detach()),
        }
        history.append(record)
        print(f"[teacher] epoch={epoch}/{args.epochs} loss={record['loss']:.6f}", flush=True)
    model.eval()
    return model, history


def calibrate_scores(
    model: ResidualUserProjector,
    x: torch.Tensor,
    y: torch.Tensor,
    embeddings: torch.Tensor,
    temperatures: list[float],
    candidate_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict]:
    totals = {temperature: 0.0 for temperature in temperatures}
    hit = 0
    margins: list[float] = []
    entropies: list[float] = []
    catalog = embeddings.to(device)
    with torch.no_grad():
        for offset in range(0, len(x), batch_size):
            batch_x = x[offset:offset + batch_size].to(device)
            batch_y = y[offset:offset + batch_size].to(device)
            scores = model(batch_x) @ catalog.T
            for temperature in temperatures:
                totals[temperature] += float(
                    F.cross_entropy(scores / temperature, batch_y, reduction="sum")
                )
            top_scores, top_indices = torch.topk(scores, k=candidate_size, dim=1)
            hit += int(top_indices.eq(batch_y[:, None]).any(dim=1).sum())
            margins.extend((top_scores[:, 0] - top_scores[:, 1]).cpu().tolist())
            probabilities = F.softmax(top_scores / 0.07, dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            entropies.extend((entropy / math.log(candidate_size)).cpu().tolist())
    nll = {str(value): totals[value] / len(x) for value in temperatures}
    selected = min(temperatures, key=lambda value: totals[value])
    sorted_margins = sorted(margins)
    sorted_entropies = sorted(entropies)
    quantile = lambda values, ratio: values[int(round(ratio * (len(values) - 1)))]
    report = {
        "candidate_size": candidate_size,
        "n_calibration": len(x),
        "target_recall_at_m": hit / len(x),
        "temperature_nll": nll,
        "selected_temperature": selected,
        "confidence_rule": {
            "name": "margin_entropy_prefix_coverage_v1",
            "margin_q25": quantile(sorted_margins, 0.25),
            "margin_q50": quantile(sorted_margins, 0.50),
            "margin_q75": quantile(sorted_margins, 0.75),
            "normalized_entropy_q25": quantile(sorted_entropies, 0.25),
            "normalized_entropy_q50": quantile(sorted_entropies, 0.50),
            "normalized_entropy_q75": quantile(sorted_entropies, 0.75),
            "prefix_coverage_component": "0.5 + 0.5 * min(1, descendant_count / 5)",
            "fit_inputs": "retained-warm train calibration fold only",
        },
    }
    return selected, report


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    paths = {
        "train_sequences": Path(args.train_sequences).resolve(),
        "pseudo_cold_items": Path(args.pseudo_cold_items).resolve(),
        "real_cold_items": Path(args.real_cold_items).resolve(),
        "item_embeddings": Path(args.item_embeddings).resolve(),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in {"status.json", "run.log", "gpu_telemetry.csv"}]
    if unexpected:
        raise FileExistsError(f"Refusing existing teacher artifacts: {unexpected}")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if "held_ground_truth" in str(path):
            raise ValueError("Teacher must not read held pseudo-cold ground truth")
    experiment_id = "GRAM_PHASE14_STAGE14_1B_ITEM_DISJOINT_R2_TEACHER_TOYS"
    atomic_json(
        output_dir / "status.json",
        {
            "experiment_id": experiment_id,
            "status": "running",
            "stage": "item_disjoint_teacher_training",
            "reason": "Training only from student-readable retained-warm prefixes.",
            "automatic_retry": False,
            "test_opened": False,
            "held_ground_truth_opened": False,
        },
    )

    # Locally produced trusted tensor payload; frozen environment predates weights_only.
    payload = torch.load(paths["item_embeddings"], map_location="cpu")
    item_ids = [str(item) for item in payload["item_ids"]]
    embeddings = F.normalize(payload["embeddings"].float(), dim=1)
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    if len(item_to_idx) != len(item_ids):
        raise ValueError("Duplicate embedding item IDs")
    pseudo = read_set(paths["pseudo_cold_items"])
    real_cold = read_set(paths["real_cold_items"])
    sequences = read_train_sequences(paths["train_sequences"])
    train_x, train_y, calibration_x, calibration_y, data_report = build_examples(
        sequences,
        item_to_idx,
        embeddings,
        pseudo | real_cold,
        args.max_history,
        args.recency_decay,
        args.seed,
        args.calibration_modulus,
    )
    model, history = train_teacher(train_x, train_y, embeddings, args, device)
    temperatures = [float(value) for value in args.score_temperatures.split(",")]
    selected_temperature, calibration = calibrate_scores(
        model,
        calibration_x,
        calibration_y,
        embeddings,
        temperatures,
        args.candidate_size,
        args.batch_size,
        device,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dim": embeddings.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
            "item_ids_sha256": hashlib.sha256("\n".join(item_ids).encode()).hexdigest(),
        },
        output_dir / "resolver_item_disjoint.pt",
    )
    hashes = {role: sha256_file(path) for role, path in paths.items()}
    config = vars(args).copy()
    config.update(
        {
            "experiment_id": experiment_id,
            "selected_score_temperature": selected_temperature,
            "split": "student_readable_train_only",
            "test_opened": False,
            "held_ground_truth_opened": False,
        }
    )
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened_inputs": [
                {"role": role, "path": str(path), "sha256": hashes[role]}
                for role, path in paths.items()
            ],
            "held_ground_truth_opened": False,
            "test_opened": False,
        },
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": "PASS_ITEM_DISJOINT_TEACHER_READY",
        "data_report": data_report,
        "training_history": history,
        "calibration": calibration,
        "selected_score_temperature": selected_temperature,
        "test_opened": False,
        "held_ground_truth_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "student_readable_train_prefix_only": True,
            "pseudo_cold_interactions_used": False,
            "real_cold_interactions_used": False,
            "validation_or_test_labels_used": False,
            "held_ground_truth_used": False,
        },
    )
    atomic_json(
        output_dir / "status.json",
        {
            "experiment_id": experiment_id,
            "status": "completed",
            "stage": "finished",
            "reason": summary["verdict"],
            "automatic_retry": False,
            "test_opened": False,
            "held_ground_truth_opened": False,
            "summary_path": str((output_dir / "summary.json").resolve()),
        },
    )
    print(json.dumps({"verdict": summary["verdict"], "temperature": selected_temperature}))


if __name__ == "__main__":
    main()
