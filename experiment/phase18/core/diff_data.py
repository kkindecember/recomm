"""Original GRAM train/validation protocol, with train-fit attributes."""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_mapping(path):
    pairs = [line.strip().split(" ", 1) for line in path.read_text().splitlines() if line.strip()]
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def fields(text):
    result = {}
    for fragment in text.split(";"):
        if ":" in fragment:
            name, value = fragment.split(":", 1)
            if name.strip() in {"categories", "brand"}:
                result[name.strip()] = value.strip()
    attributes = []
    for name in ("categories", "brand"):
        value = result.get(name, "")
        values = value.split(",") if name == "categories" else [value]
        attributes.append(sorted({v.strip() for v in values
                                  if v.strip().lower() not in {"", "na", "n/a", "none", "unknown"}}))
    return attributes


def native_collator(root, config):
    sys.path.insert(0, str(root / "GRAM/src"))
    from processor import CollatorGRAM
    from experiment.phase17.core.full_latte_gram_backend import load_gram_tokenizer
    args = SimpleNamespace(**json.loads((root / config["original_config"]).read_text()))
    tokenizer = load_gram_tokenizer(root, "G0_GRAM_B0_FRESH")
    return tokenizer, CollatorGRAM(tokenizer, args=args)


def baseline_reference(root, config):
    if "baseline_csv" in config:
        with (root / config["baseline_csv"]).open() as handle:
            row = next(row for row in csv.DictReader(handle) if row["epoch"] == str(config.get("parent_epoch", 30)))
        return ({"hit@10": float(row["validation_recall_at_10"]),
                 "ndcg@10": float(row["validation_ndcg_at_10"])}, row["sha256"], [root / config["baseline_csv"]])
    selection = json.loads((root / config["baseline_metrics_json"]).read_text())
    if (selection["dataset"] != config["dataset"] or selection["checkpoint_epoch"] != config["parent_epoch"]
            or selection["checkpoint"] != config["parent_checkpoint"]
            or selection["selection_metric"] != "validation_NDCG@10"):
        raise AssertionError("Historical baseline selection mismatch")
    log = (root / config["baseline_training_log"]).read_text()
    pattern = r"(?m)^(?:Start training recommender for phase 1, epoch (\d+)|validation ((?:hit|ndcg)@10): ([0-9.eE+-]+))$"
    epoch, values = None, {}
    for match in re.finditer(pattern, log):
        if match[1] is not None:
            epoch = int(match[1])
        elif epoch == config["parent_epoch"]:
            values[match[2]] = float(match[3])
    if set(values) != {"hit@10", "ndcg@10"} or values["ndcg@10"] != selection["selection_metric_value"]:
        raise AssertionError("Could not verify selected validation metrics against historical log")
    return values, selection["checkpoint_sha256"], [root / config["baseline_metrics_json"], root / config["baseline_training_log"]]


class DiffDataset(Dataset):
    def __init__(self, records, catalog):
        self.records, self.catalog = records, catalog

    @classmethod
    def load(cls, directory, split, catalog):
        if split not in {"train", "validation"}:
            raise ValueError("This screening runner supports train/validation only")
        records = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]
        return cls(records, catalog)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, catalog = self.records[index], self.catalog
        history = row["history"][::-1]
        lex = catalog["lexical_ids"]
        prompt = "What would user purchase after " + " ; ".join(lex[i] for i in history) + " ?"
        return {"input": [prompt] + [catalog["passages"][i] for i in history],
                "output": lex[row["target"]], "user_id": row["user_id"],
                "history_item_ids": history, "target_item_id": row["target"]}


def prepare(root, config, directory):
    if directory.exists():
        raise FileExistsError(f"Refusing to replace prepared data: {directory}")
    sys.path.insert(0, str(root / "GRAM/src"))
    from utils.indexing import gram_indexing
    from data.multi_task_dataset_gram import MultiTaskDatasetGRAM

    args = SimpleNamespace(**json.loads((root / config["original_config"]).read_text()))
    dataset_name = config.get("dataset", "Toys")
    if args.datasets != dataset_name:
        raise AssertionError("Original config dataset mismatch")
    args.data_path = str(root / "GRAM/rec_datasets")
    sequences, passages, lexical = gram_indexing(args.data_path, dataset_name, None, None,
                                                regenerate=False, args=args, id_linking=True)
    names = [""] + sorted(lexical)
    numeric = {name: index for index, name in enumerate(names) if index}
    train, validation, train_items = [], [], set()
    for user, sequence in sequences.items():
        if len(sequence) < 4:
            raise ValueError("Expected the original nonempty leave-one-out histories")
        prefix = [numeric[item] for item in sequence[:-2]]
        train_items.update(prefix)
        for position in range(1, len(prefix)):
            train.append({"user_id": user, "history": prefix[max(0, position - 20):position],
                          "target": prefix[position]})
        validation.append({"user_id": user, "history": prefix[-20:], "target": numeric[sequence[-2]]})

    data_directory = root / "GRAM/rec_datasets" / dataset_name
    metadata = read_mapping(data_directory / "item_plain_text.txt")
    item_attributes = [([], [])] + [fields(metadata[name]) for name in names[1:]]
    vocabularies, tables, missing = [], [], []
    feature_names = ("categories", "brand")
    for feature_name in config["attribute_features"]:
        feature = feature_names.index(feature_name)
        vocabulary = {value: index + 2 for index, value in enumerate(sorted(
            {value for item in train_items for value in item_attributes[item][feature]}))}
        if len(vocabulary) < 2:
            raise ValueError(f"Attribute {feature_name} has no usable variation")
        rows = [[0]] + [[vocabulary.get(value, 1) for value in attrs[feature]] or [1]
                        for attrs in item_attributes[1:]]
        width = max(map(len, rows))
        tables.append([row + [0] * (width - len(row)) for row in rows])
        vocabularies.append(vocabulary)
        missing.append(sum(row == [1] for row in rows[1:]))
    catalog = {"items": names, "lexical_ids": [""] + [lexical[name] for name in names[1:]],
               "passages": [""] + [passages[name] for name in names[1:]],
               "attribute_tables": tables, "attribute_sizes": [len(v) + 2 for v in vocabularies],
               "attribute_vocabularies": vocabularies, "attribute_features": config["attribute_features"]}

    tokenizer, collator = native_collator(root, config)
    paths = [[0] + [t for t in tokenizer.encode(lexical[name]) if t not in (1820, 9175)]
             for name in names[1:]]
    if len({tuple(path) for path in paths}) != len(paths) or max(map(len, paths)) > 33:
        raise AssertionError("Lexical decoder catalog must be unique and untruncated")
    catalog["decoder_paths"] = paths

    # Exercise the native sample construction, not a second handwritten template.
    selected_users = set(list(sequences)[:5] + list(sequences)[-1:])
    selected_users.add(max(sequences, key=lambda user: len(sequences[user])))
    parity = {}
    for split, records in (("train", train), ("validation", validation)):
        native = MultiTaskDatasetGRAM.__new__(MultiTaskDatasetGRAM)
        native.args, native.rank, native.mode = args, 0, split
        native.dataset, native.max_his, native.his_sep = dataset_name, 20, " ; "
        native.skip_empty_his, native.reverse_history = 1, 1
        native.user_seq_dict = {u: seq for u, seq in sequences.items() if u in selected_users}
        native.item2input, native.item2lexid, native.item2cfid = passages, lexical, numeric
        native.item2views = {item: (value,) for item, value in lexical.items()}
        native.data_samples = native.load_train() if split == "train" else native.load_validation()
        native.construct_sentence()
        adapter = DiffDataset([row for row in records if row["user_id"] in selected_users], catalog)
        if len(native) != len(adapter) or any(native[i] != adapter[i] for i in range(len(native))):
            raise AssertionError(f"Native {split} rendering mismatch")
        for index in (0, len(adapter) - 1):
            left, right = collator([native[index]]), collator([adapter[index]])
            for key in ("item_text_ids", "item_text_masks", "target_ids", "history_item_ids"):
                if not torch.equal(left[key], right[key]):
                    raise AssertionError(f"Native {split} tokenization mismatch: {key}")
        parity[split] = len(adapter)

    baseline, baseline_hash, baseline_sources = baseline_reference(root, config)
    parent_hash = sha256(root / config["parent_checkpoint"])
    if parent_hash != config["parent_sha256"] or baseline_hash != parent_hash:
        raise AssertionError("Historical parent checkpoint mismatch")
    source_paths = sorted(data_directory.glob("*.txt"))
    source_paths += baseline_sources + [root / config["original_config"]]
    manifest = {"state": "PREPARED", "dataset": dataset_name, "split": "original leave-one-out train/official validation",
                "train_examples": len(train), "validation_examples": len(validation),
                "catalog_items": len(names) - 1, "train_items": len(train_items),
                "attribute_sizes": catalog["attribute_sizes"], "unknown_or_missing_items": missing,
                "attribute_features": config["attribute_features"],
                "raw_missing_brand_items": sum(not attrs[1] for attrs in item_attributes[1:]),
                "native_render_parity_examples": parity, "parent_sha256": parent_hash,
                "baseline_validation": baseline,
                "sources": {str(path.relative_to(root)): sha256(path) for path in source_paths},
                "test_examples_constructed": 0}
    directory.mkdir(parents=True)
    write_json(directory / "catalog.json", catalog)
    for split, records in (("train", train), ("validation", validation)):
        with (directory / f"{split}.jsonl").open("w") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["prepared_files"] = {path.name: sha256(path) for path in sorted(directory.iterdir())}
    write_json(directory / "manifest.json", manifest)
    return manifest
