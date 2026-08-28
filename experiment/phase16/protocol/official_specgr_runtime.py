#!/usr/bin/env python3
"""Load pinned official SpecGR UniSRec with pinned RecBole layer code.

RecBole's package initializer eagerly imports optional logging/experiment
dependencies that are irrelevant to ``recbole.model.layers``.  This bootstrap
loads the pinned enum and layers modules directly, without copying or modifying
third-party source.  The executed TransformerEncoder and SpecGR UniSRec classes
therefore retain their original source files and hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SPECGR_ROOT = ROOT / ".runtime/phase15_sources/SpecGR"
RECBOLE_ROOT = ROOT / ".runtime/phase16_sources/RecBole"
SPECGR_COMMIT = "f0ded8884b1df97b5f0599d4ec300bb20b5d1eff"
RECBOLE_COMMIT = "362d31f00af801d7d99bc635c902d1df1405e79d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def verify_sources() -> dict[str, Any]:
    sources = {
        "SpecGR": (SPECGR_ROOT, SPECGR_COMMIT),
        "RecBole": (RECBOLE_ROOT, RECBOLE_COMMIT),
    }
    records: dict[str, Any] = {}
    for name, (repo, expected) in sources.items():
        actual = git(repo, "rev-parse", "HEAD")
        status = git(repo, "status", "--short")
        if actual != expected or status:
            raise RuntimeError(f"Pinned {name} source drift: commit={actual}, status={status!r}")
        records[name] = {"path": str(repo.relative_to(ROOT)), "commit": actual, "worktree_clean": True}
    records["official_files"] = {
        str((RECBOLE_ROOT / "recbole/model/layers.py").relative_to(ROOT)): sha256(
            RECBOLE_ROOT / "recbole/model/layers.py"
        ),
        str((RECBOLE_ROOT / "recbole/utils/enum_type.py").relative_to(ROOT)): sha256(
            RECBOLE_ROOT / "recbole/utils/enum_type.py"
        ),
        str((SPECGR_ROOT / "models/draft/UniSRec/model.py").relative_to(ROOT)): sha256(
            SPECGR_ROOT / "models/draft/UniSRec/model.py"
        ),
        str((SPECGR_ROOT / "models/draft/UniSRec/layers.py").relative_to(ROOT)): sha256(
            SPECGR_ROOT / "models/draft/UniSRec/layers.py"
        ),
    }
    return records


def _load_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_official_unisrec_class():
    verify_sources()
    for name in tuple(sys.modules):
        if name == "recbole" or name.startswith("recbole."):
            del sys.modules[name]

    recbole = types.ModuleType("recbole")
    recbole.__path__ = [str(RECBOLE_ROOT / "recbole")]
    sys.modules["recbole"] = recbole

    enum_module = _load_file(
        "recbole.utils.enum_type", RECBOLE_ROOT / "recbole/utils/enum_type.py"
    )
    utils_module = types.ModuleType("recbole.utils")
    utils_module.__path__ = [str(RECBOLE_ROOT / "recbole/utils")]
    utils_module.FeatureType = enum_module.FeatureType
    utils_module.FeatureSource = enum_module.FeatureSource
    sys.modules["recbole.utils"] = utils_module

    model_package = types.ModuleType("recbole.model")
    model_package.__path__ = [str(RECBOLE_ROOT / "recbole/model")]
    sys.modules["recbole.model"] = model_package
    official_layers = _load_file(
        "recbole.model.layers", RECBOLE_ROOT / "recbole/model/layers.py"
    )

    if str(SPECGR_ROOT) not in sys.path:
        sys.path.insert(0, str(SPECGR_ROOT))
    from models.draft.UniSRec.model import UniSRec

    transformer_file = Path(inspect.getsourcefile(official_layers.TransformerEncoder) or "").resolve()
    unisrec_file = Path(inspect.getsourcefile(UniSRec) or "").resolve()
    if transformer_file != (RECBOLE_ROOT / "recbole/model/layers.py").resolve():
        raise RuntimeError(f"TransformerEncoder did not come from pinned RecBole: {transformer_file}")
    if unisrec_file != (SPECGR_ROOT / "models/draft/UniSRec/model.py").resolve():
        raise RuntimeError(f"UniSRec did not come from pinned SpecGR: {unisrec_file}")
    return UniSRec


def official_unisrec_config(input_dimension: int) -> dict[str, Any]:
    """Official base values with the sole content-interface width adaptation."""
    return {
        "max_seq_length": 20,
        "n_layers": 2,
        "n_heads": 2,
        "hidden_size": 300,
        "inner_size": 256,
        "hidden_dropout_prob": 0.5,
        "attn_dropout_prob": 0.5,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "loss_type": "CE",
        "lambda": 1e-3,
        "train_stage": "inductive_ft",
        "plm_size": input_dimension,
        "adaptor_dropout_prob": 0.2,
        "adaptor_layers": [input_dimension, 300],
        "temperature": 0.07,
        "n_exps": 8,
    }


if __name__ == "__main__":
    klass = load_official_unisrec_class()
    print(f"OFFICIAL_UNISREC_RUNTIME_READY {klass.__module__}.{klass.__name__}")
