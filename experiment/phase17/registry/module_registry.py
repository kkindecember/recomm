"""One registry for all migration modules; unknown flags fail closed."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, TypeVar

from torch import nn

from experiment.phase17.core.feature_hooks import FeatureHook, FeatureHookChain
from experiment.phase17.core.generation_hooks import GenerationHookChain, GenerationScoreHook
from experiment.phase17.core.loss_hooks import AuxiliaryLossHook, DecoderLossHook, LossHookChain
from experiment.phase17.core.p0_modules import (
    BearSurvivalDecoderLoss,
    BiFlowFeatureHook,
    PrefixCurriculumDecoderLoss,
    ShortcutFiDFeatureHook,
    TransitionTeacherFeatureHook,
)
from experiment.phase17.core.p1_modules import (
    ContextRootPromptFeatureHook,
    LogitConcentrationAuxiliaryLoss,
    LongShortFiDFeatureHook,
    MaskedHistoryFeatureHook,
    OneWayBridgeFeatureHook,
    PawaLiteDecoderLoss,
    TokenSetAuxiliaryLoss,
    TreeContrastiveAuxiliaryLoss,
)
from experiment.phase17.core.runtime import MigrationRuntime


T = TypeVar("T")


def _construct(factory: Callable[..., T], config) -> T:
    """Support legacy zero-argument factories and config-aware S17 factories."""

    parameters = inspect.signature(factory).parameters
    return factory() if len(parameters) == 0 else factory(config)


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    track_id: str
    feature_factory: Callable[..., FeatureHook] | None = None
    auxiliary_factory: Callable[..., AuxiliaryLossHook] | None = None
    auxiliary_weight: float = 0.0
    decoder_factory: Callable[..., DecoderLossHook] | None = None
    generation_factory: Callable[..., GenerationScoreHook] | None = None
    item_aggregation: str | None = None


class ModuleRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModuleSpec] = {}

    def register(self, spec: ModuleSpec) -> None:
        if spec.module_id in self._specs:
            raise KeyError(f"duplicate migration module: {spec.module_id}")
        self._specs[spec.module_id] = spec

    def build(self, enabled: list[str] | tuple[str, ...], config=None) -> MigrationRuntime:
        unknown = sorted(set(enabled) - self._specs.keys())
        if unknown:
            raise KeyError(f"unknown migration modules: {unknown}")
        features: list[FeatureHook] = []
        auxiliary: list[tuple[str, float, AuxiliaryLossHook]] = []
        decoder: DecoderLossHook | None = None
        generation: list[GenerationScoreHook] = []
        aggregation = "logsumexp"
        for module_id in enabled:
            spec = self._specs[module_id]
            if spec.feature_factory:
                hook = _construct(spec.feature_factory, config)
                hook._s17_module_id = module_id
                features.append(hook)
            if spec.auxiliary_factory:
                auxiliary_hook = _construct(spec.auxiliary_factory, config)
                auxiliary_hook._s17_module_id = module_id
                auxiliary.append(
                    (
                        module_id,
                        spec.auxiliary_weight,
                        auxiliary_hook,
                    )
                )
            if spec.decoder_factory:
                if decoder is not None:
                    raise ValueError("only one decoder-loss replacement may be active")
                decoder = _construct(spec.decoder_factory, config)
                decoder._s17_module_id = module_id
            if spec.generation_factory:
                generation.append(_construct(spec.generation_factory, config))
            if spec.item_aggregation:
                if aggregation != "logsumexp" and aggregation != spec.item_aggregation:
                    raise ValueError("conflicting item aggregation policies")
                aggregation = spec.item_aggregation
        return MigrationRuntime(
            enabled_modules=tuple(enabled),
            feature_hooks=FeatureHookChain(features),
            loss_hooks=LossHookChain(decoder=decoder, auxiliary=auxiliary),
            generation_hooks=GenerationHookChain(generation),
            item_aggregation=aggregation,
        )


REGISTRY = ModuleRegistry()


REGISTRY.register(
    ModuleSpec("A0_bear", "A0", decoder_factory=lambda: BearSurvivalDecoderLoss())
)

# S17-4 P1-lite contracts.  These IDs are deliberately separate from the P0
# names so that a P1 result cannot silently overwrite the interpretation of an
# earlier mechanism.
REGISTRY.register(
    ModuleSpec("P1_pawa_lite", "P1-A", decoder_factory=lambda: PawaLiteDecoderLoss())
)
REGISTRY.register(
    ModuleSpec(
        "P1_treecl_lite",
        "P1-A",
        auxiliary_factory=lambda: TreeContrastiveAuxiliaryLoss(),
        auxiliary_weight=0.05,
    )
)
REGISTRY.register(
    ModuleSpec("P1_pctx_root", "P1-B", feature_factory=lambda: ContextRootPromptFeatureHook())
)
REGISTRY.register(
    ModuleSpec(
        "P1_sethead",
        "P1-B",
        auxiliary_factory=lambda: TokenSetAuxiliaryLoss(),
        auxiliary_weight=0.10,
    )
)
REGISTRY.register(
    ModuleSpec("P1_ls_fid", "P1-C", feature_factory=lambda: LongShortFiDFeatureHook())
)
REGISTRY.register(
    ModuleSpec("P1_mhm", "P1-C", feature_factory=lambda: MaskedHistoryFeatureHook())
)
REGISTRY.register(
    ModuleSpec(
        "P1_graphmae_prompt",
        "P1-D",
        feature_factory=lambda config: TransitionTeacherFeatureHook(
            d_model=int(config.d_model),
            transition_map=getattr(config, "s17_transition_map", ""),
        ),
    )
)
REGISTRY.register(
    ModuleSpec(
        "P1_dcrec_cl",
        "P1-D",
        feature_factory=lambda config: TransitionTeacherFeatureHook(
            d_model=int(config.d_model),
            transition_map=getattr(config, "s17_transition_map", ""),
        ),
    )
)
REGISTRY.register(
    ModuleSpec(
        "P1_sprint",
        "P1-E",
        auxiliary_factory=lambda: LogitConcentrationAuxiliaryLoss(),
        auxiliary_weight=0.01,
    )
)
REGISTRY.register(
    ModuleSpec(
        "P1_biflow_s2g",
        "P1-C0",
        feature_factory=lambda: OneWayBridgeFeatureHook("sequence_to_global"),
    )
)
REGISTRY.register(
    ModuleSpec(
        "P1_biflow_g2s",
        "P1-C0",
        feature_factory=lambda: OneWayBridgeFeatureHook("global_to_sequence"),
    )
)
REGISTRY.register(
    ModuleSpec(
        "A0_bear_proxy",
        "A0",
        decoder_factory=lambda: BearSurvivalDecoderLoss(),
    )
)
REGISTRY.register(
    ModuleSpec(
        "A1_prefixcurr",
        "A1",
        decoder_factory=lambda: PrefixCurriculumDecoderLoss(),
    )
)
REGISTRY.register(
    ModuleSpec("B0_mvi", "B0", item_aggregation="logsumexp")
)
REGISTRY.register(
    ModuleSpec("B1_latte", "B1", item_aggregation="logsumexp")
)
REGISTRY.register(
    ModuleSpec("C0_biflow", "C0", feature_factory=lambda: BiFlowFeatureHook())
)
REGISTRY.register(
    ModuleSpec(
        "D0_ted",
        "D0",
        feature_factory=lambda config: TransitionTeacherFeatureHook(
            d_model=int(config.d_model),
            transition_map=getattr(config, "s17_transition_map", ""),
        ),
    )
)
REGISTRY.register(
    ModuleSpec(
        "E0_shortcut_fid",
        "E0",
        feature_factory=lambda: ShortcutFiDFeatureHook(),
    )
)
REGISTRY.register(
    ModuleSpec(
        "E0_shortcut_fid_full_control",
        "E0",
        feature_factory=lambda: ShortcutFiDFeatureHook(selection_mode="full"),
    )
)
REGISTRY.register(
    ModuleSpec(
        "E0_shortcut_fid_random_control",
        "E0",
        feature_factory=lambda: ShortcutFiDFeatureHook(selection_mode="random_same_size"),
    )
)


def build_runtime(
    enabled: str | list[str] | tuple[str, ...] | None = None, config=None
) -> MigrationRuntime:
    if enabled is None:
        names: list[str] = []
    elif isinstance(enabled, str):
        names = [name.strip() for name in enabled.split(",") if name.strip()]
    else:
        names = list(enabled)
    return REGISTRY.build(names, config=config)
