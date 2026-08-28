from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "experiment" / "phase16" / "protocol"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROTOCOL))

from genrecedit_faithful import (  # noqa: E402
    AdmissionDiagnostics,
    CachedZObservation,
    PositionAdmissionDiagnostics,
    OneOneGenerationDeltaContext,
    ZForwardBatch,
    ZOptimizationConfig,
    aggregate_updates,
    assert_base_parameter_parity,
    batch_full_target_requests,
    build_one_one_position_bundles,
    build_full_target_requests,
    clip_delta_norm_,
    collect_covariance,
    extract_keys,
    filter_valid_z,
    linear_system_diagnostics,
    optimize_z_vectors,
    official_position_to_layer,
    probe_cached_z,
    snapshot_base_parameters,
    solve_weight_delta,
    try_cache_hits,
    update_z_lifecycle,
    validate_admission_diagnostics,
)


def _requests():
    return build_full_target_requests(
        catalog_paths={"c1": (2, 4), "c2": (2, 5, 6), "warm_catalog": (2, 3)},
        cold_paths={"c1": (2, 4), "c2": (2, 5, 6)},
        pseudo_contexts={
            "c1": [("w1", ("h1",)), ("w2", ("h2",))],
            "c2": [("w3", ("h3",)), ("w4", ("h4",))],
        },
        eos_token_id=1,
        pad_token_id=0,
    )


class GenRecEditFaithfulTests(unittest.TestCase):
    def test_full_target_requests_and_batches_are_complete(self) -> None:
        requests = _requests()
        self.assertEqual(len(requests), 10)
        self.assertEqual(
            {position: sum(map(len, batches)) for position, batches in batch_full_target_requests(requests, batch_size=3).items()},
            {0: 4, 1: 4, 2: 2},
        )
        self.assertEqual(requests[0].prefix_token_ids, ())
        self.assertEqual(requests[0].legal_token_ids, (2,))
        self.assertEqual(requests[1].prefix_token_ids, (2,))
        self.assertEqual(requests[1].legal_token_ids, (3, 4, 5))
        stronger_warm_competitor = torch.tensor([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0])
        full_catalog_probe = probe_cached_z(
            stronger_warm_competitor,
            target_token_id=requests[1].target_token_id,
            legal_token_ids=requests[1].legal_token_ids,
        )
        self.assertFalse(full_catalog_probe.legal_argmax)
        self.assertEqual(full_catalog_probe.legal_rank, 2)
        with self.assertRaisesRegex(ValueError, "EOS and padding"):
            build_full_target_requests(
                catalog_paths={"bad": (2, 1)},
                cold_paths={"bad": (2, 1)},
                pseudo_contexts={"bad": [("w", ("h",))]},
                eos_token_id=1,
                pad_token_id=0,
            )

    def test_threshold_is_cache_only_not_optimizer_success(self) -> None:
        request = _requests()[1]
        logits = torch.tensor([0.99, 0.98, 0.0, 0.97, 1.0, 0.9, 0.96, 0.95])
        probe = probe_cached_z(
            logits,
            target_token_id=request.target_token_id,
            legal_token_ids=request.legal_token_ids,
            probability_threshold=0.3,
        )
        self.assertTrue(probe.legal_argmax)
        self.assertLess(probe.full_vocabulary_probability, 0.3)
        self.assertFalse(probe.cache_hit)
        lifecycle = update_z_lifecycle(
            logits=logits.unsqueeze(0), requests=[request], active_indices=[0]
        )
        self.assertEqual(lifecycle.satisfied_indices, (0,))
        self.assertEqual(lifecycle.active_indices, ())

    def test_cache_probe_uses_first_passing_absolute_z(self) -> None:
        request = _requests()[1]
        candidates = [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 4.0])]
        cache = {(1, str(request.target_token_id), request.position): candidates}

        def forward(_request, z, _layer):
            if float(z[0]) < 2.0:
                logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 2.0, 0.0])
            else:
                logits = torch.tensor([-4.0, -4.0, -4.0, -4.0, 4.0, 1.0, -4.0])
            return CachedZObservation(logits=logits, target_init=torch.ones(2))

        hits = try_cache_hits(
            requests=[request], target_layer=1, z_cache=cache, probe_forward=forward
        )
        self.assertEqual(tuple(hits), (0,))
        self.assertTrue(torch.equal(hits[0].z_vector, candidates[1]))
        self.assertTrue(torch.equal(hits[0].delta_vector, torch.tensor([2.0, 3.0])))
        self.assertTrue(hits[0].probe.cache_hit)

    def test_official_adam_cosine_lifecycle_and_failed_z_count(self) -> None:
        rows = [request for request in _requests() if request.position == 1][:2]
        # Both rows target token 4.  The first remains a legal argmax, while
        # the second callback row is deliberately routed to legal token 5.
        self.assertEqual([row.target_token_id for row in rows], [4, 4])

        call_count = 0

        def forward(batch, deltas, active):
            nonlocal call_count
            call_count += 1
            logits = torch.zeros(len(batch), 7, device=deltas.device) + deltas[:, :1] * 0.0
            logits[0, 4] = logits[0, 4] + 2.0
            logits[0, 5] = logits[0, 5] + 1.0
            logits[1, 4] = logits[1, 4] + 1.0
            logits[1, 5] = logits[1, 5] + 2.0
            return ZForwardBatch(logits=logits, target_inits=torch.ones_like(deltas))

        config = ZOptimizationConfig(
            v_lr=0.5,
            v_num_grad_steps=3,
            v_weight_decay=0.2,
            z_vector_max=8000.0,
            eta_min=0.01,
            batch_size=2,
        )
        result = optimize_z_vectors(
            requests=rows,
            vector_dimension=2,
            device="cpu",
            forward_batch=forward,
            config=config,
        )
        self.assertEqual(result.optimizer_satisfied_indices, (0,))
        self.assertEqual(result.failed_indices, (1,))
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.failed_count, 1)
        self.assertEqual(result.lifecycle_check_steps_by_batch, ((1, 2),))
        expected_lr_1 = 0.01 + (0.5 - 0.01) * (1 + math.cos(math.pi / 3)) / 2
        expected_lr_2 = 0.01 + (0.5 - 0.01) * (1 + math.cos(2 * math.pi / 3)) / 2
        self.assertAlmostEqual(result.scheduler_lrs_by_batch[0][0], expected_lr_1)
        self.assertAlmostEqual(result.scheduler_lrs_by_batch[0][1], expected_lr_2)
        self.assertEqual(len(result.scheduler_lrs_by_batch[0]), 3)
        self.assertEqual(call_count, 3)

    def test_absolute_norm_clip(self) -> None:
        delta = torch.tensor([3.0, 4.0])
        self.assertTrue(clip_delta_norm_(delta, 2.0))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(delta)), 2.0, places=6)
        small = torch.tensor([0.5, 0.5])
        self.assertFalse(clip_delta_norm_(small, 2.0))
        self.assertTrue(torch.equal(small, torch.tensor([0.5, 0.5])))

    def test_position_wise_second_moment_uses_official_sample_cap(self) -> None:
        p0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]])
        p1 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        result = collect_covariance({0: p0, 1: p1}, mom2_n_samples=2)
        self.assertEqual(result.available_rows_by_position, {0: 3, 1: 2})
        self.assertEqual(result.used_rows_by_position, {0: 2, 1: 2})
        self.assertTrue(torch.allclose(result.covariance_by_position[0], p0[:2].double().T @ p0[:2].double() / 2))
        self.assertTrue(torch.allclose(result.covariance_by_position[1], p1.double().T @ p1.double() / 2))

    def test_key_extraction_uses_last_decoder_position(self) -> None:
        rows = [request for request in _requests() if request.position == 1]
        module = torch.nn.Linear(2, 3, bias=False)
        batches_seen = []

        def forward(batch):
            batches_seen.append(len(batch))
            base = len(batches_seen) * 10
            values = torch.tensor(
                [[[base + i, base + i + 1], [base + i + 2, base + i + 3]] for i in range(len(batch))],
                dtype=torch.float32,
            )
            module(values)

        keys = extract_keys(module=module, requests=rows, forward_batch=forward, batch_size=3)
        self.assertEqual(batches_seen, [3, 1])
        self.assertTrue(
            torch.equal(
                keys,
                torch.tensor([[12.0, 13.0], [13.0, 14.0], [14.0, 15.0], [22.0, 23.0]]),
            )
        )

    def test_valid_z_failed_count_and_closed_form_solve(self) -> None:
        z = [torch.tensor([1.0, 2.0]), None, torch.tensor([3.0, 4.0])]
        deltas = [torch.tensor([0.5, 1.0]), None, torch.tensor([1.5, 2.0])]
        selection = filter_valid_z(z, deltas)
        self.assertEqual(selection.valid_indices, (0, 2))
        self.assertEqual(selection.failed_indices, (1,))
        self.assertEqual(selection.valid_count, 2)
        self.assertEqual(selection.failed_count, 1)

        residuals = torch.stack(selection.delta_vectors)
        keys = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        covariance = torch.tensor([[3.0, 0.5], [0.5, 2.0]])
        actual = solve_weight_delta(
            residuals=residuals,
            keys=keys,
            covariance=covariance,
            covariance_lambda=4.0,
        )
        expected = (residuals.double().T @ keys.double()) @ torch.linalg.inv(
            keys.double().T @ keys.double() + 4.0 * covariance.double()
        )
        self.assertTrue(torch.allclose(actual.double(), expected))

    def test_additive_aggregation_shares_parameters_across_positions(self) -> None:
        shared = "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        other = "decoder.block.1.layer.2.DenseReluDense.wo.weight"
        aggregated = aggregate_updates(
            {
                0: {shared: torch.ones(2, 2)},
                4: {shared: 2 * torch.ones(2, 2)},
                1: {other: 4 * torch.ones(2, 2)},
            }
        )
        self.assertTrue(torch.equal(aggregated[shared], 3 * torch.ones(2, 2)))
        self.assertTrue(torch.equal(aggregated[other], 4 * torch.ones(2, 2)))

        routing = official_position_to_layer(range(6))
        self.assertEqual(routing, {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1})
        bundles = build_one_one_position_bundles(
            position_to_layer={0: routing[0], 4: routing[4]},
            aggregated_updates=aggregated,
        )
        self.assertIs(bundles[0][shared], bundles[4][shared])

    def test_admission_diagnostics_cover_required_evidence(self) -> None:
        weight_delta = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        keys = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        covariance = torch.eye(2)
        linear = linear_system_diagnostics(
            parameter_name="decoder.block.0.layer.2.DenseReluDense.wo.weight",
            contributing_positions=(0, 4),
            weight_delta=weight_delta,
            keys=keys,
            covariance=covariance,
            covariance_lambda=4.0,
        )
        diagnostics = AdmissionDiagnostics(
            per_position={
                0: PositionAdmissionDiagnostics(
                    position=0,
                    request_count=2,
                    cache_hit_count=0,
                    valid_z_count=1,
                    failed_z_count=1,
                    full_vocabulary_target_probabilities=(0.2, 0.4),
                    legal_target_ranks=(1, 2),
                )
            },
            linear_systems=(linear,),
            unedited_parity={"base_parameters_exact": True, "base_output_exact": True},
            warm_preservation={"warm_h50_delta": -0.01, "warm_rows": 20},
        )
        audit = validate_admission_diagnostics(diagnostics)
        self.assertEqual(audit["request_count"], 2)
        self.assertEqual(audit["cache_hit_count"], 0)
        self.assertEqual(linear.delta_rank, 2)
        self.assertGreaterEqual(linear.system_condition, 1.0)

    def test_one_one_trigger_restores_exact_base_and_base_output(self) -> None:
        class Dense(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wo = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.wo.weight.zero_()

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.DenseReluDense = Dense()

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.ModuleList(
                    [torch.nn.Identity(), torch.nn.Identity(), Layer()]
                )

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Module()
                self.decoder.block = torch.nn.ModuleList([Block()])

            def prepare_inputs_for_generation(self, decoder_input_ids, **kwargs):
                return {"decoder_input_ids": decoder_input_ids, **kwargs}

        model = FakeModel()
        name = "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        module = model.decoder.block[0].layer[2].DenseReluDense.wo
        hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        baseline = module(hidden).clone()
        snapshot = snapshot_base_parameters(model, [name])
        aggregated = aggregate_updates(
            {0: {name: torch.eye(2)}, 4: {name: 2 * torch.eye(2)}}
        )
        routing = official_position_to_layer((0, 4))
        position_bundles = build_one_one_position_bundles(
            position_to_layer=routing, aggregated_updates=aggregated
        )
        self.assertIs(position_bundles[0][name], position_bundles[4][name])
        context = OneOneGenerationDeltaContext(
            model=model,
            deltas_by_position=position_bundles,
            position_to_layer=routing,
            encoded_catalog_paths=[(10, 11, 12, 13), (20, 21, 22, 23, 24)],
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
        with context:
            model.prepare_inputs_for_generation(torch.tensor([[0], [0]]))
            self.assertTrue(torch.equal(module(hidden), 3 * hidden))
            model.prepare_inputs_for_generation(
                torch.tensor([[0, 10, 11, 12, 13], [0, 20, 21, 22, 23]])
            )
            triggered = module(hidden)
            self.assertTrue(torch.equal(triggered[0], baseline[0]))
            self.assertTrue(torch.equal(triggered[1], 3 * hidden[1]))
            self.assertTrue(assert_base_parameter_parity(model, snapshot)["exact"])

        self.assertTrue(assert_base_parameter_parity(model, snapshot)["exact"])
        self.assertTrue(torch.equal(module(hidden), baseline))
        self.assertEqual(context.applied_rows_by_position, {0: 2, 4: 1})


if __name__ == "__main__":
    unittest.main()
