from __future__ import annotations

import os
import inspect
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from genrecedit_gram_adapter import (  # noqa: E402
    OneOneDeltaRouter,
    OneOneGenerationDeltaContext,
    SecondMomentAccumulator,
    accumulate_probe_predictions,
    build_positionwise_requests,
    legal_next_token_ids,
    legal_target_state,
    merge_probe_counts,
    probe_accuracy_from_counts,
    select_probe_layers,
    select_branching_positionwise_smoke_requests,
    select_positionwise_smoke_requests,
    solve_closed_form_delta,
    validate_request_universe,
    validate_delta_shapes,
    validate_position_layer_selection,
    edited_parameter_name,
)


class TestGenRecEditGramAdapter(unittest.TestCase):
    def test_streaming_second_moment_and_parameter_name(self):
        accumulator = SecondMomentAccumulator(2)
        accumulator.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        accumulator.update(torch.tensor([[5.0, 6.0]]))
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = values.double().T @ values.double() / 3
        self.assertTrue(torch.allclose(accumulator.moment(), expected))
        self.assertTrue(
            torch.allclose(accumulator.moment(ridge=0.5), expected + 0.5 * torch.eye(2))
        )
        self.assertEqual(
            edited_parameter_name(4),
            "decoder.block.4.layer.2.DenseReluDense.wo.weight",
        )

    def test_requests_cover_every_variable_length_token_without_eos(self):
        requests = build_positionwise_requests(
            cold_paths={"c1": ("a", "b"), "c2": ("a", "d", "e")},
            pseudo_contexts={
                "c1": [("w1", ("h1", "h2"))],
                "c2": [("w2", ("h3",))],
            },
        )
        self.assertEqual(len(requests), 5)
        self.assertEqual(
            [(row.cold_item, row.position, row.prefix_tokens, row.target_token) for row in requests],
            [
                ("c1", 0, (), "a"),
                ("c1", 1, ("a",), "b"),
                ("c2", 0, (), "a"),
                ("c2", 1, ("a",), "d"),
                ("c2", 2, ("a", "d"), "e"),
            ],
        )

    def test_missing_cold_universe_or_context_hard_fails(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            build_positionwise_requests(
                cold_paths={"c1": ("a",), "c2": ("b",)},
                pseudo_contexts={"c1": [("w", ("h",))]},
            )
        with self.assertRaisesRegex(ValueError, "no train-only"):
            build_positionwise_requests(
                cold_paths={"c1": ("a",)}, pseudo_contexts={"c1": []}
            )

    def test_full_request_universe_and_deterministic_position_sample(self):
        cold_paths = {"c1": ("a", "b"), "c2": ("a", "d", "e")}
        requests = build_positionwise_requests(
            cold_paths=cold_paths,
            pseudo_contexts={
                "c1": [("w1", ("h1",)), ("w2", ("h2",))],
                "c2": [("w3", ("h3",)), ("w4", ("h4",))],
            },
        )
        self.assertEqual(
            validate_request_universe(
                requests=requests, cold_paths=cold_paths, contexts_per_cold=2
            ),
            {0: 4, 1: 4, 2: 2},
        )
        first = select_positionwise_smoke_requests(
            requests, requests_per_position=1, seed=1502
        )
        second = select_positionwise_smoke_requests(
            list(reversed(requests)), requests_per_position=1, seed=1502
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {0, 1, 2})

    def test_lexical_legal_children_and_target_state(self):
        paths = {"i1": (4, 5), "i2": (4, 6), "i3": (7,)}
        self.assertEqual(legal_next_token_ids(paths, (4,)), (5, 6))
        is_best, probability = legal_target_state(
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0]),
            target_token_id=5,
            legal_token_ids=(5, 6),
        )
        self.assertTrue(is_best)
        self.assertGreater(probability, 0.8)
        with self.assertRaisesRegex(ValueError, "not a legal"):
            legal_target_state(
                torch.zeros(8), target_token_id=7, legal_token_ids=(5, 6)
            )

    def test_branching_request_selection_excludes_structurally_solved_prefixes(self):
        catalog = {
            "c1": ("a", "x"),
            "c2": ("a", "y"),
            "c3": ("b", "z"),
        }
        requests = build_positionwise_requests(
            cold_paths=catalog,
            pseudo_contexts={item: [("warm", ("history",))] for item in catalog},
        )
        selected = select_branching_positionwise_smoke_requests(
            requests,
            catalog_paths=catalog,
            requests_per_position=1,
            seed=1502,
        )
        self.assertEqual(set(selected), {0, 1})
        self.assertNotEqual(selected[1][0].cold_item, "c3")
        for position, rows in selected.items():
            prefix = rows[0].prefix_tokens
            children = {
                path[position]
                for path in catalog.values()
                if path[:position] == prefix
            }
            self.assertGreaterEqual(len(children), 2)

    def test_branching_request_selection_is_input_order_independent(self):
        catalog = {
            "c1": ("a", "x"),
            "c2": ("a", "y"),
            "c3": ("b", "x"),
            "c4": ("b", "y"),
        }
        requests = build_positionwise_requests(
            cold_paths=catalog,
            pseudo_contexts={item: [("warm", ("history",))] for item in catalog},
        )
        kwargs = dict(catalog_paths=catalog, requests_per_position=2, seed=1502)
        self.assertEqual(
            select_branching_positionwise_smoke_requests(requests, **kwargs),
            select_branching_positionwise_smoke_requests(list(reversed(requests)), **kwargs),
        )
    def test_probe_selection_and_position_map_are_complete(self):
        cold_paths = {"c1": ("a", "b"), "c2": ("a", "b", "c")}
        selected = select_probe_layers(
            {
                0: {0: 0.8, 1: 0.8, 2: 0.7},
                1: {0: 0.6, 1: 0.9, 2: 0.8},
                2: {0: 0.5, 1: 0.4, 2: 0.7},
            },
            decoder_layers=3,
        )
        self.assertEqual(selected, {0: 0, 1: 1, 2: 2})
        self.assertEqual(
            validate_position_layer_selection(
                cold_paths=cold_paths, position_to_layer=selected, decoder_layers=3
            ),
            selected,
        )

    def test_closed_form_delta_and_shape_contract(self):
        residual = torch.tensor([[2.0]])
        keys = torch.tensor([[1.0]])
        covariance = torch.tensor([[1.0]])
        delta = solve_closed_form_delta(
            residual=residual,
            keys=keys,
            covariance=covariance,
            preservation_lambda=1.0,
        )
        self.assertTrue(torch.allclose(delta, torch.tensor([[1.0]])))
        name = "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        validate_delta_shapes(
            base_parameters={name: torch.zeros(2, 3)},
            deltas_by_position={0: {name: torch.ones(2, 3)}},
            position_to_layer={0: 0},
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_delta_shapes(
                base_parameters={name: torch.zeros(2, 3)},
                deltas_by_position={0: {name: torch.ones(3, 2)}},
                position_to_layer={0: 0},
            )

    def test_one_one_router_activates_only_current_position(self):
        p0 = "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        p1 = "decoder.block.1.layer.2.DenseReluDense.wo.weight"
        router = OneOneDeltaRouter(
            deltas_by_position={
                0: {p0: torch.ones(1)},
                1: {p1: torch.full((1,), 2.0)},
            },
            position_to_layer={0: 0, 1: 1},
        )
        base = torch.zeros(1)
        self.assertTrue(torch.equal(router.materialize_parameter(p0, base, 0), torch.ones(1)))
        self.assertTrue(torch.equal(router.materialize_parameter(p1, base, 0), base))
        self.assertIs(router.materialize_parameter(p0, base, 0, is_eos=True), base)
        self.assertIs(router.materialize_parameter(p0, base, 0, is_padding=True), base)

    def test_generation_context_applies_only_to_active_lexical_rows(self):
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
        context = OneOneGenerationDeltaContext(
            model=model,
            deltas_by_position={
                0: {name: torch.eye(2)},
                1: {name: 2 * torch.eye(2)},
            },
            position_to_layer={0: 0, 1: 0},
            encoded_catalog_paths=[(10,), (11, 12)],
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
        hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        with context:
            self.assertIn(
                "encoder_outputs",
                inspect.signature(model.prepare_inputs_for_generation).parameters,
            )
            model.prepare_inputs_for_generation(torch.tensor([[0], [0]]))
            first = model.decoder.block[0].layer[2].DenseReluDense.wo(hidden)
            self.assertTrue(torch.equal(first, hidden))

            # Prefix (10,) is a complete path and must route to EOS without
            # editing. Prefix (11,) remains at lexical position 1.
            model.prepare_inputs_for_generation(torch.tensor([[0, 10], [0, 11]]))
            second = model.decoder.block[0].layer[2].DenseReluDense.wo(hidden)
            self.assertTrue(torch.equal(second[0], torch.zeros_like(second[0])))
            self.assertTrue(torch.equal(second[1], 2 * hidden[1]))

            # A constrained beam search may retain -inf rows to fill
            # num_beams when a trie level has too few legal children.  Such a
            # dead row must remain unedited while a live lexical row is routed.
            model.prepare_inputs_for_generation(torch.tensor([[0, 99], [0, 11]]))
            third = model.decoder.block[0].layer[2].DenseReluDense.wo(hidden)
            self.assertTrue(torch.equal(third[0], torch.zeros_like(third[0])))
            self.assertTrue(torch.equal(third[1], 2 * hidden[1]))

        restored = model.decoder.block[0].layer[2].DenseReluDense.wo(hidden)
        self.assertTrue(torch.equal(restored, torch.zeros_like(restored)))
        self.assertEqual(context.applied_rows_by_position, {0: 2, 1: 2})
        self.assertEqual(context.dead_prefix_rows, 1)

    def test_train_only_probe_counts_exclude_eos_and_padding(self):
        labels = torch.tensor([[4, 5, 1], [4, 1, -100]])
        predictions = {
            0: torch.tensor([[4, 0, 1], [0, 1, 0]]),
            1: torch.tensor([[4, 5, 0], [4, 0, 0]]),
        }
        update = accumulate_probe_predictions(
            predictions_by_layer=predictions,
            labels=labels,
            eos_token_id=1,
        )
        counts = {}
        merge_probe_counts(counts, update)
        accuracy = probe_accuracy_from_counts(counts, decoder_layers=2)
        self.assertEqual(counts[0], {0: [1, 2], 1: [2, 2]})
        self.assertEqual(counts[1], {0: [0, 1], 1: [1, 1]})
        self.assertEqual(accuracy, {0: {0: 0.5, 1: 1.0}, 1: {0: 0.0, 1: 1.0}})


if __name__ == "__main__":
    unittest.main()
