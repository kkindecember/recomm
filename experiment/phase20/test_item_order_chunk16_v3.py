"""Reuse objective checks; v3 adds a real-GPU cross-batch numerical audit."""
from experiment.phase20.test_item_order_chunk16 import (
    test_chunk16_and_chunk4_parameter_gradients_match,
    test_gradient_calibration_is_consistent_across_chunks,
)


def test_formal_training_and_evaluation_unchanged():
    import inspect
    from experiment.phase20 import item_order_transfer as old,item_order_transfer_chunk16_v3 as new
    for method in ('train_arm','evaluate','backward','load_cache'):
        assert inspect.getsource(getattr(old.OrderWork,method))==inspect.getsource(getattr(new.OrderWork,method))
