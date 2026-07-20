"""
Runner module for GRAM
Handles training and evaluation execution
"""

from .base import BaseRunner
from .single_runner_gram import SingleRunnerGRAM
from .distributed_runner_gram import DistributedRunnerGRAM


# Factory function to get appropriate runner
def get_runner(
    runner_type,
    model_rec,
    model_gen,
    tokenizer,
    train_loader_id,
    train_loader_rec,
    valid_loader,
    device,
    args,
    rank=0,
):
    """
    Factory function to get the appropriate runner based on type

    Args:
        runner_type (str): 'single', or 'distributed'
        ... other args for runner initialization

    Returns:
        BaseRunner: Appropriate runner instance
    """
    if runner_type == "single":
        return SingleRunnerGRAM(
            model_rec,
            model_gen,
            tokenizer,
            train_loader_id,
            train_loader_rec,
            valid_loader,
            device,
            args,
        )
    elif runner_type == "distributed":
        return DistributedRunnerGRAM(
            model_rec,
            model_gen,
            tokenizer,
            train_loader_id,
            train_loader_rec,
            valid_loader,
            device,
            args,
            rank,
        )
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")


__all__ = ["BaseRunner", "SingleRunnerGRAM", "DistributedRunnerGRAM", "get_runner"]
