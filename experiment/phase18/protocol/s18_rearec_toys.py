"""Bind the frozen ReaRec protocol to Toys in an isolated module instance.

The Beauty adapter and protocol remain byte-identical to their admitted profile.
Both datasets execute the same training, validation and budget implementation.
"""
import importlib.util

from experiment.phase18.core import rearec_toys_adapter as a

LOCAL_SOURCES = (
    'experiment/phase18/core/rearec_adapter.py',
    'experiment/phase18/core/rearec_toys_adapter.py',
    'experiment/phase18/protocol/s18_rearec.py',
    'experiment/phase18/protocol/s18_rearec_toys.py',
    'experiment/phase18/run_stage18_rearec_toys.sh',
    'experiment/phase18/analysis/s18_rearec_toys_inputs.py',
)


def main():
    spec = importlib.util.spec_from_file_location(
        '_s18_rearec_toys_protocol', a.ROOT / 'experiment/phase18/protocol/s18_rearec.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    protocol.a = a
    protocol.LOCAL_SOURCES = LOCAL_SOURCES
    protocol.main()


if __name__ == '__main__':
    main()
