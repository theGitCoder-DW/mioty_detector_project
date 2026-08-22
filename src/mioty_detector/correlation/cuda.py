"""CUDA backend boundary for the final acceleration milestone.

Do not spread GPU code through the project. The final implementation belongs
here and must produce the same numerical output convention as cpu.py.
"""
from .base import CorrelationBackend

class CUDACorrelationBackend(CorrelationBackend):
    """Future CUDA/CuPy implementation of the correlation engine."""
    def __init__(self):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("Install the CUDA extra/CuPy before using the CUDA backend.") from exc
        self.cp = cp

    def correlate(self, received_iq, references):
        raise NotImplementedError("CUDA correlation is the final project milestone.")
