"""Backend interface for the expensive correlation operation."""
from abc import ABC, abstractmethod

class CorrelationBackend(ABC):
    """Common interface shared by CPU and future CUDA implementations."""
    @abstractmethod
    def correlate(self, received_iq, references):
        """Return correlation magnitudes with shape (references, time)."""
        raise NotImplementedError
