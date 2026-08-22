"""CPU golden/reference implementation of the matched filter."""
import numpy as np
from scipy.signal import correlate
from .base import CorrelationBackend

class CPUCorrelationBackend(CorrelationBackend):
    """FFT-based SciPy correlation; keep this as the CUDA correctness baseline."""
    def correlate(self, received_iq, references):
        received_iq = np.asarray(received_iq, dtype=np.complex64)
        references = np.asarray(references, dtype=np.complex64)
        rows = []
        for reference in references:
            corr = correlate(received_iq, reference, mode="valid", method="fft")
            energy = np.sum(np.abs(reference)**2)
            rows.append(np.abs(corr) / np.sqrt(energy))
        return np.vstack(rows).astype(np.float32)
