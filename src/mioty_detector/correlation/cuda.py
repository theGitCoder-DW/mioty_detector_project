"""
CUDA correlation backend using CuPy.

This module is the GPU counterpart of correlation/cpu.py.

The CPU implementation uses:

    scipy.signal.correlate(..., method="fft")

Here we reproduce the same mathematical operation using:

    FFT(received)
        *
    FFT(conjugate(reverse(reference)))
        |
        v
      IFFT
        |
        v
    valid correlation

The important point is that the references are processed as a BATCH.

For our MIOTY detector this means:

    24 frequency references
             |
             v
       batched FFT
             |
             v
    24 correlation traces

rather than launching 24 independent Python-level GPU operations.
"""

import numpy as np


class CUDACorrelationBackend:
    """
    GPU correlation backend implemented with CuPy.

    The public interface intentionally matches CPUCorrelationBackend:

        correlate(received_iq, references) -> NumPy array

    Input:
        received_iq:
            np.ndarray, shape (N,), complex64

        references:
            np.ndarray, shape (M, L), complex64

    Output:
        np.ndarray, shape (M, N-L+1), float32
    """

    def __init__(self, device_id=0):
        """
        Initialize the CUDA backend.

        Parameters
        ----------
        device_id:
            CUDA device to use. Usually 0 if there is one GPU.
        """

        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CuPy is not installed.\n"
                "Install the appropriate CuPy CUDA package, "
                "for example:\n\n"
                "    pip install cupy-cuda12x\n"
            ) from exc

        self.cp = cp

        if not cp.is_available():
            raise RuntimeError(
                "CuPy is installed, but no CUDA-capable GPU "
                "is available."
            )

        self.device = cp.cuda.Device(device_id)
        self.device.use()

        print(
            "CUDA backend initialized: "
            
        )

    def correlate(
        self,
        received_iq,
        references,
    ):
        """
        Perform batched FFT-based correlation on the GPU.

        Parameters
        ----------
        received_iq:
            Complex input IQ stream.
            Shape: (N,)

        references:
            Frequency-shifted pilot references.
            Shape: (M, L)

        Returns
        -------
        np.ndarray
            Correlation magnitude.
            Shape: (M, N-L+1)

        Notes
        -----
        The returned array is copied back from GPU memory to CPU memory
        before returning, so the rest of the existing application can
        continue using NumPy without modification.
        """

        cp = self.cp

        # ------------------------------------------------------------
        # 1. Convert NumPy arrays to GPU arrays.
        #
        # This is one of the few CPU -> GPU transfers we perform.
        # ------------------------------------------------------------

        received_gpu = cp.asarray(
            received_iq,
            dtype=cp.complex64,
        )

        references_gpu = cp.asarray(
            references,
            dtype=cp.complex64,
        )

        n_samples = received_gpu.shape[0]
        n_reference = references_gpu.shape[1]
        n_references = references_gpu.shape[0]

        if n_reference > n_samples:
            raise ValueError(
                "Reference length cannot exceed "
                "received signal length."
            )

        # ------------------------------------------------------------
        # 2. Determine the FFT length.
        #
        # Linear correlation through FFT requires:
        #
        #       N + L - 1
        #
        # samples.
        #
        # A fast FFT length is preferable for performance.
        # ------------------------------------------------------------

        from cupyx.scipy.fft import next_fast_len

        fft_length = next_fast_len(
            n_samples + n_reference - 1
        )

        # ------------------------------------------------------------
        # 3. FFT of the received signal.
        #
        # Only ONE FFT is required for the received IQ stream.
        #
        # Shape:
        #
        #       (fft_length,)
        # ------------------------------------------------------------

        received_fft = cp.fft.fft(
            received_gpu,
            n=fft_length,
        )

        # ------------------------------------------------------------
        # 4. Prepare correlation filters.
        #
        # Correlation can be expressed as convolution with:
        #
        #       conjugate(reverse(reference))
        #
        # references_gpu has shape:
        #
        #       (M, L)
        #
        # where M = number of frequency hypotheses.
        # ------------------------------------------------------------

        correlation_filters = cp.conj(
            references_gpu[:, ::-1]
        )

        # ------------------------------------------------------------
        # 5. Batched FFT of all references.
        #
        # This is the important GPU operation.
        #
        # Instead of:
        #
        #   for reference in references:
        #       FFT(reference)
        #
        # we perform one batched FFT:
        #
        #       M × fft_length
        # ------------------------------------------------------------

        reference_fft = cp.fft.fft(
            correlation_filters,
            n=fft_length,
            axis=1,
        )

        # ------------------------------------------------------------
        # 6. Multiply in frequency domain.
        #
        # Broadcasting expands:
        #
        #       received_fft
        #
        # from:
        #
        #       (fft_length,)
        #
        # to:
        #
        #       (M, fft_length)
        #
        # without explicitly copying it M times.
        # ------------------------------------------------------------

        frequency_domain_product = (
            reference_fft
            * received_fft[None, :]
        )

        # ------------------------------------------------------------
        # 7. Inverse FFT.
        #
        # This gives the complete linear convolution for every
        # frequency reference.
        # ------------------------------------------------------------

        full_correlation = cp.fft.ifft(
            frequency_domain_product,
            axis=1,
        )

        # ------------------------------------------------------------
        # 8. Extract the VALID correlation region.
        #
        # For:
        #
        #       N = received length
        #       L = reference length
        #
        # valid correlation has:
        #
        #       N - L + 1
        #
        # samples.
        #
        # For convolution, valid begins at:
        #
        #       L - 1
        #
        # ------------------------------------------------------------

        valid_start = n_reference - 1
        valid_stop = valid_start + (
            n_samples - n_reference + 1
        )

        valid_correlation = full_correlation[
            :,
            valid_start:valid_stop,
        ]

        # ------------------------------------------------------------
        # 9. Calculate normalized magnitude.
        #
        # This mirrors the normalization in cpu.py.
        # ------------------------------------------------------------

        reference_energy = cp.sum(
            cp.abs(references_gpu) ** 2,
            axis=1,
        )

        normalization = cp.sqrt(
            reference_energy
        )

        correlation_magnitude = (
            cp.abs(valid_correlation)
            / normalization[:, None]
        )

        # ------------------------------------------------------------
        # 10. Transfer ONLY the final result back to CPU.
        #
        # The FFTs and correlation stay on the GPU.
        # ------------------------------------------------------------

        result = cp.asnumpy(
            correlation_magnitude
        )

        return result.astype(
            np.float32,
            copy=False,
        )

    def synchronize(self):
        """
        Wait until all queued CUDA work has completed.

        Useful for benchmarking.
        """

        self.cp.cuda.Stream.null.synchronize()

    def device_info(self):
        """Return basic information about the selected GPU."""

        return {
            "name": self.device.name,
            "device_id": self.device.id,
        }