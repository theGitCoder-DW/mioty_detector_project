"""Signal-processing orchestration above the correlation backend.

This module decides what to correlate, how to search frequency, and how to
turn correlation surfaces into burst detections. It contains no plotting or
CUDA-specific code.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from .config import (CHANNEL_SPACING_HZ, FREQUENCY_SEARCH_STEP_HZ,
                     NUM_EU1_CHANNELS, OVERSAMPLING, PILOT_BITS)
from .pilot import generate_pilot_reference

@dataclass(frozen=True)
class BurstDetection:
    sample_index: int
    frequency_hz: float
    magnitude: float

def generate_frequency_grid(
    num_channels=NUM_EU1_CHANNELS,
    channel_spacing_hz=CHANNEL_SPACING_HZ,
):
    """Generate the 24 EU1 channel frequency hypotheses.

    These are the frequencies used for the primary 2-D correlation.

    The resulting correlation surface has the form:

        frequency channel × time

    i.e. approximately:

        24 × number_of_time_samples

    Fine frequency resolution, such as 25 Hz, is intentionally NOT applied
    here. It should be used later only around promising correlation peaks.
    """

    centers = (
        np.arange(num_channels)
        - (num_channels - 1) / 2.0
    ) * channel_spacing_hz

    return centers.astype(np.float64)

def generate_frequency_shifted_references(
    pilot_bits=PILOT_BITS,
    sample_rate=None,
    frequencies_hz=None,
    oversampling=OVERSAMPLING,
):
    """Create the reference bank for the primary 24-channel search."""

    if sample_rate is None:
        raise ValueError(
            "sample_rate is required."
        )

    if frequencies_hz is None:
        frequencies_hz = generate_frequency_grid()

    base_reference = generate_pilot_reference(
        pilot_bits,
        oversampling=oversampling,
    )

    n = len(base_reference)

    time = (
        np.arange(n, dtype=np.float64)
        / sample_rate
    )

    references = np.empty(
        (
            len(frequencies_hz),
            n,
        ),
        dtype=np.complex64,
    )

    for row, frequency in enumerate(
        frequencies_hz
    ):
        references[row] = (
            base_reference
            * np.exp(
                1j
                * 2
                * np.pi
                * frequency
                * time
            )
        ).astype(np.complex64)

    return (
        np.asarray(
            frequencies_hz,
            dtype=np.float64,
        ),
        references,
    )

def adaptive_threshold(surface, sigma_multiplier=8.0):
    """Robust median + MAD threshold for the correlation surface."""
    x = np.asarray(surface).ravel()
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return float(median + sigma_multiplier*1.4826*mad)

def find_bursts_2d(frequency_hz, correlation_surface, threshold,
                   min_spacing=None, pilot_bits=PILOT_BITS, oversampling=OVERSAMPLING):
    """Find local maxima in every frequency row."""
    if min_spacing is None:
        min_spacing = len(pilot_bits)*oversampling
    detections = []
    for row, f in enumerate(frequency_hz):
        corr = correlation_surface[row]
        indices, _ = find_peaks(corr, height=threshold, distance=min_spacing)
        detections.extend(BurstDetection(int(i), float(f), float(corr[i])) for i in indices)
    return sorted(detections, key=lambda d: d.sample_index)

def strongest_detection(detections):
    return max(detections, key=lambda d: d.magnitude) if detections else None

def correlate_single_frequency(received_iq, reference_iq):
    """Small convenience wrapper around the CPU reference backend."""
    from .correlation.cpu import CPUCorrelationBackend
    return CPUCorrelationBackend().correlate(received_iq, np.asarray([reference_iq]))[0]
