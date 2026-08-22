"""MIOTY pilot/reference waveform generation.

This preserves the transmitter-derived GFSK reference used by the working
correlator. Keeping it isolated makes modulation changes safe and local.
"""
import numpy as np
from .config import BT, DEVIATION, OVERSAMPLING, PILOT_BITS, SYMBOL_RATE_UNITS

def bits_to_array(bit_string=PILOT_BITS):
    if any(b not in "01" for b in bit_string):
        raise ValueError("Pilot bits must contain only 0 and 1.")
    return np.array([int(b) for b in bit_string], dtype=np.float32)

def gaussian_filter_taps(oversampling=OVERSAMPLING, bt=BT):
    n_half = int(oversampling / bt)
    n = np.arange(-n_half, n_half + 1, dtype=np.float32)
    ta = np.float32(1.0 / oversampling)
    t = (n / oversampling).astype(np.float32)
    g = (np.sqrt(np.float32(2*np.pi)/np.float32(np.log(2))) * np.float32(bt) * ta *
         np.exp(-2*(np.float32(np.pi)*np.float32(bt)*t)**2/np.float32(np.log(2))))
    g = g.astype(np.float32)
    g /= g.sum()
    return g

def gfsk_modulate(bits, oversampling=OVERSAMPLING, deviation=DEVIATION,
                  symbol_rate_units=SYMBOL_RATE_UNITS, bt=BT,
                  random_phase=False, amplitude=1.0):
    """Generate the complex baseband reference used by the matched filter."""
    bits = np.asarray(bits, dtype=np.float32)
    modulation = np.where(bits > 0.5, np.float32(deviation), -np.float32(deviation))
    modulation = np.repeat(modulation, oversampling).astype(np.float32)
    taps = gaussian_filter_taps(oversampling, bt)
    pad = len(taps) // 2
    padded = np.pad(modulation, (pad, pad), mode="edge")
    filtered = np.convolve(padded, taps, mode="valid").astype(np.float32)[:len(modulation)]
    phase = np.random.uniform(0, 2*np.pi) if random_phase else 0.0
    phase_inc = filtered.astype(np.float64) * (2*np.pi) / (oversampling*np.float64(symbol_rate_units))
    phase_track = phase + np.cumsum(phase_inc)
    return (np.float32(amplitude) * np.exp(1j*phase_track)).astype(np.complex64)

def generate_pilot_reference(pilot_bits=PILOT_BITS, oversampling=OVERSAMPLING):
    """Generate the 12-symbol MIOTY pilot reference."""
    return gfsk_modulate(bits_to_array(pilot_bits), oversampling=oversampling)
