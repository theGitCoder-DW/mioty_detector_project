import numpy as np
from mioty_detector.pilot import generate_pilot_reference
from mioty_detector.detector import correlate_single_frequency

def test_detection():
    ref=generate_pilot_reference(); offset=750; rng=np.random.default_rng(1)
    x=((rng.normal(size=4000)+1j*rng.normal(size=4000))*0.01).astype(np.complex64)
    x[offset:offset+len(ref)]+=ref
    assert int(np.argmax(correlate_single_frequency(x,ref)))==offset
