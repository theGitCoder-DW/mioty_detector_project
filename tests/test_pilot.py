from mioty_detector.config import PILOT_BITS, OVERSAMPLING
from mioty_detector.pilot import bits_to_array, generate_pilot_reference
import numpy as np

def test_pilot():
    assert len(bits_to_array(PILOT_BITS)) == 12
    ref=generate_pilot_reference()
    assert len(ref)==12*OVERSAMPLING
    assert ref.dtype==np.complex64
