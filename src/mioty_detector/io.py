"""Raw IQ input helpers."""
from pathlib import Path
import numpy as np

test_data1 = "C:\Research_Project\mioty_detector_project_refactored\mioty_detector_project\test_data\MiotyFM_EU1.iq"
#test_data2 = "C:\Research_Project\mioty_detector_project_refactored\mioty_detector_project\test_data\MiotyFM_EU2.iq"

def load_iq_file(path, dtype=np.complex64):
    """Load a raw interleaved complex-float IQ capture."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=dtype)
