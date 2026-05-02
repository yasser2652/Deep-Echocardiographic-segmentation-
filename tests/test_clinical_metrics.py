import math

import numpy as np

from src.clinical_metrics import estimate_area_based_volume_ml, estimate_biplane_area_length_volume_ml, lvef_percent


def test_lvef_formula():
    assert lvef_percent(100.0, 40.0) == 60.0
    assert math.isnan(lvef_percent(0.0, 10.0))


def test_volume_estimates_are_positive_for_nonempty_masks():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 12:22] = 1
    single = estimate_area_based_volume_ml(mask, spacing=(1.0, 1.0))
    biplane = estimate_biplane_area_length_volume_ml(mask, mask, spacing_2ch=(1.0, 1.0), spacing_4ch=(1.0, 1.0))
    assert single > 0
    assert biplane > 0

