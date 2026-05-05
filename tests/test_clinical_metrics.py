import math

import numpy as np

from src.clinical_metrics import (
    estimate_area_based_volume_ml,
    estimate_biplane_area_length_volume_ml,
    estimate_biplane_simpson_volume_ml,
    estimate_patient_volumes,
    lvef_percent,
)


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
    simpson = estimate_biplane_simpson_volume_ml(mask, mask, spacing_2ch=(1.0, 1.0), spacing_4ch=(1.0, 1.0))
    assert simpson > 0


def _rect_mask(size=64, top=16, bottom=48, left=22, right=42):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def test_patient_volume_sanity_passes_for_larger_ed_than_es():
    ed = _rect_mask(top=12, bottom=52, left=20, right=44)
    es = _rect_mask(top=20, bottom=44, left=25, right=39)
    volumes = estimate_patient_volumes(
        {("2CH", "ED"): ed, ("4CH", "ED"): ed, ("2CH", "ES"): es, ("4CH", "ES"): es},
        {("2CH", "ED"): (1.0, 1.0), ("4CH", "ED"): (1.0, 1.0), ("2CH", "ES"): (1.0, 1.0), ("4CH", "ES"): (1.0, 1.0)},
    )
    assert volumes["LVEDV"] > volumes["LVESV"]
    assert 0 <= volumes["LVEF"] <= 90


def test_patient_volume_warns_for_empty_and_missing_masks():
    empty = np.zeros((32, 32), dtype=np.uint8)
    volumes = estimate_patient_volumes({("2CH", "ED"): empty})
    warnings_text = volumes["clinical_warnings_text"]
    assert "Missing required" in warnings_text
    assert "Empty LV cavity" in warnings_text or "No LV mask" in warnings_text


def test_patient_volume_warns_when_lvedv_not_greater_than_lvesv():
    ed = _rect_mask(top=20, bottom=44, left=25, right=39)
    es = _rect_mask(top=12, bottom=52, left=20, right=44)
    volumes = estimate_patient_volumes(
        {("2CH", "ED"): ed, ("4CH", "ED"): ed, ("2CH", "ES"): es, ("4CH", "ES"): es},
        {("2CH", "ED"): (1.0, 1.0), ("4CH", "ED"): (1.0, 1.0), ("2CH", "ES"): (1.0, 1.0), ("4CH", "ES"): (1.0, 1.0)},
    )
    assert "LVEDV is not greater than LVESV" in volumes["clinical_warnings_text"]
