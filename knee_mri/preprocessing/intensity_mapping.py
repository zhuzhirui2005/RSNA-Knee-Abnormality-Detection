# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""Fixed per-Series dynamic-range mappings for native encoded MRI inputs.

The companded candidate keeps 88--90% of the original percentile-window slope
while allocating output range to the stored intensity tails. It is continuous
and strictly increasing over native uint16 values. No labels or per-slice
statistics enter this construction. Classification benefit is not established.
"""

from __future__ import annotations
import numpy as np

POLICIES = ("series_percentile", "native_full_range", "series_companded")
POLICY_IDS = {name: i for i, name in enumerate(POLICIES)}


def curve_knots(window, policy="series_percentile"):
    if policy not in POLICY_IDS:
        raise ValueError("unknown intensity mapping policy")
    lo, hi = map(float, window)
    if not np.isfinite([lo, hi]).all():
        raise ValueError("nonfinite display window")
    if policy == "native_full_range":
        return np.array([[0.0, 0.0], [65535.0, 1.0]], np.float64)
    if policy == "series_percentile":
        upper = lo + max(hi - lo, 1.0)
        rows = [(lo, 0.0), (upper, 1.0)]
        if lo > 0:
            rows.insert(0, (0.0, 0.0))
        if upper < 65535:
            rows.append((65535.0, 1.0))
        return np.asarray(rows, np.float64)
    if not 0 <= lo <= hi <= 65535:
        raise ValueError("companding requires a valid uint16 window")
    if hi - lo < 1:
        return np.array([[0.0, 0.0], [65535.0, 1.0]], np.float64)
    rows = [(0.0, 0.0)]
    if lo > 0:
        rows.append((lo, 0.02))
    rows.append((hi, 0.9 if hi < 65535 else 1.0))
    if hi < 65535:
        rows.append((65535.0, 1.0))
    return np.asarray(rows, np.float64)


def normalize_intensity(resized, window, policy="series_percentile"):
    """Use after the same physical crop and INTER_AREA resize as production."""
    if policy == "series_percentile":
        low, high = window
        return np.clip((resized - low) / max(high - low, 1.0), 0, 1)
    knots = curve_knots(window, policy)
    return np.interp(resized, knots[:, 0], knots[:, 1]).astype(np.float32)


def mapping_receipt(window, policy="series_percentile"):
    knots = curve_knots(window, policy)
    packed = np.zeros((4, 2), np.float64)
    packed[: len(knots)] = knots
    return {
        "intensity_curve_knots": packed,
        "intensity_curve_count": np.int64(len(knots)),
        "intensity_mapping_id": np.int64(POLICY_IDS[policy]),
    }


def augment_intensity(values, gain, policy="legacy"):
    """Torch input augmentation; shared across a real acquisition's windows.

    g*x/(1+(g-1)*x) maps [0,1] to [0,1], fixes the endpoints, and has
    derivative g/(1+(g-1)*x)^2 > 0 for g>0. Unlike hard clipping after gain,
    it preserves the high-tail ordering. This is not a claim of AUC benefit.
    """
    if policy == "legacy":
        return (values * gain).clamp(0, 1)
    if policy != "bounded_monotone":
        raise ValueError("unknown intensity augmentation policy")
    return gain * values / (1 + (gain - 1) * values)
