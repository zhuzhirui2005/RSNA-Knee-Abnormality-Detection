# Adapted from Heliosli's RSNA Knee pipeline. See LICENSE and README.md for attribution.
"""Fixed bilateral detail support when coronal medial/lateral identity is unknown.

These are screen-position views, never MCL masks or inferred patient sides.
The whole native field is always retained. Reference annotations do not enter
view selection. IDs reuse the existing global_grid embedding range (0..5).
"""

import numpy as np
from .input_provenance import view_geometry


def coronal_bilateral_views(native_shape, spacings, output_size, local_mm=100.0):
    """Return native loader (view ID, physical extent, (row,col) centre) tuples.

    Deduplication considers all actual per-slice spacings; a full-field-sized
    local crop is not duplicated, and two clamped identical sides become one.
    The caller applies this override only to Coronal and skips anatomic
    recentering for these views. Sagittal/Axial keep their original recipe.
    """
    spacings = np.asarray(spacings, np.float64)
    if spacings.ndim == 1:
        spacings = spacings[None]
    if spacings.ndim != 2 or spacings.shape[1] != 2 or (not len(spacings)):
        raise ValueError("in-plane spacings must be row/column pairs")
    if not np.isfinite(spacings).all() or (spacings <= 0).any():
        raise ValueError("bilateral physical views require known positive spacings")
    spacings = np.unique(spacings, axis=0)
    proposed = [
        (0, None, (0.5, 0.5)),
        (2, float(local_mm), (0.5, 0.25)),
        (3, float(local_mm), (0.5, 0.75)),
    ]
    selected = []
    seen = set()
    for view in proposed:
        _, crop, center = view
        signature = []
        for spacing in spacings:
            g = view_geometry(
                native_shape, spacing, output_size, crop_mm=crop, center=center
            )
            signature.append((g.crop_xyxy, g.valid_xyxy))
        signature = tuple(signature)
        if signature not in seen:
            selected.append(view)
            seen.add(signature)
    return selected


def rectangle_union_area(rectangles):
    """Exact continuous-edge area, including overlapping detail fields."""
    rectangles = np.asarray(rectangles, np.float64).reshape(-1, 4)
    rectangles = rectangles[np.all(rectangles[:, 2:] > rectangles[:, :2], axis=1)]
    if not len(rectangles):
        return 0.0
    xs = np.unique(rectangles[:, [0, 2]])
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = rectangles[(rectangles[:, 0] < right) & (rectangles[:, 2] > left)]
        intervals = sorted(map(tuple, active[:, [1, 3]]))
        length = 0.0
        end = -np.inf
        for start, stop in intervals:
            length += max(0.0, stop - max(start, end))
            end = max(end, stop)
        total += (right - left) * length
    return float(total)


def box_union_retention(box, rectangles):
    box = np.asarray(box, float)
    rectangles = np.asarray(rectangles, float).reshape(-1, 4)
    clipped = np.concatenate(
        [
            np.maximum(rectangles[:, :2], box[:2]),
            np.minimum(rectangles[:, 2:], box[2:]),
        ],
        axis=1,
    )
    return rectangle_union_area(clipped) / float(np.prod(box[2:] - box[:2]))
