# Adapted from Heliosli's RSNA Knee pipeline. See LICENSE and README.md for attribution.
"""Image-based knee side check, shared by the training-cache fix and Kaggle inference.

The native cache stores coronal and axial series in a canonical frame where the study's side has already been applied:
lateral should sit at +column for both knees (a right knee is mirrored as the last pixel operation, see
training/dicom_geometry.flip_for_plane).  When the side is wrong, every coronal/axial image of the study is mirrored.
side_inputs() turns one canonical native volume into a few normalised 2-D images; the classifier says whether lateral is
at +column (correct orientation) or at -column (mirrored, i.e. the study's side is wrong)."""

import numpy as np
import cv2

SIDE_SIZE = 160
SIDE_FRACS = (0.4, 0.5, 0.6)
SIDE_PLANES = ("Coronal", "Axial")


def side_inputs(vol):
    """canonical native volume [n, H, W] (any dtype) -> uint8 [k, SIDE_SIZE, SIDE_SIZE], k <= 3 middle slices."""
    n = int(vol.shape[0])
    if n < 1:
        return np.zeros((0, SIDE_SIZE, SIDE_SIZE), np.uint8)
    idx = sorted({int(round(f * (n - 1))) for f in SIDE_FRACS})
    out = []
    for i in idx:
        im = np.asarray(vol[i], np.float32)
        lo, hi = np.percentile(im[::4, ::4], [1, 99])
        im = np.clip((im - lo) / max(float(hi - lo), 1e-6), 0, 1)
        im = cv2.resize(im, (SIDE_SIZE, SIDE_SIZE), interpolation=cv2.INTER_AREA)
        out.append(np.rint(im * 255).astype(np.uint8))
    return np.stack(out)


def build_side_net():
    import timm, torch.nn as nn

    net = timm.create_model("resnet34", pretrained=False, num_classes=1, in_chans=1)
    return net
