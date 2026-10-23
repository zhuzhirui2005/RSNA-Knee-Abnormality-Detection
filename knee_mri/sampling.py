# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""Physical window sampling, shared by training and inference."""

import math
import torch


def covering_centers(n, phase=1, stride=3):
    if stride <= 1:
        return list(range(n))
    c = list(range(min(phase, n - 1), n, stride))
    if c[0] > 1:
        c.insert(0, 0)
    if c[-1] < n - 2:
        c.append(n - 1)
    return c


def phys_step(spacing_mm, unit_mm):
    """slices per ~unit_mm of anatomy: 1 for an ordinary 3-4 mm series, ~9 for a 0.4 mm 3-D stack (unit_mm <= 0: always 1)."""
    if unit_mm <= 0:
        return 1
    return max(1, int(round(unit_mm / max(float(spacing_mm), 0.1))))


def window_rows(
    blocks,
    meta_of,
    mode,
    slices,
    eval_stride=3,
    cover_stride=3,
    phys_mm=0.0,
    rng=None,
    slice_jitter=0.0,
    train_sampling="cover",
):
    """The window list of one study.  blocks: [{'n_slices': n, 'views': [(file, view_id, start), ...]}]; meta_of(file, row)
    -> the px meta row of that window (column 5 = log1p(slice spacing in mm)).  Returns tuples
    (file, centre row, row of slice -r, ..., row of slice +r).  'train' covers every (series, view) at stride
    cover_stride x phys_step with a random phase (and takes the K neighbours two steps apart with probability slice_jitter);
    'eval' covers at stride eval_stride x phys_step.  phys_mm = 0 reproduces the old per-slice behaviour."""
    cand = []
    r_ = slices // 2
    for b in blocks:
        n = b["n_slices"]
        for f, view_id, start in b["views"]:
            ps = phys_step(math.expm1(float(meta_of(f, start)[5])), phys_mm)
            if mode == "train":
                if train_sampling == "cover":
                    st = cover_stride * ps
                    ks = covering_centers(n, int(rng.integers(0, st)), st)
                else:
                    ks = range(n)
            else:
                ks = covering_centers(n, 1, eval_stride * ps)
            for k in ks:
                step_ = ps * (
                    2
                    if (
                        mode == "train"
                        and slice_jitter > 0
                        and rng.random() < slice_jitter
                    )
                    else 1
                )
                cand.append(
                    (f, start + k)
                    + tuple(
                        start + min(n - 1, max(0, k + d * step_))
                        for d in range(-r_, r_ + 1)
                    )
                )
    return cand


def collate(batch):
    W = max(len(b[0]) for b in batch)
    S = batch[0][0].shape[-1]
    img = torch.zeros((len(batch), W, batch[0][0].shape[1], S, S), dtype=torch.uint8)
    meta = torch.zeros((len(batch), W, batch[0][1].shape[1]), dtype=torch.float32)
    mask = torch.zeros((len(batch), W), dtype=torch.bool)
    for j, (x, m, _) in enumerate(batch):
        img[j, : len(x)] = x
        meta[j, : len(x)] = m
        mask[j, : len(x)] = True
    return img, meta, mask, torch.tensor([b[2] for b in batch])
