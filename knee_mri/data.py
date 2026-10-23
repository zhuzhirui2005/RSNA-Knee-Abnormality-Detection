"""Losslessly compressed rendered views and shared physical window sampling."""

from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import torch
import zstandard as zstd
from .sampling import window_rows


def cache_path(root, uid):
    return Path(root) / (hashlib.sha256(str(uid).encode()).hexdigest()[:24] + ".npz")


def write_cache(path, uid, item, side, side_check):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = item["px"].numpy()
    delta = np.diff(pixels, axis=2, prepend=np.zeros(pixels.shape[:2] + (1,), np.uint8))
    arrays = {
        "pixels_zstd": np.frombuffer(
            zstd.ZstdCompressor(level=3).compress(delta), np.uint8
        ),
        "pixel_shape": np.array(pixels.shape),
        "meta": item["meta"].numpy(),
        "blocks": np.array(json.dumps(item["blocks"])),
        "study_uid": np.array(uid),
        "side": np.array(json.dumps(side)),
        "recipe": np.array(
            json.dumps(
                {"size": int(pixels.shape[-1]), "views": "v4", "side_check": side_check}
            )
        ),
    }
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for name, value in arrays.items():
            with z.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, value, allow_pickle=False)
    tmp.replace(path)


def read_cache(path):
    with np.load(path, allow_pickle=False) as z:
        if "pixels" in z:
            pixels = z["pixels"]
        else:
            shape = tuple(z["pixel_shape"])
            raw = zstd.ZstdDecompressor().decompress(
                z["pixels_zstd"], max_output_size=int(np.prod(shape))
            )
            pixels = np.cumsum(
                np.frombuffer(raw, np.uint8).reshape(shape), axis=2, dtype=np.uint8
            )
        return pixels, z["meta"], json.loads(str(z["blocks"]))


def make_windows(pixels, metadata, blocks, *, training=False, seed=0, limit=176):
    blocks = [
        dict(b, views=[(0, view, start) for view, start in b["views"]]) for b in blocks
    ]
    rng = np.random.default_rng(seed) if training else None
    rows = window_rows(
        blocks,
        lambda f, r: metadata[r],
        "train" if training else "eval",
        3,
        eval_stride=1,
        cover_stride=3,
        phys_mm=3.5,
        rng=rng,
        train_sampling="cover",
    )
    if training and len(rows) > limit:
        rows = [rows[j] for j in np.sort(rng.choice(len(rows), limit, replace=False))]
    if not rows:
        raise ValueError("No usable physical windows")
    r = np.asarray(rows, np.int64)
    images = np.stack([pixels[r[:, 2 + d]] for d in range(3)], axis=1)
    m = metadata[r[:, 1]][:, [1, 3, 4, 5, 6, 7, 0, 2, 10]]
    return torch.from_numpy(images), torch.from_numpy(m.astype(np.float32))


class Studies(torch.utils.data.Dataset):
    def __init__(self, cache, uids, training=False, seed=0, epoch=0, limit=176):
        self.cache, self.uids = cache, list(uids)
        self.training, self.seed, self.epoch, self.limit = training, seed, epoch, limit

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        item = read_cache(cache_path(self.cache, self.uids[i]))
        images, metadata = make_windows(
            *item,
            training=self.training,
            seed=self.seed + 1000003 * self.epoch + i,
            limit=self.limit,
        )
        return images, metadata, i
