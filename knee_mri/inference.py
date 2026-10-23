"""Study-level inference. Fold logits are averaged before probability blending."""
import contextlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from .constants import LABELS, UID
from .data import cache_path, read_cache, write_cache, make_windows, Studies
from .checkpoints import load_model
from .preprocessing.reader import DicomReader


def autocast(device, precision):
    if str(device).startswith('cuda') and precision != 'fp32':
        return torch.autocast('cuda', dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[precision])
    return contextlib.nullcontext()


def predict_windows(model, images, metadata, device, crop, chunk, precision):
    offset = (images.shape[-1] - crop) // 2
    if offset < 0:
        raise ValueError('Crop is larger than the rendered views')
    images = images[..., offset:offset+crop, offset:offset+crop]
    # Encode small batches on the GPU; keep all unencoded uint8 windows on CPU.
    # Attention is applied once over the complete study, never per chunk.
    tokens = []
    with torch.inference_mode(), autocast(device, precision):
        for start in range(0, len(images), chunk):
            x = images[start:start+chunk].to(device).float().div_(255.)
            n = len(x)
            if n % 32:
                x = torch.cat([x, x.new_zeros((32-n%32, *x.shape[1:]))])
            x = ((x-model.mean)/model.std).contiguous(memory_format=torch.channels_last)
            tokens.append(model.spatial_pool(model.enc(x))[:n])
        meta = metadata.unsqueeze(0).to(device)
        mask = torch.ones((1, len(images)), dtype=torch.bool, device=device)
        return model.head(torch.cat(tokens), meta, mask).float().cpu().numpy()[0]


def probabilities_from_logits(logits):
    # Averaging logits and averaging probabilities are not interchangeable.
    z = np.asarray(logits, dtype=np.float32).mean(axis=0)
    return torch.sigmoid(torch.from_numpy(z)).numpy()


def publish_predictions(output, uids, probabilities):
    p = np.asarray(probabilities)
    if p.shape != (len(uids), len(LABELS)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Incomplete or invalid prediction matrix; no output published')
    frame = pd.DataFrame(p, columns=LABELS)
    frame.insert(0, UID, uids)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(output)


def make_reader(data, split, size, weights, temporary, device, side_check):
    return DicomReader(data, split, [], temporary, '', size, views='v4', slices=3, phys_mm=3.5,
        anchor_weights=Path(weights)/'anatomy.pt',
        side_model=Path(weights)/'side_cls.pt' if side_check else None, device=device)


def prepare(data, split, weights, cache, size=384, device='cpu', side_check=True, ids=None):
    uids = pd.read_csv(Path(data)/f'{split}.csv', dtype={UID: str})[UID].tolist()
    if ids is not None: uids = [u for u in uids if u in set(ids)]
    with tempfile.TemporaryDirectory(prefix='knee-prepare-') as tmp:
        reader = make_reader(data, split, size, weights, tmp, device, side_check)
        for i, uid in enumerate(uids):
            path = cache_path(cache, uid)
            if not path.exists():
                item, side = reader.read(uid)
                write_cache(path, uid, item, side, side_check)
            else:
                with np.load(path, allow_pickle=False) as z:
                    expected = {'size': size, 'views': 'v4', 'side_check': side_check}
                    if 'recipe' not in z or json.loads(str(z['recipe'])) != expected:
                        raise ValueError('Cache recipe differs; use a separate cache directory')
            print(f'{i+1}/{len(uids)} {uid}', flush=True)


def predict_cached(checkpoint, cache, uids, crop=352, device='cuda:0', chunk=64, precision='fp16'):
    model = load_model(checkpoint, crop, device)
    ds = Studies(cache, uids)
    rows = [predict_windows(model, *ds[i][:2], device, crop, chunk, precision) for i in range(len(ds))]
    return torch.sigmoid(torch.from_numpy(np.stack(rows))).numpy()


def predict(data, split, weights, output, device='cuda:0', chunk=64, precision='fp16', branch='ensemble', cache=None):
    uids = pd.read_csv(Path(data)/f'{split}.csv', dtype={UID: str})[UID].tolist()
    if not uids or len(set(uids)) != len(uids): raise ValueError('Expected distinct study identifiers')
    paths = Path(weights)
    specifications = [('llm', 384, 352, True, .45), ('distilled', 384, 352, True, .45), ('public', 320, 288, False, .10)]
    if branch != 'ensemble': specifications = [(*x[:4], 1.) for x in specifications if x[0] == branch]
    totals = np.zeros((len(uids), len(LABELS)), np.float64)
    # Process one model at a time; a persistent cache avoids repeated DICOM decoding.
    with tempfile.TemporaryDirectory(prefix='knee-predict-') as temporary:
        cache_root = Path(cache or (Path(temporary)/'cache'))
        for name, size, crop, side_check, coefficient in specifications:
            rendered = cache_root / f'views{size}_side{int(side_check)}'
            prepare(data, split, weights, rendered, size, device, side_check)
            models = [paths/f'{name}_fold{i}.pt' for i in range(5)] if name != 'public' else [paths/'public_convnext_small.pt']
            logits = []
            for checkpoint in models:
                model = load_model(checkpoint, crop, device)
                ds = Studies(rendered, uids)
                rows = []
                for i in range(len(ds)):
                    images, metadata, _ = ds[i]
                    rows.append(predict_windows(model, images, metadata, device, crop, chunk, precision))
                    print(f'{checkpoint.name} {i+1}/{len(ds)}', flush=True)
                logits.append(np.stack(rows))
                del model
                if str(device).startswith('cuda'): torch.cuda.empty_cache()
            totals += coefficient * probabilities_from_logits(logits)
        publish_predictions(output, uids, totals)
