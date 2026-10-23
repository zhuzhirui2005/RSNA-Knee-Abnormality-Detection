# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
from __future__ import annotations
import hashlib
import json
import random
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
SLOTS = {('Sagittal', 1): 1, ('Sagittal', 0): 2, ('Coronal', 1): 3, ('Coronal', 0): 4, ('Axial', 1): 5, ('Axial', 0): 6, ('Sagittal', -1): 7, ('Coronal', -1): 8, ('Axial', -1): 9}

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
        stream.write(payload)
        tmp = Path(stream.name)
    try:
        for attempt in range(7):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(min(0.05 * 2 ** attempt, 0.8))
    finally:
        if tmp.exists():
            tmp.unlink()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
