"""Fetch release weights and the original upstream preprocessing checkpoints."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from importlib.resources import files
import zipfile


def download_weights(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(files('knee_mri').joinpath('resources/weights.json').read_text())
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}/"
    for entry in manifest['files']:
        path = root / entry['file']
        if path.exists() and sha256(path) == entry['sha256']:
            print(f'{path.name}: ready', flush=True)
            continue
        url = entry.get('url', base + entry['file'])
        print(f"Downloading {path.name} ({entry['bytes']/1e6:.1f} MB)", flush=True)
        tmp = path.with_suffix('.download')
        try:
            with urlopen(Request(url, headers={'User-Agent': 'rsna-knee-mri/0.1'}), timeout=120) as response, tmp.open('wb') as out:
                while block := response.read(4 * 1024**2):
                    out.write(block)
            # Kaggle may return a single-file ZIP for a named dataset file.
            if zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as z:
                    member = next(n for n in z.namelist() if Path(n).name == path.name)
                    with z.open(member) as src, path.with_suffix('.unpacked').open('wb') as dst:
                        while block := src.read(4 * 1024**2): dst.write(block)
                path.with_suffix('.unpacked').replace(tmp)
            if sha256(tmp) != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {path.name}. Source: {entry.get("source", url)}')
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''): digest.update(block)
    return digest.hexdigest()
