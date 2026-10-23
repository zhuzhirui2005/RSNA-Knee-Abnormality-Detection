# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
import contextlib, hashlib, io, json, os, shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
FORCED_SIDE = {}
class DicomReader(torch.utils.data.Dataset):

    def __init__(self, data, split, uids, tmp, code, size, eval_stride=3, views='v1', slices=3, anchor_weights=None, phys_mm=0.0, side_model=None, side_threshold=2.0, side_max_x=0.0, side_position_mm=20.0, extent_meta=False, device='cpu'):
        self.data, self.split, self.uids = (str(data), split, list(uids))
        self.device = device
        self.extent_meta = bool(extent_meta)
        self.tmp, self.code, self.size = (str(tmp), str(code), int(size))
        self.eval_stride, self.views, self.slices = (int(eval_stride), str(views), int(slices))
        self.phys_mm = float(phys_mm)
        self.anchor_weights = None if anchor_weights is None else str(anchor_weights)
        if self.views in ('v3k', 'v4') and (not self.anchor_weights):
            raise ValueError('views v3k need --anchor-weights (SKM-TEA checkpoint) to rebuild the knee anchors at inference')
        self.native = None
        self.side_model = None if side_model is None else str(side_model)
        self.side_threshold, self.side_max_x = (float(side_threshold), float(side_max_x))
        self.side_position_mm = float(side_position_mm)
        self.side_net = None
        self.source_directory = True

    def _setup(self):
        from . import native_cache, views, geometry_bank, anchors
        self.PI = SimpleNamespace(OneStudySource=native_cache.DirectorySource)
        self.native = native_cache
        self.builder = views.StudySlices(self.tmp, [], self.size, self.views)
        self.bank = None
        if self.views in ('v3k', 'v4'):
            from . import geometry_bank as GB
            from . import anchors as KB
            self.GB, self.KB = (GB, KB)
            dev = self.device
            if dev == 'cpu':
                torch.set_num_threads(2)
            self.bank = KB.AnchorBank(KB.load_model(self.anchor_weights, dev), dev)
        from . import dicom_geometry as G
        if not getattr(G, '_forced_side_wrapped', False):
            import dataclasses
            orig = G.decide_study_side

            def decide(study, tag_values, sagittal_x, _orig=orig):
                r = _orig(study, tag_values, sagittal_x)
                f = FORCED_SIDE.get(str(study))
                return r if f is None else dataclasses.replace(r, side=f, source='image_side_check')
            G.decide_study_side = decide
            G._forced_side_wrapped = True
        if self.side_model:
            from . import side as SC
            self.SC = SC
            sdev = self.device
            net = SC.build_side_net()
            net.load_state_dict(torch.load(self.side_model, map_location='cpu', weights_only=True), strict=True)
            self.side_net, self.side_dev = (net.eval().to(sdev), sdev)

    def _prepare(self, uid, work):
        with contextlib.redirect_stdout(io.StringIO()):
            self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=self.source_directory)
        if self.bank is not None or self.side_net is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.GB.build(root=self.data, manifest=str(work / 'manifest.json'), output=str(work / 'geometry'), split=self.split, source_directory=self.source_directory)

    def _native_and_geometry(self, uid, work):
        """native render + geometry bank for one study; with a side model, the image side check (may re-render)."""
        self.PI.OneStudySource.current = uid
        self._prepare(uid, work)
        if self.side_net is None:
            return None
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        with open(work / 'studies' / (key + '.json'), encoding='utf-8') as fh:
            study = json.load(fh)
        sides = {s.get('side') for s in study['series']}
        cur = next(iter(sides)) if len(sides) == 1 else None
        status = set()
        for s in study['series']:
            gp = work / 'geometry' / (hashlib.sha256((uid + '/' + s['SeriesInstanceUID']).encode()).hexdigest()[:24] + '.json')
            if gp.exists():
                status.add(json.loads(gp.read_text(encoding='utf-8')).get('laterality_status'))
        sag = [float(np.median(s['positions'])) for s in study['series'] if s.get('plane') == 'Sagittal' and s.get('positions')]
        x = float(np.median(sag)) if sag else None
        note = {'side': cur, 'status': '|'.join(sorted((str(v) for v in status))), 'x': None if x is None else round(x, 1), 'score': None, 'flipped': False}
        if status == {'explicit_study_tags'} and cur in ('L', 'R'):
            pos = None if x is None or abs(x) < self.side_position_mm else 'L' if x >= 0 else 'R'
            if pos is None or pos == cur:
                return note
            note['tag_contradicted_by_position'] = True
        if self.side_max_x > 0 and cur in ('L', 'R') and (x is not None) and (abs(x) >= self.side_max_x):
            return note
        ims = []
        for s in study['series']:
            if s.get('plane') not in self.SC.SIDE_PLANES:
                continue
            p = Path(s['path'])
            if not p.exists():
                p = work / 'series' / os.path.basename(str(s['path']).replace(chr(92), '/'))
            if p.exists():
                ims.append(self.SC.side_inputs(np.load(p, mmap_mode='r')))
        ims = [v for v in ims if len(v)]
        if not ims:
            return note
        xb = torch.from_numpy(np.concatenate(ims)).float().div_(255.0)[:, None]
        with torch.inference_mode():
            z = self.side_net(((xb - 0.45) / 0.25).to(self.side_dev)).float().squeeze(1)
        score = float(z.mean())
        note['score'] = round(score, 3)
        if score < -self.side_threshold:
            new = 'R' if cur in (None, 'L') else 'L'
            FORCED_SIDE[uid] = new
            shutil.rmtree(work, ignore_errors=True)
            self._prepare(uid, work)
            with open(work / 'studies' / (key + '.json'), encoding='utf-8') as fh:
                got = {s.get('side') for s in json.load(fh)['series']}
            if got != {new}:
                raise ValueError('forced side %s not applied (%s)' % (new, got))
            note.update(flipped=True, new_side=new)
        return note

    def read(self, uid):
        if self.native is None:
            self._setup()
        work = Path(self.tmp) / ('study_' + hashlib.sha256(uid.encode()).hexdigest()[:24])
        try:
            self._native_and_geometry(uid, work)
            self.builder.cache = str(work)
            key = self.KB.study_key(uid)
            study = json.loads((work / 'studies' / (key + '.json')).read_text())
            record = self.bank.record(uid, study, geo_dir=str(work / 'geometry'))
            (work / 'anchors').mkdir(exist_ok=True)
            (work / 'anchors' / (key + '.json')).write_text(json.dumps(record))
            self.builder.anchors = str(work / 'anchors')
            item = self.builder.build(uid)
            if not len(item['px']):
                raise ValueError('No usable DICOM slices: ' + uid)
            return item, sorted({str(x.get('side')) for x in study['series']})
        finally:
            FORCED_SIDE.pop(uid, None)
            shutil.rmtree(work, ignore_errors=True)
