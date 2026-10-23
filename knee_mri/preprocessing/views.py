# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""Physical multiview rendering with joint, patella and posterior crops."""
import sys, os, json, time, hashlib, argparse
import numpy as np
import torch
LOCAL_MM = 100.0
THIN_MM, SLAB_MM = (1.75, 3.5)
POLICY = 'series_companded'
PLANE_ID = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
META_COLS = ['series_idx', 'slot', 'plane', 'view_id', 'rel_pos', 'log_spacing', 'extent_130', 'fluid', 'fatsat', 'slice_order', 'n_slices', 'real_series_idx']
UIDK = 'StudyInstanceUID'

def study_json(cache, uid):
    return os.path.join(cache, 'studies', hashlib.sha256(uid.encode()).hexdigest()[:24] + '.json')
PF_CENTER = {'Sagittal': (0.38, 0.6), 'Axial': (0.38, 0.5)}
POST_CENTER = {'Sagittal': (0.55, 0.35), 'Axial': (0.62, 0.5)}

def knee_anchor(vol, plane):
    """(row_frac, col_frac) of the knee in the series' in-plane frame from the foreground of the 3 middle slices.
    Falls back to the field centre when the leg fills the field or nothing is found."""
    import cv2
    m = len(vol) // 2
    mid = np.asarray(vol[max(0, m - 1):m + 2]).astype(np.float32).mean(0)
    H, W = mid.shape
    hi = float(np.percentile(mid, 99.5))
    if hi <= 0:
        return ((0.5, 0.5), 'empty')
    fg = (mid > 0.08 * hi).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return ((0.5, 0.5), 'no_fg')
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = [int(v) for v in stats[k, :4]]
    if w >= 0.97 * W and h >= 0.97 * H:
        return ((0.5, 0.5), 'fills_field')
    comp = lab == k
    if plane in ('Sagittal', 'Coronal'):
        width = comp[y:y + h].sum(1).astype(np.float32)
        if len(width) >= 15:
            kern = np.ones(15, np.float32) / 15
            width = np.convolve(width, kern, mode='same')
        row = (y + int(np.argmax(width))) / max(1, H - 1)
    else:
        row = (y + h / 2.0) / max(1, H - 1)
    col = (x + w / 2.0) / max(1, W - 1)
    return ((float(row), float(col)), 'ok')

def anchored(anchor, spacing_mm, shape, d_row_mm, d_col_mm):
    """centre = anchor shifted by physical offsets (rows: +down, cols: +right), as (row_frac, col_frac)."""
    sy, sx = (float(spacing_mm[0]), float(spacing_mm[1]))
    H, W = shape
    return (min(1.0, max(0.0, anchor[0] + d_row_mm / (sy * max(1, H - 1)))), min(1.0, max(0.0, anchor[1] + d_col_mm / (sx * max(1, W - 1)))))

def series_views(plane, native_shape, spacings, size, views='v1', anchor=None, spacing_mm=None, patella=None):
    if views in ('v3k', 'v4'):
        full = [(0, None, (0.5, 0.5))] if views == 'v4' else []
        shp = tuple(native_shape[-2:])
        a = anchor if anchor is not None else (0.5, 0.5)
        sp = spacing_mm if spacing_mm is not None else (1.0, 1.0)
        if plane == 'Sagittal':
            pf = patella if patella is not None else anchored(a, sp, shp, -25.0, +40.0)
            return full + [(1, LOCAL_MM, a), (4, LOCAL_MM, pf), (5, LOCAL_MM, anchored(a, sp, shp, +5.0, -35.0))]
        if plane == 'Axial':
            pf = patella if patella is not None else anchored(a, sp, shp, -40.0, 0.0)
            return full + [(1, LOCAL_MM, a), (4, LOCAL_MM, pf), (5, LOCAL_MM, anchored(a, sp, shp, +35.0, 0.0))]
        return full + [(2, LOCAL_MM, anchored(a, sp, shp, 0.0, -30.0)), (3, LOCAL_MM, anchored(a, sp, shp, 0.0, +30.0))]
    if plane == 'Coronal':
        if views == 'v2add':
            return []
        from .coronal_views import coronal_bilateral_views
        out = coronal_bilateral_views(tuple(native_shape[-2:]), spacings, size, LOCAL_MM)
        if views == 'v3':
            return [v for v in out if int(v[0]) != 0]
        return out
    if views == 'v2add':
        return [(4, LOCAL_MM, PF_CENTER[plane])] if plane in PF_CENTER else []
    if views == 'v3':
        return [(1, LOCAL_MM, (0.5, 0.5)), (4, LOCAL_MM, PF_CENTER[plane]), (5, LOCAL_MM, POST_CENTER[plane])]
    out = [(0, None, (0.5, 0.5)), (1, LOCAL_MM, (0.5, 0.5))]
    if views == 'v2':
        out.append((4, LOCAL_MM, PF_CENTER[plane]))
    elif views != 'v1':
        raise ValueError('unknown view set ' + str(views))
    return out

class StudySlices(torch.utils.data.Dataset):

    def __init__(self, cache, uids, size, views='v1', anchors=None):
        self.cache, self.uids, self.size, self.views = (cache, uids, size, views)
        self.anchors = anchors
        self._resolver = None

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        try:
            return self.build(uid)
        except Exception as exc:
            return {'uid': uid, 'error': repr(exc)[:400], 'px': torch.zeros((0, self.size, self.size), dtype=torch.uint8), 'meta': torch.zeros((0, len(META_COLS)), dtype=torch.float32), 'blocks': [], 'removed': 0}

    def build(self, uid):
        import cv2
        cv2.setNumThreads(1)
        from .input_provenance import view_geometry
        from .intensity_mapping import normalize_intensity
        from .native_data import select_display_window
        from .exact_duplicates import ExactDuplicateResolver
        if self._resolver is None:
            self._resolver = ExactDuplicateResolver(None, None, False)
        S = self.size
        with open(study_json(self.cache, uid), encoding='utf-8') as fh:
            study = json.load(fh)
        series, skipped = ([], [])
        for s in study['series']:
            path = s['path']
            if not os.path.exists(path):
                path = os.path.join(self.cache, 'series', os.path.basename(path.replace(chr(92), '/')))
            if not os.path.exists(path):
                skipped.append(s['SeriesInstanceUID'])
                continue
            series.append(dict(s, path=path))
        if not series:
            raise ValueError('no series file present for this study (%d missing)' % len(skipped))
        study = dict(study, series=series)
        plan = self._resolver.resolve(study)
        self._resolver.cache.clear()
        px, meta, blocks = ([], [], [])
        for real_idx in plan['representative_indices']:
            s = study['series'][real_idx]
            path = s['path']
            volume = np.load(path, mmap_mode='r', allow_pickle=False)
            pos = np.asarray(s['positions'], float)
            if volume.ndim != 3 or len(pos) != len(volume) or (not len(pos)) or (not np.isfinite(pos).all()):
                raise ValueError('native volume/slice-position mapping is incomplete')
            if s['plane'] == 'Sagittal' and s.get('side') == 'R':
                pos = -pos
            order = np.argsort(pos, kind='stable')
            pos = pos[order]
            spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 0.0
            window = select_display_window(s, POLICY)
            fluid = s.get('fluid_sensitive', -1)
            fat = s.get('fat_suppression', -1)
            sidx = len(blocks)
            vol = np.asarray(volume[order])
            order_rep = order
            if 0 < spacing < THIN_MM:
                g = max(1, int(round(SLAB_MM / spacing)))
                n_new = len(pos) // g + (1 if len(pos) % g >= max(1, g // 2) else 0)
                if n_new >= 2:
                    chunks = [vol[j * g:min(len(vol), (j + 1) * g)] for j in range(n_new)]
                    vol = np.stack([np.rint(c.astype(np.float32).mean(0)).astype(vol.dtype) for c in chunks])
                    pos = np.array([pos[j * g:min(len(pos), (j + 1) * g)].mean() for j in range(n_new)])
                    order_rep = np.array([order[j * g] for j in range(n_new)])
                    spacing = float(np.median(np.diff(pos)))
            block = {'series': s['SeriesInstanceUID'], 'plane': s['plane'], 'n_slices': int(len(pos)), 'views': []}
            anchor = pat = None
            sp0 = None
            if self.views in ('v3k', 'v4'):
                if self.anchors:
                    ap_ = os.path.join(self.anchors, hashlib.sha256(uid.encode()).hexdigest()[:24] + '.json')
                    ent = (json.load(open(ap_)).get('series', {}) if os.path.exists(ap_) else {}).get(s['SeriesInstanceUID'])
                    if ent and ent.get('in_field', True):
                        anchor = tuple((float(v) for v in ent['joint']))
                        pat = tuple((float(v) for v in ent['patella'])) if ent.get('patella') else None
                if anchor is None:
                    anchor = (0.5, 0.5)
                sp0 = s['spacing_mm'][int(order_rep[0])]
            for view_id, crop, center_xy in series_views(s['plane'], volume.shape, s['spacing_mm'], S, self.views, anchor, sp0, pat):
                block['views'].append([int(view_id), len(px)])
                for k in range(len(pos)):
                    sp = s['spacing_mm'][int(order_rep[k])]
                    g = view_geometry(vol.shape[1:], sp, S, crop_mm=crop, center=center_xy)
                    x, y, x1, y1 = g.crop_xyxy
                    pxo, pyo, _, _ = g.valid_xyxy
                    oh, ow = g.resized_shape
                    resized = cv2.resize(vol[k, y:y1, x:x1].astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
                    resized = normalize_intensity(resized, window, POLICY)
                    out = np.zeros((S, S), np.uint8)
                    out[pyo:pyo + oh, pxo:pxo + ow] = np.rint(np.clip(resized, 0, 1) * 255).astype(np.uint8)
                    px.append(out)
                    relative = 2 * (pos[k] - pos[0]) / max(pos[-1] - pos[0], 1e-06) - 1
                    meta.append([sidx, s['slot'], PLANE_ID[s['plane']], view_id, relative, np.log1p(abs(spacing)), g.extent / 130.0, -1 if fluid is None else fluid, -1 if fat is None else fat, k, len(pos), real_idx])
            blocks.append(block)
            del volume, vol
        if px:
            px = torch.from_numpy(np.stack(px))
            meta = torch.tensor(meta, dtype=torch.float32)
        else:
            px = torch.zeros((0, S, S), dtype=torch.uint8)
            meta = torch.zeros((0, len(META_COLS)), dtype=torch.float32)
        return {'uid': uid, 'error': None, 'px': px, 'meta': meta, 'blocks': blocks, 'skipped_series': skipped, 'removed': int(plan.get('removed_series', 0))}

def count_upper(cache, uid, size, views='v1'):
    with open(study_json(cache, uid), encoding='utf-8') as fh:
        study = json.load(fh)
    n = 0
    for s in study['series']:
        try:
            v = len(series_views(s['plane'], s['native_shape'], s['spacing_mm'], size, 'v3' if views in ('v3k', 'v4') else views)) + (1 if views == 'v4' else 0)
        except Exception:
            v = 3
        n += v * len(s['positions'])
    return n
