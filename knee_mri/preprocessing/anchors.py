# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""Anatomical knee and patella anchors projected into each series."""
import argparse, csv, hashlib, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
GEO = None
CACHE = None
LO, HI = (0.15, 0.85)
AGREE_MM = 20.0
PRIOR_MM = 35.0
OBLIQUE_COS = 0.85
SPAN_MM = 4.0
MASS_MIN = 250
N_SLICES = 7

def key_for(uid, sid):
    return hashlib.sha256((uid + '/' + sid).encode()).hexdigest()[:24]

def study_key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:24]

def load_model(weights, dev):
    """the SKM-TEA V-Net through the mainline loader (needs the `training` package on sys.path)."""
    from .anatomy import load_segmenter
    return load_segmenter(weights, dev)

def project(aff, X):
    """aff [Z,3,3] (col,row,1 -> LPS mm); X [3] LPS -> (z, col, row, |distance to that plane|) of the nearest slice."""
    o = aff[:, :, 2]
    n = np.cross(aff[0, :, 0], aff[0, :, 1])
    n /= np.linalg.norm(n)
    d = (o - X[None]) @ n
    z = int(np.argmin(np.abs(d)))
    M = aff[z, :, :2]
    cr, *_ = np.linalg.lstsq(M, X - o[z], rcond=None)
    return (z, float(cr[0]), float(cr[1]), float(abs(d[z])))

def centroid(prob, thr=0.5):
    m = prob > thr
    if m.sum() < 40:
        return (None, 0)
    ys, xs = np.nonzero(m)
    w = prob[m]
    return ((float((ys * w).sum() / w.sum()), float((xs * w).sum() / w.sum())), int(m.sum()))

class AnchorBank:

    def __init__(self, model, dev, geo_dir=None):
        self.model, self.dev, self.geo_dir = (model, dev, geo_dir)

    def geometry(self, uid, sid, geo_dir=None):
        p = os.path.join(geo_dir or self.geo_dir, key_for(uid, sid) + '.json')
        return json.load(open(p)) if os.path.exists(p) else None

    @torch.no_grad()
    def segment(self, series, geo, n_slices=N_SLICES):
        from .anatomy_runtime import native_volume
        vol = native_volume(series['path'])
        mean, std = (float(vol.mean()), max(float(vol.std()), 1.0))
        aff = np.asarray(geo['affines_col_row_origin_lps_mm'], float)
        if abs(aff[0, 1, 0]) / np.linalg.norm(aff[0, :, 0]) < OBLIQUE_COS or abs(aff[0, 2, 1]) / np.linalg.norm(aff[0, :, 1]) < OBLIQUE_COS:
            return (None, 'oblique')
        flip_x = aff[0, 1, 0] < 0
        flip_y = aff[0, 2, 1] > 0
        m = len(vol) // 2
        dz = float(np.linalg.norm(aff[1, :, 2] - aff[0, :, 2])) if len(aff) > 1 else 4.0
        step = max(1, int(round(SPAN_MM / max(dz, 0.001))))
        zs = sorted({min(len(vol) - 1, max(0, m + k * step)) for k in range(-(n_slices // 2), n_slices // 2 + 1)})
        out = []
        for z in zs:
            x = (np.asarray(vol[z], np.float32) - mean) / std
            if flip_x:
                x = x[:, ::-1]
            if flip_y:
                x = x[::-1]
            sy, sx = series['spacing_mm'][z]
            shape = (max(16, round(x.shape[1] * sx / 0.3125)), max(16, round(x.shape[0] * sy / 0.3125)))
            x = cv2.resize(np.ascontiguousarray(x), shape, interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(x.copy())[None, None].to(self.dev)
            h, w = t.shape[-2:]
            t = F.pad(t, (0, -w % 16, 0, -h % 16), mode='replicate')
            p = self.model(t).sigmoid()[..., :h, :w]
            p = F.interpolate(p, size=vol.shape[-2:], mode='bilinear', align_corners=False)[0].float().cpu().numpy()
            if flip_x:
                p = p[:, :, ::-1]
            if flip_y:
                p = p[:, ::-1]
            out.append((z, np.ascontiguousarray(p)))
        return (out, 'ok')

    def candidate(self, s, geo):
        aff = np.asarray(geo['affines_col_row_origin_lps_mm'], float)
        segs, why = self.segment(s, geo)
        if segs is None:
            return (None, why)
        pts, wts, ppts, pwts = ([], [], [], [])
        for z, p in segs:
            joint = np.max(p[1:4], axis=0)
            c, n = centroid(joint, 0.5)
            if c is not None:
                pts.append(aff[z] @ np.array([c[1], c[0], 1.0]))
                wts.append(n)
            cp, npx = centroid(p[0], 0.5)
            if cp is not None and npx >= 60:
                ppts.append(aff[z] @ np.array([cp[1], cp[0], 1.0]))
                pwts.append(npx)
        mass, pmass = (int(sum(wts)), int(sum(pwts)))
        if mass < MASS_MIN:
            return (None, 'mass<%d(%d)' % (MASS_MIN, mass))
        X = np.average(np.array(pts), axis=0, weights=np.array(wts, float))
        Xp = np.average(np.array(ppts), axis=0, weights=np.array(pwts, float)) if pmass >= 120 else None
        if Xp is not None and (not (Xp[1] < X[1] and Xp[2] > X[2] - 10)):
            Xp = None
        h, w = s['native_shape'][-2:]
        z, c, r, _ = project(aff, X)
        rf, cf = (r / max(1, h - 1), c / max(1, w - 1))
        if not (LO <= rf <= HI and LO <= cf <= HI):
            return (None, 'implausible(rf=%.2f,cf=%.2f)' % (rf, cf))
        return (dict(X=X, Xp=Xp, mass=mass, pmass=pmass, sid=s['SeriesInstanceUID']), 'ok')

    def slab_prior(self, uid, d, geo_dir=None):
        per = {}
        for s in d['series']:
            geo = self.geometry(uid, s['SeriesInstanceUID'], geo_dir)
            if geo is None or not geo['quality']['aligned']:
                continue
            aff = np.asarray(geo['affines_col_row_origin_lps_mm'], float)
            h, w = s['native_shape'][-2:]
            z = len(aff) // 2
            per.setdefault(s['plane'], []).append(aff[z] @ np.array([(w - 1) / 2.0, (h - 1) / 2.0, 1.0]))
        if not per:
            return None
        G = np.zeros(3)
        for ax, planes in enumerate((('Sagittal',), ('Coronal',), ('Sagittal', 'Coronal'))):
            cs = [c for pl in planes for c in per.get(pl, [])] or [c for pl in per for c in per[pl]]
            G[ax] = float(np.median([c[ax] for c in cs]))
        return G

    def record(self, uid, d, geo_dir=None):
        """anchor record of one study from its native-cache study dict `d` and the geometry directory."""
        sags = [s for s in d['series'] if s['plane'] == 'Sagittal' and os.path.exists(s['path'])]
        sags.sort(key=lambda s: -(s.get('fluid_sensitive') or 0))
        rec = {'study': uid, 'version': 2, 'joint_lps': None, 'patella_lps': None, 'method': None, 'why': [], 'series': {}}
        G = self.slab_prior(uid, d, geo_dir)
        rec['slab_prior_lps'] = None if G is None else [float(v) for v in G]
        cands = []
        for s in sags:
            geo = self.geometry(uid, s['SeriesInstanceUID'], geo_dir)
            if geo is None or not geo['quality']['aligned']:
                rec['why'].append('no_geo' if geo is None else 'not_aligned')
                continue
            try:
                c, why = self.candidate(s, geo)
            except Exception as exc:
                c, why = (None, 'error:' + repr(exc)[:80])
            if c is None:
                rec['why'].append(why)
                continue
            if G is not None and np.linalg.norm(c['X'] - G) > PRIOR_MM:
                rec['why'].append('far_from_prior(%.0fmm)' % np.linalg.norm(c['X'] - G))
                continue
            cands.append(c)
        X = Xp = None
        if cands:
            cands.sort(key=lambda c: -c['mass'])
            best = cands[0]
            close = [c for c in cands if np.linalg.norm(c['X'] - best['X']) <= AGREE_MM]
            if len(close) >= 2:
                X = np.average(np.array([c['X'] for c in close]), axis=0, weights=np.array([c['mass'] for c in close], float))
                rec['method'] = 'skm_tea_joint_avg%d' % len(close)
            else:
                X = best['X']
                rec['method'] = 'skm_tea_joint'
                if len(cands) >= 2:
                    rec['why'].append('disagree(%.0fmm)' % np.linalg.norm(cands[1]['X'] - best['X']))
            pc = [c for c in close if c['Xp'] is not None] or [c for c in cands if c['Xp'] is not None]
            if pc:
                Xp = max(pc, key=lambda c: c['pmass'])['Xp']
        elif G is not None:
            X = G
            rec['method'] = 'slab_prior'
        else:
            rec['why'].append('no_geometry')
        if X is not None:
            rec['joint_lps'] = [float(v) for v in X]
            if Xp is not None:
                rec['patella_lps'] = [float(v) for v in Xp]
            for s in d['series']:
                geo = self.geometry(uid, s['SeriesInstanceUID'], geo_dir)
                if geo is None:
                    continue
                aff = np.asarray(geo['affines_col_row_origin_lps_mm'], float)
                h, w = s['native_shape'][-2:]
                z, c, r, dist = project(aff, X)
                entry = {'joint': [r / max(1, h - 1), c / max(1, w - 1)], 'in_field': bool(0 <= r < h and 0 <= c < w), 'dist_mm': round(dist, 1), 'patella': None}
                if Xp is not None:
                    zp, cp, rp, dp = project(aff, Xp)
                    entry['patella'] = [rp / max(1, h - 1), cp / max(1, w - 1)]
                rec['series'][s['SeriesInstanceUID']] = entry
        return rec

def plausible_v1(rec, d):
    """does an existing record pass the v2 gate?  (joint fractions of every sagittal series inside the window)"""
    if not str(rec.get('method', '')).startswith('skm_tea_joint'):
        return False
    for s in d['series']:
        if s['plane'] != 'Sagittal':
            continue
        e = rec['series'].get(s['SeriesInstanceUID'])
        if e is None or not e['in_field']:
            return False
        rf, cf = e['joint']
        if not (LO <= rf <= HI and LO <= cf <= HI):
            return False
    return True
