# Adapted from Heliosli's RSNA Knee pipeline. See LICENSE and README.md for attribution.
"""Collapse only byte-identical native evidence with equal consumed inputs.

Equality is local to a Study and this dataset's geometry/anatomy configuration.
It is not a claim that two acquisitions belong to the same anatomical position
or natural person. File/Series aliases are retained even when only one image
copy is encoded. Old checkpoints keep every Series unless explicitly enabled.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from .common import UID

RENDER_KEYS = (
    "positions",
    "spacing_mm",
    "native_shape",
    "plane",
    "slot",
    "fat_suppression",
    "fluid_sensitive",
    "side",
    "intensity",
)
GEOMETRY_KEYS = ("affines_col_row_origin_lps_mm", "quality")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _key(uid, sid):
    return hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24]


def _file_state(path):
    path = Path(path)
    if not path.is_file():
        return (str(path), None, None, None)
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _array_hash(array):
    digest = hashlib.sha256()
    digest.update(
        _json({"shape": list(array.shape), "dtype": str(array.dtype)}).encode()
    )
    for plane in array:
        digest.update(np.ascontiguousarray(plane).tobytes())
    return digest.hexdigest()


def _payload_equal(first, second):
    if set(first) != set(second):
        return False
    return all(
        (
            np.asarray(first[k]).dtype == np.asarray(second[k]).dtype
            and np.array_equal(first[k], second[k])
            for k in first
        )
    )


class ExactDuplicateResolver:
    """Lazy per-Study plans; source or auxiliary file changes invalidate them."""

    def __init__(
        self,
        geometry_dir=None,
        anatomy_dir=None,
        include_pixels=True,
        *,
        anatomy_reader=None,
    ):
        self.geometry_dir = Path(geometry_dir) if geometry_dir else None
        self.anatomy_dir = Path(anatomy_dir) if anatomy_dir else None
        self.include_pixels = bool(include_pixels)
        self.cache = {}
        self.anatomy_reader = anatomy_reader

    def _paths(self, uid, series):
        key = _key(uid, series["SeriesInstanceUID"])
        anatomy = self.anatomy_dir / (key + ".npz") if self.anatomy_dir else None
        if self.anatomy_reader is not None:
            row = self.anatomy_reader.rows.get((uid, series["SeriesInstanceUID"]))
            if row is not None:
                anatomy = Path(row["path"])
        return (
            self.geometry_dir / (key + ".json") if self.geometry_dir else None,
            anatomy,
        )

    def _contract(self, study):
        records = []
        for series in study["series"]:
            gp, ap = self._paths(study[UID], series)
            files = [_file_state(series["path"])]
            if gp is not None:
                files.append(_file_state(gp))
            if ap is not None:
                files.append(_file_state(ap))
                files.append(_file_state(ap.with_suffix(".json")))
                files.append(_file_state(self.anatomy_dir / "manifest.json"))
                files.append(_file_state(ap.parent.parent / "anatomy_v2" / ap.name))
            records.append(
                {
                    "id": series["SeriesInstanceUID"],
                    "meta": {k: series.get(k) for k in RENDER_KEYS},
                    "files": files,
                }
            )
        contract = {
            "records": records,
            "anatomy_producer_signature": getattr(
                self.anatomy_reader, "signature", None
            ),
        }
        return hashlib.sha256(_json(contract).encode()).hexdigest()

    def _consumer_differences(self, uid, first, second):
        differences = [k for k in RENDER_KEYS if first.get(k) != second.get(k)]
        if differences:
            return differences
        g1, a1 = self._paths(uid, first)
        g2, a2 = self._paths(uid, second)
        if g1 is not None:
            if not g1.is_file() or not g2.is_file():
                return ["required_geometry_missing"]
            objects = [
                json.loads(path.read_text(encoding="utf-8")) for path in (g1, g2)
            ]
            differences.extend(
                (
                    "geometry." + k
                    for k in GEOMETRY_KEYS
                    if objects[0].get(k) != objects[1].get(k)
                )
            )
        if a1 is not None:
            if not a1.is_file() or not a2.is_file():
                return [*differences, "required_anatomy_missing"]
            if self.anatomy_reader is not None:
                try:
                    one = self.anatomy_reader.load(
                        uid, first, g1, include_pixels=self.include_pixels
                    )
                    two = self.anatomy_reader.load(
                        uid, second, g2, include_pixels=self.include_pixels
                    )
                except (ValueError, FileNotFoundError) as exc:
                    return [*differences, "anatomy_contract_rejected: " + str(exc)]
            else:
                from .anatomy_cache import load_anatomy

                one, two = (
                    load_anatomy(a1, self.include_pixels),
                    load_anatomy(a2, self.include_pixels),
                )
            if not _payload_equal(one, two):
                differences.append("anatomy_or_roi_payload")
        return differences

    def resolve(self, study):
        contract = self._contract(study)
        uid = study[UID]
        old = self.cache.get(uid)
        if old is not None and old["contract_sha256"] == contract:
            return old
        rows = study["series"]
        groups = {}
        arrays = {}
        candidate_pairs = []
        metadata_groups = {}
        metadata_rejections = []
        sample_bytes = 0
        full_pairs_checked = 0
        for index, series in enumerate(rows):
            metadata_groups.setdefault(
                _json({k: series.get(k) for k in RENDER_KEYS}), []
            ).append(index)
            for other, previous in enumerate(rows[:index]):
                if previous.get("native_shape") != series.get("native_shape"):
                    continue
                differences = [
                    k for k in RENDER_KEYS if previous.get(k) != series.get(k)
                ]
                if differences:
                    metadata_rejections.append(
                        {
                            "first_index": other,
                            "second_index": index,
                            "consumer_differences": differences,
                            "pixels_exactly_equal": None,
                            "collapsed": False,
                        }
                    )
        for metadata, members in metadata_groups.items():
            if len(members) < 2:
                continue
            for index in members:
                array = np.load(rows[index]["path"], mmap_mode="r", allow_pickle=False)
                arrays[index] = array
                if array.ndim != 3 or array.size == 0:
                    raise ValueError("invalid native array for evidence equivalence")
                start = max(0, array.shape[1] // 2 - 8)
                sample = np.ascontiguousarray(
                    array[array.shape[0] // 2, start : start + 16, :]
                )
                sample_bytes += sample.nbytes
                key = (
                    metadata,
                    array.shape,
                    str(array.dtype),
                    hashlib.sha256(sample.tobytes()).hexdigest(),
                )
                groups.setdefault(key, []).append(index)
        canonical = list(range(len(rows)))
        pixel_hashes = {}
        for members in groups.values():
            representatives = []
            for index in members:
                matched = False
                for other in representatives:
                    full_pairs_checked += 1
                    if not np.array_equal(arrays[index], arrays[other]):
                        continue
                    if other not in pixel_hashes:
                        pixel_hashes[other] = _array_hash(arrays[other])
                    pixel_hashes[index] = pixel_hashes[other]
                    differences = self._consumer_differences(
                        uid, rows[other], rows[index]
                    )
                    pair = {
                        "first_index": other,
                        "second_index": index,
                        "first_series": rows[other]["SeriesInstanceUID"],
                        "second_series": rows[index]["SeriesInstanceUID"],
                        "native_array_sha256": pixel_hashes[index],
                        "pixels_exactly_equal": True,
                        "consumer_differences": differences,
                        "collapsed": not differences,
                    }
                    candidate_pairs.append(pair)
                    if not differences:
                        canonical[index] = other
                        matched = True
                        break
                if not matched:
                    representatives.append(index)
        retained = [i for i, c in enumerate(canonical) if i == c]
        aliases = [
            {
                "canonical_index": i,
                "canonical_series": rows[i]["SeriesInstanceUID"],
                "member_indices": [j for j, c in enumerate(canonical) if c == i],
                "member_series": [
                    rows[j]["SeriesInstanceUID"]
                    for j, c in enumerate(canonical)
                    if c == i
                ],
            }
            for i in retained
        ]
        result = {
            "policy": "collapse_exact",
            "contract_sha256": contract,
            "representative_indices": retained,
            "canonical_index": canonical,
            "aliases": aliases,
            "pairs": candidate_pairs,
            "metadata_rejections": metadata_rejections,
            "candidate_io": {
                "mri_arrays_opened": len(arrays),
                "contiguous_stripe_bytes": sample_bytes,
                "full_array_pairs_checked": full_pairs_checked,
            },
            "removed_series": len(rows) - len(retained),
            "geometry_compared": self.geometry_dir is not None,
            "anatomy_compared": self.anatomy_dir is not None,
            "scope": "equal consumed encoding evidence within one Study; not anatomical identity or label truth",
        }
        self.cache[uid] = result
        return result


def keep_plan(study):
    rows = study["series"]
    indices = list(range(len(rows)))
    return {
        "policy": "keep",
        "representative_indices": indices,
        "canonical_index": indices,
        "aliases": [
            {
                "canonical_index": i,
                "canonical_series": s["SeriesInstanceUID"],
                "member_indices": [i],
                "member_series": [s["SeriesInstanceUID"]],
            }
            for i, s in enumerate(rows)
        ],
        "pairs": [],
        "removed_series": 0,
        "scope": "legacy Series multiplicity retained",
    }
