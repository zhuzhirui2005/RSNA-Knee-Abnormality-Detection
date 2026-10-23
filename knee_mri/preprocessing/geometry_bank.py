# Adapted from Heliosli's RSNA Knee pipeline. See LICENSE and README.md for attribution.
"""Preserve the native-array pixel -> patient LPS mapping and alignment quality."""

from pathlib import Path
import argparse, hashlib, json, time
import io
import numpy as np
import pydicom

from . import dicom_geometry as geom
from .common import UID, save_json, file_hash
from .geometry_runtime import GeometryRuntime, reusable_geometry

COMPATIBLE_GEOMETRY_BUILDERS = {
    "5e97ed15420930e1615f399190352be43364bc5c7fce04be7e417c457762b840"
}


def explicit_header_sides(ds):
    """Read both independent tags; neither field may hide a disagreement."""
    return {
        side
        for side in (
            geom.normalize_laterality(getattr(ds, "Laterality", None)),
            geom.normalize_laterality(getattr(ds, "ImageLaterality", None)),
        )
        if side is not None
    }


def resolve_anatomy_laterality(geometries):
    """Study-wide explicit evidence is authoritative for named compartments.

    Native array orientation may also use IPP-x as a rendering convention.
    That fallback does not establish anatomical left/right identity. Legacy
    geometry records are accepted only when their explicit tags agree across
    the entire Study; a conflicting Study cannot regain trust per Series.
    """
    values = (
        list(geometries.values()) if isinstance(geometries, dict) else list(geometries)
    )
    tags = set()
    for value in values:
        evidence = value.get("study_explicit_sides", value.get("raw_explicit_sides"))
        tags.update(
            (
                x
                for x in (
                    evidence if evidence is not None else [value.get("explicit_side")]
                )
                if x in ("L", "R")
            )
        )
    conflict = len(tags) > 1 or any(
        (v.get("laterality_status") == "conflicting_study_tags" for v in values)
    )
    side = next(iter(tags)) if len(tags) == 1 and (not conflict) else None
    return (
        side,
        {
            "status": "conflicting_study_tags"
            if conflict
            else "explicit_study_tags"
            if side
            else "unknown_no_explicit_tags",
            "explicit_sides": sorted(tags),
            "scope": "study; independent of native rendering side",
        },
    )


def pixel_affine(ds, plane, side):
    row = np.asarray(ds.ImageOrientationPatient[:3], float)
    col = np.asarray(ds.ImageOrientationPatient[3:], float)
    origin = np.asarray(ds.ImagePositionPatient, float)
    sy, sx = map(float, ds.PixelSpacing)
    h, w = (int(ds.Rows), int(ds.Columns))
    _, _, operation, _, _ = geom.canonical_inplane_axes(row, col, plane)
    corners = np.array([[0, w - 1], [(h - 1) * w, h * w - 1]], np.int64)
    transformed = geom._apply_inplane_transform(corners, operation)
    shape = geom._apply_inplane_transform(np.zeros((h, w), np.uint8), operation).shape
    if geom.flip_for_plane(plane, side):
        transformed = transformed[:, ::-1]

    def point(index):
        return origin + int(index) % w * sx * row + int(index) // w * sy * col

    p0 = point(transformed[0, 0])
    pc = point(transformed[0, 1])
    pr = point(transformed[1, 0])
    return np.stack(
        [(pc - p0) / max(1, shape[1] - 1), (pr - p0) / max(1, shape[0] - 1), p0], axis=1
    )


def alignment_quality(affines):
    a = np.asarray(affines, float)
    normal = np.cross(a[0, :, 0], a[0, :, 1])
    normal /= np.linalg.norm(normal)
    origins = a[:, :, 2]
    relative = origins - origins[0]
    depth = origins @ normal
    drift = np.linalg.norm(relative - (relative @ normal)[:, None] * normal, axis=1)
    axis_difference = float(np.max(np.abs(a[:, :, :2] - a[0, :, :2])))
    return {
        "aligned": bool(axis_difference < 0.001 and drift.max() < 1.0),
        "maximum_inplane_drift_mm": float(drift.max()),
        "maximum_axis_difference": axis_difference,
        "depth_mm": depth.tolist(),
        "normal": normal.tolist(),
    }


def build(
    root=None,
    manifest=None,
    output=None,
    split="train",
    source_directory=True,
    *,
    runtime=None,
):
    if not source_directory:
        raise ValueError("Geometry construction expects an extracted DICOM directory")
    root = Path(root)
    manifest = Path(manifest or root / "cache_native/manifest.json")
    output = Path(output or root / "geometry")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = []
    owned = runtime is None
    runtime = runtime or GeometryRuntime()
    started = time.perf_counter()
    meta = runtime.source_metadata(root, data["config"], source_directory)
    builder_hash = file_hash(__file__)
    allowed = {builder_hash, *COMPATIBLE_GEOMETRY_BUILDERS}
    stats = {
        "studies": len(data["studies"]),
        "reused_series": 0,
        "built_series": 0,
        "dicom_headers_read": 0,
        "geometry_records_written": 0,
        "manifest_written": False,
    }
    opens_before = runtime.archive_opens
    tags = [
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "Rows",
        "Columns",
        "FrameOfReferenceUID",
        "Laterality",
        "ImageLaterality",
    ]
    try:
        for study in data["studies"]:
            study_records = []
            for series in study["series"]:
                uid = study[UID]
                sid = series["SeriesInstanceUID"]
                key = hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24]
                path = output / (key + ".json")
                previous = json.loads(path.read_text()) if path.exists() else None
                if previous is not None and reusable_geometry(
                    previous, uid, series, meta["etag"], allowed
                ):
                    study_records.append((path, dict(previous), previous))
                    stats["reused_series"] += 1
                    continue
                affines = []
                sides = set()
                frame_ids = set()
                for name in series["files"]:
                    entry = f"{split}_series/{uid}/{sid}/{name}"
                    with (root / entry).open("rb") as handle:
                        ds = pydicom.dcmread(
                            handle, stop_before_pixels=True, specific_tags=tags
                        )
                    affines.append(
                        pixel_affine(ds, series["plane"], series.get("side")).tolist()
                    )
                    sides.update(explicit_header_sides(ds))
                    frame_ids.add(str(getattr(ds, "FrameOfReferenceUID", "")))
                    stats["dicom_headers_read"] += 1
                quality = alignment_quality(affines)
                record = {
                    UID: uid,
                    "SeriesInstanceUID": sid,
                    "affines_col_row_origin_lps_mm": affines,
                    "quality": quality,
                    "explicit_side": next(iter(sides)) if len(sides) == 1 else None,
                    "raw_explicit_sides": sorted(sides),
                    "frame_of_reference_uid": next(iter(frame_ids))
                    if len(frame_ids) == 1
                    else None,
                    "source_archive_etag": meta["etag"],
                    "source_builder_sha256": builder_hash,
                }
                study_records.append((path, record, previous))
                stats["built_series"] += 1
            raw = [
                {"raw_explicit_sides": record["raw_explicit_sides"]}
                for _, record, _ in study_records
            ]
            side, evidence = resolve_anatomy_laterality(raw)
            for path, record, previous in study_records:
                record.update(
                    explicit_side=side,
                    study_explicit_sides=evidence["explicit_sides"],
                    laterality_status=evidence["status"],
                )
                if record != previous:
                    save_json(path, record)
                    stats["geometry_records_written"] += 1
                records.append(
                    {
                        "path": str(path.resolve()),
                        "aligned": record["quality"]["aligned"],
                    }
                )
            print(f"geometry {len(records)} Series", flush=True)
        payload = {
            "source_manifest_sha256": file_hash(manifest),
            "series": records,
            "aligned_series": sum((r["aligned"] for r in records)),
            "scope": "physical-coordinate validity, not lesion localization accuracy",
        }
        path = output / "manifest.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous != payload:
            save_json(path, payload)
            stats["manifest_written"] = True
    finally:
        stats.update(
            seconds=time.perf_counter() - started,
            archive_opens=runtime.archive_opens - opens_before,
        )
        runtime.last_build = stats
        if owned:
            runtime.close()
    return stats
