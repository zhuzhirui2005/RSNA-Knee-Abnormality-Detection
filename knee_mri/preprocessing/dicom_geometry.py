# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""Shared, fail-closed DICOM geometry for every ConvNeXt cache builder.

The contract is deliberately narrow:

* one laterality decision is resolved per study;
* slices are ordered by physical IPP projection;
* every plane is rendered in one declared patient-space in-plane basis;
* photometric interpretation and modality scaling are explicit;
* row and column PixelSpacing are independent;
* requested crop support is proved before interpolation;
* no replicated/synthetic anatomy is allowed;
* the laterality mirror is the final pixel operation for every member.

All canonical, FOV, translation, rotation and de-roll builders import this
module.  Keeping those operations here prevents cache members from silently
acquiring different anatomy conventions.
"""

from __future__ import annotations
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import cv2
import numpy as np
import pydicom

try:
    from pydicom.dataset import Dataset as PydicomDataset
    from pydicom.datadict import tag_for_keyword
    from pydicom.pixels import pixel_array as pydicom_pixel_array
except ImportError:
    PydicomDataset = None
    tag_for_keyword = None
    pydicom_pixel_array = None
GEOMETRY_CONTRACT = "cnx_dicom_physical_geometry_v3"
POSITION_TOLERANCE_MM = 0.001
SLOTS = [
    ("Sagittal", 1),
    ("Sagittal", 0),
    ("Coronal", 1),
    ("Coronal", 0),
    ("Axial", 1),
    ("Axial", 0),
]
SLOT_ID = {slot: index for index, slot in enumerate(SLOTS)}
CARDINAL_NORMAL = {
    "Sagittal": np.asarray([1.0, 0.0, 0.0], np.float64),
    "Coronal": np.asarray([0.0, 1.0, 0.0], np.float64),
    "Axial": np.asarray([0.0, 0.0, 1.0], np.float64),
}
CANONICAL_FIRST_AXIS = {
    "Sagittal": np.asarray([0.0, -1.0, 0.0], np.float64),
    "Coronal": np.asarray([1.0, 0.0, 0.0], np.float64),
    "Axial": np.asarray([1.0, 0.0, 0.0], np.float64),
}
SUPPORTED_PHOTOMETRIC = {"MONOCHROME1", "MONOCHROME2"}
PIXEL_PAYLOAD_TAGS = (
    ()
    if tag_for_keyword is None
    else tuple(
        (
            tag_for_keyword(keyword)
            for keyword in (
                "PixelSpacing",
                "ImageOrientationPatient",
                "PhotometricInterpretation",
                "RescaleSlope",
                "RescaleIntercept",
            )
        )
    )
)
INPLANE_TRANSFORMS = {
    "identity": np.asarray([[1, 0], [0, 1]], np.int8),
    "rotate_90": np.asarray([[0, 1], [-1, 0]], np.int8),
    "rotate_180": np.asarray([[-1, 0], [0, -1]], np.int8),
    "rotate_270": np.asarray([[0, -1], [1, 0]], np.int8),
    "mirror_first": np.asarray([[-1, 0], [0, 1]], np.int8),
    "mirror_second": np.asarray([[1, 0], [0, -1]], np.int8),
    "transpose": np.asarray([[0, 1], [1, 0]], np.int8),
    "anti_transpose": np.asarray([[0, -1], [-1, 0]], np.int8),
}


class GeometryError(RuntimeError):
    """Raw headers cannot support a declared physical operation."""


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for value in iter(lambda: handle.read(chunk), b""):
            digest.update(value)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_laterality(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return None


def resolve_laterality_fields(
    laterality, image_laterality
) -> tuple[str | None, str | None, str | None]:
    """Normalize both tags independently; conflicting valid tags are unsafe."""
    first = normalize_laterality(laterality)
    second = normalize_laterality(image_laterality)
    if first is not None and second is not None and (first != second):
        raise GeometryError(f"conflicting Laterality/ImageLaterality: {first}/{second}")
    return (first or second, first, second)


def _finite_float(value, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if not np.isfinite(result):
        raise GeometryError(f"non-finite DICOM scalar: {value}")
    return result


def _float_vector(value, length: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, np.float64)
    except Exception:
        return None
    if array.shape != (length,) or not np.isfinite(array).all():
        return None
    return array


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-08:
        raise GeometryError("zero/non-finite DICOM orientation vector")
    return value / norm


def canonical_normal(
    row_axis: np.ndarray, col_axis: np.ndarray, plane: str
) -> tuple[np.ndarray, float]:
    """Orient the slice normal to the declared positive patient-space axis."""
    if plane not in CARDINAL_NORMAL:
        raise GeometryError(f"unknown anatomical plane: {plane}")
    normal = _unit(np.cross(_unit(row_axis), _unit(col_axis)))
    cardinal = CARDINAL_NORMAL[plane]
    dominant = int(np.argmax(np.abs(normal)))
    if dominant != int(np.argmax(cardinal)):
        raise GeometryError(
            f"declared {plane} plane disagrees with slice normal {normal.tolist()}"
        )
    if float(np.dot(normal, cardinal)) < 0:
        normal = -normal
    cosine = float(np.clip(np.dot(normal, cardinal), -1.0, 1.0))
    obliquity = float(np.degrees(np.arccos(cosine)))
    return (normal, obliquity)


def canonical_inplane_axes(
    row_axis: np.ndarray, col_axis: np.ndarray, plane: str
) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    """Choose the unique D4 reorientation closest to the plane's target basis.

    The operation is discrete, so small acquisition obliquity is preserved for
    the explicit de-roll branch instead of being silently removed from every
    cache member.
    """
    source_first = _unit(np.asarray(row_axis, np.float64))
    source_second = _unit(np.asarray(col_axis, np.float64))
    if abs(float(np.dot(source_first, source_second))) > 0.001:
        raise GeometryError("non-orthogonal IOP axes")
    normal, obliquity = canonical_normal(source_first, source_second, plane)
    desired = CANONICAL_FIRST_AXIS[plane]
    target_first = desired - normal * float(np.dot(desired, normal))
    target_first = _unit(target_first)
    target_second = _unit(np.cross(normal, target_first))
    best = None
    for name, transform in INPLANE_TRANSFORMS.items():
        first = _unit(transform[0, 0] * source_first + transform[0, 1] * source_second)
        second = _unit(transform[1, 0] * source_first + transform[1, 1] * source_second)
        handedness = float(np.dot(np.cross(first, second), normal))
        if handedness < 0.995:
            continue
        score = float(np.dot(first, target_first)) + float(
            np.dot(second, target_second)
        )
        candidate = (score, name, first, second)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise GeometryError("no canonical in-plane orientation")
    _, name, first, second = best
    residual = float(
        np.degrees(
            np.arctan2(
                float(np.dot(target_first, second)), float(np.dot(target_first, first))
            )
        )
    )
    if abs(residual) > 45.0001:
        raise GeometryError(
            f"in-plane orientation residual exceeds D4 bound: {residual}"
        )
    return (first, second, name, residual, obliquity)


@dataclass(frozen=True)
class HeaderRecord:
    path: str
    sop_uid: str
    instance_number: int
    ipp: tuple[float, float, float] | None
    iop: tuple[float, float, float, float, float, float] | None
    spacing: tuple[float, float] | None
    rows: int
    columns: int
    laterality: str | None
    laterality_tag: str | None
    image_laterality_tag: str | None
    photometric_interpretation: str
    rescale_slope: float
    rescale_intercept: float


@dataclass(frozen=True)
class SeriesOrder:
    files: tuple[str, ...]
    positions_mm: tuple[float, ...]
    order_method: str
    fallback_reason: str
    canonical_normal: tuple[float, float, float] | None
    header_sha256: str
    skipped_headers: int
    laterality_values: tuple[str, ...]
    ipp_x_median: float | None
    source_first_axis: tuple[float, float, float] | None
    source_second_axis: tuple[float, float, float] | None
    canonical_first_axis: tuple[float, float, float] | None
    canonical_second_axis: tuple[float, float, float] | None
    inplane_transform: str
    residual_roll_degrees: float | None
    normal_obliquity_degrees: float | None
    photometric_values: tuple[str, ...]
    rescale_slope_range: tuple[float, float]
    rescale_intercept_range: tuple[float, float]
    rows: int
    columns: int
    spacing: tuple[float, float]


@dataclass(frozen=True)
class StudySide:
    side: str | None
    source: str
    tag_values: tuple[str, ...]
    sagittal_ipp_x_median: float | None


@dataclass(frozen=True)
class PixelPayload:
    image: np.ndarray
    spacing: tuple[float, float]
    row_axis: np.ndarray
    col_axis: np.ndarray
    photometric_interpretation: str = "MONOCHROME2"
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0


@dataclass(frozen=True)
class RenderReceipt:
    available: bool
    reason: str
    requested_crop_mm: float
    realized_fov_y_mm: float
    realized_fov_x_mm: float
    angle_degrees: float
    shift_x_mm: float
    shift_y_mm: float
    realized_shift_x_mm: float
    realized_shift_y_mm: float
    source_rows: int
    source_columns: int
    spacing_row_mm: float
    spacing_column_mm: float
    crop_bounds_yx: tuple[int, int, int, int] | None
    source_polygon_xy: tuple[tuple[float, float], ...] | None
    border_contact_fraction: float


def read_header(path: Path | str) -> HeaderRecord:
    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        specific_tags=[
            "SOPInstanceUID",
            "InstanceNumber",
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
            "Rows",
            "Columns",
            "Laterality",
            "ImageLaterality",
            "PhotometricInterpretation",
            "RescaleSlope",
            "RescaleIntercept",
        ],
    )
    return header_from_dataset(ds, str(path))


def header_from_dataset(ds, path: str) -> HeaderRecord:
    """Same geometry parser for file-backed and archive-backed DICOMs."""
    try:
        instance = int(ds.InstanceNumber)
    except Exception as exc:
        raise GeometryError(f"missing/non-integer InstanceNumber: {path}") from exc
    ipp = _float_vector(getattr(ds, "ImagePositionPatient", None), 3)
    iop = _float_vector(getattr(ds, "ImageOrientationPatient", None), 6)
    spacing = _float_vector(getattr(ds, "PixelSpacing", None), 2)
    if spacing is not None and (spacing <= 0).any():
        spacing = None
    laterality, laterality_tag, image_laterality_tag = resolve_laterality_fields(
        getattr(ds, "Laterality", None), getattr(ds, "ImageLaterality", None)
    )
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photometric not in SUPPORTED_PHOTOMETRIC:
        raise GeometryError(
            f"unsupported PhotometricInterpretation {photometric!r}: {path}"
        )
    slope = _finite_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _finite_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    if slope == 0:
        raise GeometryError(f"zero RescaleSlope: {path}")
    return HeaderRecord(
        path=str(path),
        sop_uid=str(getattr(ds, "SOPInstanceUID", "")),
        instance_number=instance,
        ipp=None if ipp is None else tuple((float(x) for x in ipp)),
        iop=None if iop is None else tuple((float(x) for x in iop)),
        spacing=None if spacing is None else tuple((float(x) for x in spacing)),
        rows=int(getattr(ds, "Rows", 0) or 0),
        columns=int(getattr(ds, "Columns", 0) or 0),
        laterality=laterality,
        laterality_tag=laterality_tag,
        image_laterality_tag=image_laterality_tag,
        photometric_interpretation=photometric,
        rescale_slope=slope,
        rescale_intercept=intercept,
    )


def _header_payload(record: HeaderRecord) -> dict:
    value = asdict(record)
    value["path"] = Path(record.path).name
    return value


def _verified_instance_fallback(
    records: Sequence[HeaderRecord], plane: str, tolerance_mm: float
) -> tuple[
    list[HeaderRecord],
    list[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    float,
    float,
]:
    """Infer missing positions only when available geometry proves the order."""
    physical = [
        record
        for record in records
        if record.ipp is not None and record.iop is not None
    ]
    if len(physical) < 2:
        raise GeometryError(
            "InstanceNumber fallback lacks two physical IPP/IOP anchors"
        )
    first_iop = np.asarray(physical[0].iop, np.float64)
    source_first = _unit(first_iop[:3])
    source_second = _unit(first_iop[3:])
    normal, _ = canonical_normal(source_first, source_second, plane)
    canonical_first, canonical_second, operation, residual, obliquity = (
        canonical_inplane_axes(source_first, source_second, plane)
    )
    anchors = []
    for record in physical:
        iop = np.asarray(record.iop, np.float64)
        candidate_first = _unit(iop[:3])
        candidate_second = _unit(iop[3:])
        if (
            float(np.dot(candidate_first, source_first)) < 0.995
            or float(np.dot(candidate_second, source_second)) < 0.995
        ):
            raise GeometryError("InstanceNumber fallback has inconsistent in-plane IOP")
        position = float(np.dot(np.asarray(record.ipp, np.float64), normal))
        anchors.append((record.instance_number, position))
    anchors.sort()
    instances = np.asarray([value[0] for value in anchors], np.float64)
    positions = np.asarray([value[1] for value in anchors], np.float64)
    delta = np.diff(positions)
    if not (np.all(delta > tolerance_mm) or np.all(delta < -tolerance_mm)):
        raise GeometryError("InstanceNumber fallback disagrees with physical IPP order")
    design = np.stack((instances, np.ones_like(instances)), axis=1)
    slope, intercept = np.linalg.lstsq(design, positions, rcond=None)[0]
    fitted = design @ np.asarray([slope, intercept])
    typical = float(np.median(np.abs(delta))) if len(delta) else 0.0
    residual_limit = max(0.1, 0.1 * typical)
    if (
        abs(float(slope)) <= tolerance_mm
        or float(np.max(np.abs(fitted - positions))) > residual_limit
    ):
        raise GeometryError("InstanceNumber fallback is not physically linear enough")
    reverse = bool(slope < 0)
    ordered = sorted(records, key=lambda value: value.instance_number, reverse=reverse)
    inferred = [float(slope * value.instance_number + intercept) for value in ordered]
    return (
        ordered,
        inferred,
        normal,
        canonical_first,
        canonical_second,
        operation,
        residual,
        obliquity,
    )


def order_series(
    series_dir: Path,
    plane: str | None,
    position_tolerance_mm: float = POSITION_TOLERANCE_MM,
    allow_instance_fallback: bool = False,
    provided_records: Sequence[HeaderRecord] | None = None,
) -> SeriesOrder:
    """Return physically ordered DICOM files and a header-geometry receipt.

    The slice normal is oriented toward a fixed patient-space cardinal axis, so
    the order does not reverse merely because a scanner writes IOP with the
    opposite sign.  If complete IPP/IOP geometry is unavailable, the fallback
    is explicit and requires physical IPP/IOP anchors that prove its direction.
    """
    records: list[HeaderRecord] = []
    skipped = 0
    if provided_records is not None:
        records = list(provided_records)
    else:
        for path in sorted(Path(series_dir).glob("*.dcm")):
            try:
                records.append(read_header(path))
            except Exception:
                skipped += 1
    if skipped:
        raise GeometryError(
            f"{skipped} unreadable DICOM header(s) in {series_dir}; partial series are forbidden"
        )
    if not records:
        raise GeometryError(f"no readable DICOM headers: {series_dir}")
    if any((record.rows <= 0 or record.columns <= 0 for record in records)):
        raise GeometryError(f"missing/non-positive Rows or Columns: {series_dir}")
    dimensions = {(record.rows, record.columns) for record in records}
    if len(dimensions) != 1:
        raise GeometryError(
            f"pixel matrix changes within series {series_dir}: {sorted(dimensions)}"
        )
    if any((record.spacing is None for record in records)):
        raise GeometryError(f"missing PixelSpacing in series: {series_dir}")
    spacings = np.asarray([record.spacing for record in records], np.float64)
    if float(np.max(np.ptp(spacings, axis=0))) > 1e-05:
        raise GeometryError(f"PixelSpacing changes within series: {series_dir}")
    laterality_values = tuple(
        sorted(
            {record.laterality for record in records if record.laterality is not None}
        )
    )
    photometric_values = tuple(
        sorted({record.photometric_interpretation for record in records})
    )
    if len(photometric_values) != 1:
        raise GeometryError(
            f"PhotometricInterpretation changes within series: {series_dir}"
        )
    slopes = np.asarray([record.rescale_slope for record in records], np.float64)
    intercepts = np.asarray(
        [record.rescale_intercept for record in records], np.float64
    )
    ipp_x = [record.ipp[0] for record in records if record.ipp is not None]
    ipp_x_median = float(np.median(ipp_x)) if ipp_x else None
    normal = None
    source_first = source_second = None
    output_first = output_second = None
    inplane_transform = ""
    residual_roll = normal_obliquity = None
    complete_geometry = all((r.ipp is not None and r.iop is not None for r in records))
    fallback_reason = ""
    positions: list[float] = []
    if complete_geometry:
        iop0 = np.asarray(records[0].iop, np.float64)
        first_row = _unit(iop0[:3])
        first_column = _unit(iop0[3:])
        if abs(float(np.dot(first_row, first_column))) > 0.001:
            raise GeometryError(f"non-orthogonal IOP axes: {series_dir}")
        normal, normal_obliquity = canonical_normal(first_row, first_column, str(plane))
        output_first, output_second, inplane_transform, residual_roll, _ = (
            canonical_inplane_axes(first_row, first_column, str(plane))
        )
        source_first, source_second = (first_row, first_column)
        for record in records:
            iop = np.asarray(record.iop, np.float64)
            candidate_row = _unit(iop[:3])
            candidate_column = _unit(iop[3:])
            candidate = _unit(np.cross(candidate_row, candidate_column))
            if abs(float(np.dot(candidate, normal))) < 0.995:
                raise GeometryError(f"slice normals vary within series: {series_dir}")
            if (
                float(np.dot(candidate_row, first_row)) < 0.995
                or float(np.dot(candidate_column, first_column)) < 0.995
            ):
                raise GeometryError(
                    f"in-plane IOP axes vary within series: {series_dir}"
                )
            positions.append(float(np.dot(np.asarray(record.ipp), normal)))
        order = np.argsort(np.asarray(positions), kind="stable")
        sorted_positions = np.asarray(positions)[order]
        if len(sorted_positions) > 1:
            delta = np.diff(sorted_positions)
            if np.any(delta <= position_tolerance_mm):
                raise GeometryError(
                    f"duplicate/non-increasing IPP projection in {series_dir}; minimum delta={float(delta.min()):.6g} mm"
                )
        records = [records[int(index)] for index in order]
        positions = [
            float(sorted_positions[index]) for index in range(len(sorted_positions))
        ]
        method = "ipp_projection_cardinalized"
    else:
        if not allow_instance_fallback:
            raise GeometryError(f"IPP/IOP incomplete: {series_dir}")
        instances = [record.instance_number for record in records]
        if len(set(instances)) != len(instances):
            raise GeometryError(f"InstanceNumber fallback has duplicates: {series_dir}")
        (
            records,
            positions,
            normal,
            output_first,
            output_second,
            inplane_transform,
            residual_roll,
            normal_obliquity,
        ) = _verified_instance_fallback(records, str(plane), position_tolerance_mm)
        first_physical = next((record for record in records if record.iop is not None))
        first_iop = np.asarray(first_physical.iop, np.float64)
        source_first = _unit(first_iop[:3])
        source_second = _unit(first_iop[3:])
        method = "instance_number_physically_verified"
        fallback_reason = "incomplete_geometry_with_verified_physical_anchors"
    payload = {
        "contract": GEOMETRY_CONTRACT,
        "plane": plane,
        "order_method": method,
        "fallback_reason": fallback_reason,
        "normal": None if normal is None else normal.tolist(),
        "source_first_axis": None if source_first is None else source_first.tolist(),
        "source_second_axis": None if source_second is None else source_second.tolist(),
        "canonical_first_axis": None if output_first is None else output_first.tolist(),
        "canonical_second_axis": None
        if output_second is None
        else output_second.tolist(),
        "inplane_transform": inplane_transform,
        "residual_roll_degrees": residual_roll,
        "normal_obliquity_degrees": normal_obliquity,
        "photometric_values": photometric_values,
        "rescale_slope_range": [float(slopes.min()), float(slopes.max())],
        "rescale_intercept_range": [float(intercepts.min()), float(intercepts.max())],
        "laterality_values": laterality_values,
        "ipp_x_median": ipp_x_median,
        "headers": [_header_payload(record) for record in records],
    }
    return SeriesOrder(
        files=tuple((record.path for record in records)),
        positions_mm=tuple(positions),
        order_method=method,
        fallback_reason=fallback_reason,
        canonical_normal=None if normal is None else tuple((float(x) for x in normal)),
        header_sha256=canonical_json_sha256(payload),
        skipped_headers=skipped,
        laterality_values=laterality_values,
        ipp_x_median=ipp_x_median,
        source_first_axis=None
        if source_first is None
        else tuple((float(x) for x in source_first)),
        source_second_axis=None
        if source_second is None
        else tuple((float(x) for x in source_second)),
        canonical_first_axis=None
        if output_first is None
        else tuple((float(x) for x in output_first)),
        canonical_second_axis=None
        if output_second is None
        else tuple((float(x) for x in output_second)),
        inplane_transform=inplane_transform,
        residual_roll_degrees=None if residual_roll is None else float(residual_roll),
        normal_obliquity_degrees=None
        if normal_obliquity is None
        else float(normal_obliquity),
        photometric_values=photometric_values,
        rescale_slope_range=(float(slopes.min()), float(slopes.max())),
        rescale_intercept_range=(float(intercepts.min()), float(intercepts.max())),
        rows=int(records[0].rows),
        columns=int(records[0].columns),
        spacing=tuple((float(x) for x in records[0].spacing)),
    )


def decide_study_side(
    study: str, tag_values: Iterable[str], sagittal_x: Iterable[float]
) -> StudySide:
    """Pure study-level laterality decision shared by manifest and callers."""
    tags = tuple(sorted({value for value in tag_values if value in {"L", "R"}}))
    sagittal_values = [
        float(value) for value in sagittal_x if np.isfinite(float(value))
    ]
    sagittal_median = float(np.median(sagittal_values)) if sagittal_values else None
    if len(tags) == 1:
        return StudySide(tags[0], "dicom_laterality_tag", tags, sagittal_median)
    if len(tags) > 1:
        return StudySide(None, "conflicting_laterality_tags", tags, sagittal_median)
    if sagittal_median is not None:
        return StudySide(
            "L" if sagittal_median >= 0 else "R",
            "sagittal_ipp_x",
            tags,
            sagittal_median,
        )
    return StudySide(None, "unknown_no_flip", tags, None)


def resolve_study_side(
    study: str, records: Sequence[Mapping], series_root: Path
) -> StudySide:
    """Resolve side once per study; unknown/conflicting metadata never flips."""
    tag_values: set[str] = set()
    sagittal_x: list[float] = []
    for record in records:
        uid = str(record["SeriesInstanceUID"])
        try:
            ordered = order_series(
                Path(series_root) / str(study) / uid,
                str(record.get("Anatomical_Plane", "")),
            )
        except GeometryError:
            continue
        tag_values.update(ordered.laterality_values)
        if (
            str(record.get("Anatomical_Plane", "")) == "Sagittal"
            and ordered.ipp_x_median is not None
        ):
            sagittal_x.append(float(ordered.ipp_x_median))
    return decide_study_side(study, tag_values, sagittal_x)


def flip_for_plane(plane: str, side: str | None) -> bool:
    return str(plane) != "Sagittal" and side == "R"


def read_pixels(path: Path | str) -> PixelPayload:
    try:
        if pydicom_pixel_array is None or PydicomDataset is None:
            ds = pydicom.dcmread(str(path))
            image = ds.pixel_array.astype(np.float32)
        else:
            ds = PydicomDataset()
            image = pydicom_pixel_array(
                str(path), ds_out=ds, specific_tags=list(PIXEL_PAYLOAD_TAGS)
            ).astype(np.float32)
    except Exception as exc:
        raise GeometryError(f"pixel decode failed: {path}") from exc
    return pixels_from_dataset(ds, image=image, path=str(path))


def pixels_from_dataset(ds, image=None, path="archive") -> PixelPayload:
    """Apply the identical modality/photometric contract to an in-memory DICOM."""
    if image is None:
        image = ds.pixel_array.astype(np.float32)
    spacing = _float_vector(getattr(ds, "PixelSpacing", None), 2)
    if spacing is None or (spacing <= 0).any():
        raise GeometryError(f"missing/invalid PixelSpacing: {path}")
    iop = _float_vector(getattr(ds, "ImageOrientationPatient", None), 6)
    if iop is None:
        raise GeometryError(f"missing/invalid ImageOrientationPatient: {path}")
    row = _unit(iop[:3])
    col = _unit(iop[3:])
    if abs(float(np.dot(row, col))) > 0.001:
        raise GeometryError(f"non-orthogonal IOP axes: {path}")
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photometric not in SUPPORTED_PHOTOMETRIC:
        raise GeometryError(
            f"unsupported PhotometricInterpretation {photometric!r}: {path}"
        )
    slope = _finite_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = _finite_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    if slope == 0:
        raise GeometryError(f"zero RescaleSlope: {path}")
    image = image * slope + intercept
    if not np.isfinite(image).all():
        raise GeometryError(f"non-finite modality-scaled pixels: {path}")
    if photometric == "MONOCHROME1":
        image = float(image.min()) + float(image.max()) - image
    return PixelPayload(
        image=image,
        spacing=(float(spacing[0]), float(spacing[1])),
        row_axis=row,
        col_axis=col,
        photometric_interpretation=photometric,
        rescale_slope=slope,
        rescale_intercept=intercept,
    )


def rotated_axes(
    row: np.ndarray, col: np.ndarray, angle_radians: float
) -> tuple[np.ndarray, np.ndarray]:
    cosine, sine = (np.cos(angle_radians), np.sin(angle_radians))
    return (row * cosine + col * sine, -row * sine + col * cosine)


def deroll_angle_degrees(row: np.ndarray, col: np.ndarray, plane: str) -> float:
    _, _, _, residual, _ = canonical_inplane_axes(row, col, str(plane))
    return float(residual)


def _apply_inplane_transform(image: np.ndarray, operation: str) -> np.ndarray:
    """Apply the selected D4 operation exactly, without interpolation."""
    if operation == "identity":
        result = image
    elif operation == "rotate_90":
        result = np.rot90(image, 1)
    elif operation == "rotate_180":
        result = np.rot90(image, 2)
    elif operation == "rotate_270":
        result = np.rot90(image, 3)
    elif operation == "mirror_first":
        result = image[:, ::-1]
    elif operation == "mirror_second":
        result = image[::-1, :]
    elif operation == "transpose":
        result = image.T
    elif operation == "anti_transpose":
        result = image[::-1, ::-1].T
    else:
        raise GeometryError(f"unknown in-plane transform: {operation}")
    return np.ascontiguousarray(result)


def canonicalize_payload(payload: PixelPayload, plane: str) -> PixelPayload:
    """Put a slice in the declared discrete patient-space image basis.

    Performing the D4 operation before the affine preserves pixels exactly and
    correctly swaps row/column spacing and dimensions for 90-degree variants.
    """
    first, second, operation, _, _ = canonical_inplane_axes(
        payload.row_axis, payload.col_axis, str(plane)
    )
    transform = INPLANE_TRANSFORMS[operation]
    ps_row, ps_col = payload.spacing
    output_col_spacing = (
        abs(int(transform[0, 0])) * ps_col + abs(int(transform[0, 1])) * ps_row
    )
    output_row_spacing = (
        abs(int(transform[1, 0])) * ps_col + abs(int(transform[1, 1])) * ps_row
    )
    return PixelPayload(
        image=_apply_inplane_transform(payload.image, operation),
        spacing=(float(output_row_spacing), float(output_col_spacing)),
        row_axis=first,
        col_axis=second,
        photometric_interpretation=payload.photometric_interpretation,
        rescale_slope=payload.rescale_slope,
        rescale_intercept=payload.rescale_intercept,
    )


def _affine_destination_to_source(
    payload: PixelPayload, plane: str, angle_degrees: float
) -> np.ndarray:
    rows, columns = payload.image.shape
    ps_row, ps_col = payload.spacing
    base_row, base_col, _, _, _ = canonical_inplane_axes(
        payload.row_axis, payload.col_axis, str(plane)
    )
    out_row, out_col = rotated_axes(base_row, base_col, np.radians(angle_degrees))
    matrix = np.asarray(
        [
            [
                float(np.dot(out_row, payload.row_axis)),
                float(np.dot(out_col, payload.row_axis)) * ps_row / ps_col,
                0.0,
            ],
            [
                float(np.dot(out_row, payload.col_axis)) * ps_col / ps_row,
                float(np.dot(out_col, payload.col_axis)),
                0.0,
            ],
        ],
        np.float64,
    )
    center = np.asarray([(columns - 1) / 2.0, (rows - 1) / 2.0])
    matrix[:, 2] = center - matrix[:, :2] @ center
    return matrix


def _crop_bounds(
    payload: PixelPayload, crop_mm: float, shift_x_mm: float, shift_y_mm: float
) -> tuple[int, int, int, int]:
    rows, columns = payload.image.shape
    ps_row, ps_col = payload.spacing
    height = max(1, int(round(float(crop_mm) / ps_row)))
    width = max(1, int(round(float(crop_mm) / ps_col)))
    center_y = (rows - 1) / 2.0 + float(shift_y_mm) / ps_row
    center_x = (columns - 1) / 2.0 + float(shift_x_mm) / ps_col
    y0 = int(round(center_y - (height - 1) / 2.0))
    x0 = int(round(center_x - (width - 1) / 2.0))
    return (y0, y0 + height, x0, x0 + width)


def _support_polygon(
    matrix: np.ndarray, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    corners = np.asarray(
        [[x0, y0, 1.0], [x1 - 1, y0, 1.0], [x1 - 1, y1 - 1, 1.0], [x0, y1 - 1, 1.0]],
        np.float64,
    )
    return corners @ matrix.T


def render_payload(
    payload: PixelPayload,
    *,
    plane: str,
    side: str | None,
    crop_mm: float,
    angle_degrees: float = 0.0,
    shift_x_mm: float = 0.0,
    shift_y_mm: float = 0.0,
    output_size: int = 336,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray | None, RenderReceipt]:
    """Canonicalize and render one decoded DICOM with true source support."""
    canonical = canonicalize_payload(payload, plane)
    return render_canonical_payload(
        canonical,
        plane=plane,
        side=side,
        crop_mm=crop_mm,
        angle_degrees=angle_degrees,
        shift_x_mm=shift_x_mm,
        shift_y_mm=shift_y_mm,
        output_size=output_size,
        interpolation=interpolation,
    )


def render_canonical_payload(
    payload: PixelPayload,
    *,
    plane: str,
    side: str | None,
    crop_mm: float,
    angle_degrees: float = 0.0,
    shift_x_mm: float = 0.0,
    shift_y_mm: float = 0.0,
    output_size: int = 336,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray | None, RenderReceipt]:
    """Render a payload already transformed to the declared D4 basis."""
    rows, columns = payload.image.shape
    ps_row, ps_col = payload.spacing
    bounds = _crop_bounds(payload, crop_mm, shift_x_mm, shift_y_mm)
    y0, y1, x0, x1 = bounds
    matrix = _affine_destination_to_source(payload, plane, angle_degrees)
    polygon = _support_polygon(matrix, bounds)
    bounds_inside = y0 >= 0 and x0 >= 0 and (y1 <= rows) and (x1 <= columns)
    polygon_inside = (
        float(polygon[:, 0].min()) >= 0.0
        and float(polygon[:, 1].min()) >= 0.0
        and (float(polygon[:, 0].max()) <= columns - 1)
        and (float(polygon[:, 1].max()) <= rows - 1)
    )
    realized_y = max(0, y1 - y0) * ps_row
    realized_x = max(0, x1 - x0) * ps_col
    realized_center_y = (y0 + y1 - 1) / 2.0
    realized_center_x = (x0 + x1 - 1) / 2.0
    realized_shift_y = (realized_center_y - (rows - 1) / 2.0) * ps_row
    realized_shift_x = (realized_center_x - (columns - 1) / 2.0) * ps_col
    common = dict(
        requested_crop_mm=float(crop_mm),
        realized_fov_y_mm=float(realized_y),
        realized_fov_x_mm=float(realized_x),
        angle_degrees=float(angle_degrees),
        shift_x_mm=float(shift_x_mm),
        shift_y_mm=float(shift_y_mm),
        realized_shift_x_mm=float(realized_shift_x),
        realized_shift_y_mm=float(realized_shift_y),
        source_rows=int(rows),
        source_columns=int(columns),
        spacing_row_mm=float(ps_row),
        spacing_column_mm=float(ps_col),
        crop_bounds_yx=tuple((int(x) for x in bounds)),
        source_polygon_xy=tuple(
            (tuple((float(v) for v in point)) for point in polygon)
        ),
        border_contact_fraction=0.0,
    )
    if not bounds_inside or not polygon_inside:
        reason = (
            "destination_crop_outside_native_frame"
            if not bounds_inside
            else "rotated_crop_polygon_outside_genuine_source_support"
        )
        return (None, RenderReceipt(False, reason, **common))
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64)
    if float(np.max(np.abs(matrix - identity))) < 1e-10:
        transformed = payload.image
    else:
        transformed = cv2.warpAffine(
            payload.image,
            matrix,
            (columns, rows),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    crop = transformed[y0:y1, x0:x1]
    if crop.shape != (y1 - y0, x1 - x0) or crop.size == 0:
        return (None, RenderReceipt(False, "unexpected_crop_shape", **common))
    sample = crop[::4, ::4]
    low, high = np.percentile(sample, [1, 99])
    scaled = np.clip((crop - low) / max(float(high - low), 1e-06), 0, 1)
    image = cv2.resize(scaled, (output_size, output_size), interpolation=interpolation)
    image = np.clip(image, 0, 1)
    if flip_for_plane(plane, side):
        image = image[:, ::-1]
    result = (image * 255).astype(np.uint8)
    return (np.ascontiguousarray(result), RenderReceipt(True, "", **common))


def render_member(
    path: Path | str,
    *,
    plane: str,
    side: str | None,
    crop_mm: float,
    angle_degrees: float = 0.0,
    shift_x_mm: float = 0.0,
    shift_y_mm: float = 0.0,
    output_size: int = 336,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray | None, RenderReceipt]:
    """Decode and render one DICOM under the shared physical contract."""
    try:
        payload = read_pixels(path)
    except GeometryError as exc:
        receipt = RenderReceipt(
            False,
            str(exc),
            float(crop_mm),
            0.0,
            0.0,
            float(angle_degrees),
            float(shift_x_mm),
            float(shift_y_mm),
            0.0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            None,
            None,
            0.0,
        )
        return (None, receipt)
    return render_payload(
        payload,
        plane=plane,
        side=side,
        crop_mm=crop_mm,
        angle_degrees=angle_degrees,
        shift_x_mm=shift_x_mm,
        shift_y_mm=shift_y_mm,
        output_size=output_size,
        interpolation=interpolation,
    )


def pick_depth_indices(
    length: int, n_slice: int = 16, band: tuple[float, float] = (0.12, 0.88)
) -> tuple[list[int], int]:
    if length <= 0:
        return ([], 0)
    start = int(round(float(band[0]) * (length - 1)))
    stop = int(round(float(band[1]) * (length - 1)))
    available = list(range(start, stop + 1))
    if len(available) >= n_slice:
        picks = [
            available[int(round(value))]
            for value in np.linspace(0, len(available) - 1, n_slice)
        ]
        return (picks, 0)
    return (available, (n_slice - len(available)) // 2)


def source_receipt(path: Path | str, plane: str) -> dict:
    """Small first-slice physical receipt used by the manifest builder."""
    payload = read_pixels(path)
    rows, columns = payload.image.shape
    first, second, operation, angle, obliquity = canonical_inplane_axes(
        payload.row_axis, payload.col_axis, plane
    )
    return {
        "rows": int(rows),
        "columns": int(columns),
        "spacing_row_mm": float(payload.spacing[0]),
        "spacing_column_mm": float(payload.spacing[1]),
        "source_fov_y_mm": float(rows * payload.spacing[0]),
        "source_fov_x_mm": float(columns * payload.spacing[1]),
        "roll_degrees": float(angle),
        "source_first_axis_json": json.dumps(payload.row_axis.tolist()),
        "source_second_axis_json": json.dumps(payload.col_axis.tolist()),
        "canonical_first_axis_json": json.dumps(first.tolist()),
        "canonical_second_axis_json": json.dumps(second.tolist()),
        "inplane_transform": operation,
        "normal_obliquity_degrees": float(obliquity),
        "photometric_interpretation": payload.photometric_interpretation,
        "rescale_slope": float(payload.rescale_slope),
        "rescale_intercept": float(payload.rescale_intercept),
    }


def initialize_cpu_worker() -> None:
    """Prevent process-pool oversubscription in OpenCV/BLAS kernels."""
    cv2.setNumThreads(1)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def fsync_path(path: Path) -> None:
    """Durably persist a completed artifact and its directory entry."""
    path = Path(path)
    fd = path.open("rb")
    try:
        os.fsync(fd.fileno())
    finally:
        fd.close()
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_replace(temporary: Path, destination: Path) -> None:
    """Atomically publish a fully flushed file and its directory entry."""
    temporary = Path(temporary)
    destination = Path(destination)
    fsync_path(temporary)
    os.replace(temporary, destination)
    directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def pwrite_all(fd: int, payload: bytes | memoryview, offset: int) -> None:
    """Write the complete payload at an explicit offset or fail loudly."""
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], int(offset) + written)
        if count <= 0:
            raise OSError(f"short positional write at {offset + written}: {count}")
        written += count


def require_npy(
    path: Path | str, shape: tuple[int, ...], dtype, *, mmap_mode: str | None = "r"
) -> np.ndarray:
    """Open an array only when its on-disk header matches the exact contract."""
    path = Path(path)
    if not path.is_file():
        raise GeometryError(f"missing NumPy artifact: {path}")
    try:
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except Exception as exc:
        raise GeometryError(f"unreadable NumPy artifact: {path}") from exc
    expected_dtype = np.dtype(dtype)
    if tuple(value.shape) != tuple(shape) or value.dtype != expected_dtype:
        raise GeometryError(
            f"NumPy artifact contract mismatch for {path}: {value.shape}/{value.dtype} != {shape}/{expected_dtype}"
        )
    return value


def ensure_unique_trainable_parameters(groups: Iterable[Mapping]) -> None:
    seen: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in seen:
                raise GeometryError("parameter appears in multiple optimizer groups")
            seen.add(identity)
