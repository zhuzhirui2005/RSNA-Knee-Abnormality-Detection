import json
import numpy as np
import pandas as pd
import pytest
import torch
from knee_mri.constants import UID, LABELS
from knee_mri.data import write_cache, read_cache, make_windows
from knee_mri.inference import probabilities_from_logits, publish_predictions
from knee_mri.preprocessing.input_provenance import view_geometry
from knee_mri.training import read_labels, blend_teacher


def test_cache_is_lossless(tmp_path):
    pixels = np.random.default_rng(11).integers(0, 256, (5, 32, 32), dtype=np.uint8)
    metadata = np.zeros((5, 12), np.float32)
    item = {
        "px": torch.from_numpy(pixels),
        "meta": torch.from_numpy(metadata),
        "blocks": [{"n_slices": 5, "views": [[0, 0]]}],
    }
    path = tmp_path / "case.npz"
    write_cache(path, "synthetic", item, ["L"], True)
    restored, meta, blocks = read_cache(path)
    np.testing.assert_array_equal(restored, pixels)
    np.testing.assert_array_equal(meta, metadata)
    assert blocks == item["blocks"]


def test_physical_windows_and_edge_repetition():
    # Half-mm slices use seven-index neighbors to represent approximately 3.5 mm.
    pixels = np.arange(20, dtype=np.uint8)[:, None, None] * np.ones(
        (20, 8, 8), np.uint8
    )
    meta = np.zeros((20, 12), np.float32)
    meta[:, 5] = np.log1p(0.5)
    meta[:, 10] = 20
    images, features = make_windows(pixels, meta, [{"n_slices": 20, "views": [[0, 0]]}])
    assert images[0, :, 0, 0].tolist() == [0, 1, 8]
    assert features.shape[1] == 9
    assert images[-1, :, 0, 0].tolist() == [12, 19, 19]


def test_fold_aggregation_uses_logits():
    logits = np.array([[0.0], [4.0]], np.float32)
    actual = probabilities_from_logits(logits)
    assert actual[0] == pytest.approx(1 / (1 + np.exp(-2)), abs=1e-7)
    assert abs(actual[0] - torch.sigmoid(torch.tensor(logits)).mean().item()) > 0.1


def test_anisotropic_letterbox_and_crop():
    geometry = view_geometry((200, 100), (1.0, 2.0), 384)
    assert geometry.resized_shape == (384, 384)
    local = view_geometry((200, 100), (1.0, 2.0), 384, crop_mm=100)
    assert local.crop_shape == (100, 50)
    assert local.extent == 100


def test_missing_prediction_never_overwrites_result(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text("previous valid result")
    with pytest.raises(ValueError):
        publish_predictions(path, ["a", "b"], np.full((1, 12), 0.5))
    assert path.read_text() == "previous valid result"


def test_group_leakage_is_rejected(tmp_path):
    data = pd.DataFrame([[0.2] * 12, [0.8] * 12], columns=LABELS)
    data[UID] = ["a", "b"]
    data["fold"] = [0, 1]
    data["group"] = ["same-patient"] * 2
    path = tmp_path / "labels.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="crosses"):
        read_labels(path)


def test_teacher_blend_keeps_only_training_partition(tmp_path):
    training = pd.DataFrame([[0.2] * 12, [0.8] * 12], columns=LABELS, index=["a", "b"])
    teacher = pd.DataFrame([[0.6] * 12, [0.4] * 12], columns=LABELS, index=["a", "b"])
    path = tmp_path / "teacher.csv"
    teacher.rename_axis(UID).to_csv(path)
    result = blend_teacher(training, path, 0.25)
    np.testing.assert_allclose(result[:, 0], [0.3, 0.7], atol=1e-7)
    teacher.loc["held-out-study"] = 0.5
    teacher.rename_axis(UID).to_csv(path)
    with pytest.raises(ValueError, match="training partition"):
        blend_teacher(training, path, 0.25)
