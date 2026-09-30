"""Object detection and pose estimation on COCO-style archives.

These cover the path Vision Studio drives: a ZIP of images plus an annotation JSON,
trained with Faster / Keypoint R-CNN.  Every trainer here is built with ``weights=None``
and a small ``image_size`` so the suite never downloads pretrained weights and stays
fast enough to run on every push.
"""

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from src.engines.vision import (
    CocoDetectionDataset,
    CVAutoMLTrainer,
    detection_collate,
    draw_detections,
    find_coco_annotation,
    score_detections,
)

CATEGORIES = [
    {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 1, "name": "person", "supercategory": "body"},
]
# After the background shift: 1=person (id 1), 2=bicycle (id 3).
KEYPOINT_NAMES = ["head", "shoulder", "hip"]
SOURCE_W, SOURCE_H = 40, 30
TRAIN_IMAGE_SIZE = 64


def _annotation(index, image_id, category_id, with_keypoints):
    box = [2.0 + index, 1.0 + index, 10.0, 6.0]  # x, y, w, h in source pixels
    ann = {
        "id": image_id * 10 + index,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": box,
        "area": box[2] * box[3],
        "iscrowd": 0,
    }
    if with_keypoints:
        ann["num_keypoints"] = 3
        ann["keypoints"] = [
            box[0] + 1, box[1] + 1, 2,
            box[0] + 4, box[1] + 3, 2,
            box[0] + 7, box[1] + 5, 1,
        ]
    return ann


def _write_coco_bundle(root, images=6, with_keypoints=True, nested=True):
    """images/ and annotations/ under an archive-style top folder, non-square images."""
    image_dir = root / "dataset" / "images"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    coco_images, coco_annotations = [], []
    ann_id = 1
    for i in range(images):
        arr = rng.integers(0, 255, (SOURCE_H, SOURCE_W, 3), dtype=np.uint8)
        Image.fromarray(arr).convert("RGB").save(image_dir / f"img_{i}.png")
        coco_images.append({
            "id": i + 1,
            "file_name": f"img_{i}.png",
            "width": SOURCE_W,
            "height": SOURCE_H,
        })
        for slot, category in enumerate((1, 3)):
            ann = _annotation(slot, i + 1, category, with_keypoints)
            ann["id"] = ann_id
            ann_id += 1
            coco_annotations.append(ann)
    payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES,
        "keypoints": KEYPOINT_NAMES,
    }
    target = root / "dataset" / "annotations" if nested else image_dir
    target.mkdir(parents=True, exist_ok=True)
    annotation_path = target / "instances.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    return root / "dataset", annotation_path


def _trainer(task_type):
    return CVAutoMLTrainer(task_type=task_type, weights=None, image_size=TRAIN_IMAGE_SIZE)


def _train(task_type, data_dir, **kwargs):
    params = {"n_epochs": 1, "batch_size": 2, "lr": 0.01, "optimizer_name": "sgd"}
    params.update(kwargs)
    return _trainer(task_type).train(data_dir=str(data_dir), **params)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_coco_dataset_shifts_category_ids_behind_the_background_label(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    dataset = CocoDetectionDataset(str(data_dir), str(annotation))

    assert dataset.class_names == ["background", "person", "bicycle"]
    assert dataset.label_by_id == {1: 1, 3: 2}
    _, target = dataset[0]
    assert set(target["labels"].tolist()) == {1, 2}
    assert target["labels"].dtype == torch.int64


def test_coco_dataset_converts_xywh_to_xyxy_in_resized_space(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    dataset = CocoDetectionDataset(str(data_dir), str(annotation), image_size=32)
    image, target = dataset[0]

    assert image.shape == (3, 32, 32)
    # Annotation 0 of image 1 is [2, 1, 10, 6] in a 40x30 source.
    expected = torch.tensor([
        2 * 32 / SOURCE_W, 1 * 32 / SOURCE_H,
        12 * 32 / SOURCE_W, 7 * 32 / SOURCE_H,
    ])
    assert torch.allclose(target["boxes"][0], expected)
    assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
    assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()


def test_coco_dataset_keypoints_are_rescaled_and_padded_to_one_width(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    dataset = CocoDetectionDataset(
        str(data_dir), str(annotation), image_size=32, require_keypoints=True)
    _image, target = dataset[0]

    assert dataset.num_keypoints == 3
    assert target["keypoints"].shape == (2, 3, 3)
    assert target["keypoints"].dtype == torch.float32
    assert target["num_keypoints"].tolist() == [3, 3]
    first = target["keypoints"][0, 0]
    assert math.isclose(first[0].item(), 3 * 32 / SOURCE_W, rel_tol=1e-5)
    assert math.isclose(first[1].item(), 2 * 32 / SOURCE_H, rel_tol=1e-5)


def test_coco_dataset_without_keypoints_omits_the_keypoint_target(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    _image, target = CocoDetectionDataset(str(data_dir), str(annotation), image_size=32)[0]

    assert "keypoints" not in target
    assert "num_keypoints" not in target


def test_coco_dataset_finds_images_below_the_requested_folder(tmp_path):
    """A ZIP usually extracts to images/... rather than a flat folder."""
    data_dir, annotation = _write_coco_bundle(tmp_path)
    dataset = CocoDetectionDataset(str(data_dir / "images"), str(annotation), image_size=32)
    assert len(dataset) == 6


def test_coco_dataset_reports_a_usable_error_when_nothing_pairs_up(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    for entry in payload["images"]:
        entry["file_name"] = "absent.png"
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="No usable"):
        CocoDetectionDataset(str(data_dir), str(annotation))


def test_coco_dataset_rejects_a_json_that_is_not_coco(tmp_path):
    not_coco = tmp_path / "labels.json"
    not_coco.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not a COCO annotation file"):
        CocoDetectionDataset(str(tmp_path), str(not_coco))


def test_detection_collate_keeps_one_entry_per_image():
    images, targets = detection_collate([
        (torch.zeros(3, 4, 4), {"boxes": torch.zeros(1, 4)}),
        (torch.zeros(3, 4, 4), {"boxes": torch.zeros(2, 4)}),
    ])

    assert len(images) == len(targets) == 2
    assert targets[1]["boxes"].shape == (2, 4)


# ---------------------------------------------------------------------------
# Annotation discovery
# ---------------------------------------------------------------------------
def test_find_coco_annotation_walks_into_nested_folders(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    assert find_coco_annotation(str(data_dir)) == str(annotation)


def test_find_coco_annotation_ignores_other_json_files(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    (data_dir / "metadata.json").write_text('{"runs": []}', encoding="utf-8")
    (data_dir / "broken.json").write_text("{not json", encoding="utf-8")

    assert find_coco_annotation(str(data_dir)) == str(annotation)


def test_find_coco_annotation_returns_none_without_one(tmp_path):
    (tmp_path / "img.png").write_bytes(b"")
    assert find_coco_annotation(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _sample(boxes, labels=None, keypoints=None):
    out = {"boxes": torch.tensor(boxes, dtype=torch.float32)}
    if labels is not None:
        out["labels"] = torch.tensor(labels, dtype=torch.int64)
    if keypoints is not None:
        out["keypoints"] = torch.tensor(keypoints, dtype=torch.float32)
    return out


def test_score_detections_scores_a_perfect_match_as_one():
    boxes = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 40.0]]
    metrics = score_detections([_sample(boxes, [1, 2])], [_sample(boxes, [1, 2])])

    assert metrics["tp"] == 2 and metrics["fp"] == 0 and metrics["fn"] == 0
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.0
    assert metrics["pck"] is None


def test_score_detections_counts_a_wrong_class_box_as_false_positive_and_negative():
    gt = [_sample([[0.0, 0.0, 10.0, 10.0]], [1])]
    pred = [_sample([[0.0, 0.0, 10.0, 10.0]], [2])]
    metrics = score_detections(gt, pred)

    assert metrics["tp"] == 0
    assert metrics["fp"] == 1 and metrics["fn"] == 1
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.0


def test_score_detections_ignores_a_box_below_the_iou_threshold():
    gt = [_sample([[0.0, 0.0, 10.0, 10.0]], [1])]
    pred = [_sample([[0.0, 0.0, 4.0, 4.0]], [1])]
    metrics = score_detections(gt, pred)

    assert metrics["tp"] == 0 and metrics["fn"] == 1 and metrics["fp"] == 1


def test_score_detections_matches_each_ground_truth_with_one_prediction():
    gt = [_sample([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]], [1, 1])]
    pred = [_sample([[0.0, 0.0, 10.0, 10.0]], [1])]
    metrics = score_detections(gt, pred)

    assert metrics["tp"] == 1 and metrics["fn"] == 1 and metrics["fp"] == 0
    assert metrics["recall"] == 0.5 and metrics["precision"] == 1.0


def test_score_detections_handles_empty_sides():
    assert score_detections([_sample([[0, 0, 5, 5]], [1])], [_sample([])])["fn"] == 1
    assert score_detections([_sample([])], [_sample([[0, 0, 5, 5]], [1])])["fp"] == 1
    assert score_detections([_sample([])], [_sample([])])["f1"] == 0.0


def test_score_detections_pck_measures_keypoint_distance():
    box = [[0.0, 0.0, 20.0, 20.0]]
    gt_kps = [[[5.0, 5.0, 2], [10.0, 10.0, 2]]]
    metrics = score_detections(
        [_sample(box, [1], gt_kps)],
        [_sample(box, [1], [[[5.0, 5.0, 1], [10.0, 10.0, 1]]])],
    )

    # 0.1 * sqrt(20 * 20) = 2.0 pixels of slack.
    assert metrics["pck"] == 1.0


def test_score_detections_pck_penalises_a_distant_keypoint():
    box = [[0.0, 0.0, 20.0, 20.0]]
    gt_kps = [[[5.0, 5.0, 2], [10.0, 10.0, 2]]]
    metrics = score_detections(
        [_sample(box, [1], gt_kps)],
        [_sample(box, [1], [[[25.0, 25.0, 1], [10.0, 10.0, 1]]])],
    )

    assert metrics["pck"] == 0.5


def test_score_detections_pck_counts_undetected_instances_as_wrong():
    box = [[0.0, 0.0, 20.0, 20.0]]
    gt = [_sample(box, [1], [[[5.0, 5.0, 2], [10.0, 10.0, 2]]])]
    pred = [_sample([[0.0, 0.0, 20.0, 20.0], [80.0, 80.0, 90.0, 90.0]], [1, 1],
                    [[[5.0, 5.0, 1], [10.0, 10.0, 1]]])]
    metrics = score_detections(gt, pred)

    assert metrics["pck"] == 1.0
    assert metrics["fp"] == 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task_type", ["object_detection", "pose_estimation"])
def test_detection_training_records_measured_metrics(task_type, tmp_path):
    data_dir, _annotation = _write_coco_bundle(
        tmp_path, with_keypoints=task_type == "pose_estimation")
    trainer = _trainer(task_type)
    epochs = []

    model = trainer.train(
        data_dir=str(data_dir), n_epochs=2, batch_size=2, lr=0.01,
        optimizer_name="sgd", callback=lambda *args: epochs.append(args))

    assert model is trainer.best_model is not None
    assert len(trainer.history) == len(epochs) == 2
    for entry in trainer.history:
        assert entry["loss"] == entry["loss"] and entry["loss"] > 0, (
            f"training loss must be a positive number, not a placeholder: {entry}")
        assert 0.0 <= entry["val_acc"] <= 1.0
        assert 0.0 <= entry["val_loss"] <= 1e6
        assert 0.0 <= entry["f1"] <= 1.0
        assert entry["tp"] + entry["fn"] > 0, "validation must have scored real instances"
    # The epoch tuple keeps the shape the other vision loops report.
    assert len(epochs[-1]) == 6
    assert epochs[-1][0] == 1


@pytest.mark.parametrize("task_type", ["object_detection", "pose_estimation"])
def test_detection_training_reads_class_count_from_the_annotations(task_type, tmp_path):
    data_dir, _annotation = _write_coco_bundle(
        tmp_path, with_keypoints=task_type == "pose_estimation")
    trainer = CVAutoMLTrainer(task_type=task_type, num_classes=99,
                              weights=None, image_size=TRAIN_IMAGE_SIZE)
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd")

    assert trainer.num_classes == 3, "two COCO categories plus the background row"
    assert trainer.class_names == ["background", "person", "bicycle"]


def test_pose_training_sizes_the_keypoint_head_to_the_annotations(tmp_path):
    data_dir, _annotation = _write_coco_bundle(tmp_path, with_keypoints=True)
    trainer = _trainer("pose_estimation")
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd")

    assert trainer.num_keypoints == 3
    head = trainer.best_model.roi_heads.keypoint_predictor
    assert head.kps_score_lowres.out_channels == 3
    assert trainer.best_model.roi_heads.box_predictor.cls_score.out_features == 3


def test_pose_training_falls_back_to_coco_keypoint_count_without_annotations(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path, with_keypoints=True)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    for ann in payload["annotations"]:
        ann["keypoints"] = ann["keypoints"][:3]
    payload["keypoints"] = KEYPOINT_NAMES[:1]
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    trainer = _trainer("pose_estimation")
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd")

    assert trainer.num_keypoints == 1
    assert trainer.best_model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 1


def test_pose_training_reports_pck_as_the_headline_metric(tmp_path):
    data_dir, _annotation = _write_coco_bundle(tmp_path, with_keypoints=True)
    trainer = _trainer("pose_estimation")
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd")

    entry = trainer.history[-1]
    assert entry["pck"] is not None and 0.0 <= entry["pck"] <= 1.0
    assert entry["val_acc"] == entry["pck"]


def test_detection_training_ignores_geometric_augmentation_but_keeps_colour(tmp_path):
    data_dir, _annotation = _write_coco_bundle(tmp_path, with_keypoints=False)
    trainer = _trainer("object_detection")
    augment = trainer._detection_augmentation(
        {"horizontal_flip": True, "random_rotation": 15, "color_jitter": True})

    assert [type(op).__name__ for op in augment.transforms] == ["ColorJitter"]
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd",
                  augmentation_config={"horizontal_flip": True, "color_jitter": True})
    assert len(trainer.history) == 1


def test_detection_training_without_an_annotation_file_explains_the_layout(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    annotation.unlink()
    trainer = _trainer("object_detection")

    with pytest.raises(ValueError, match="COCO annotation JSON"):
        trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2)
    assert trainer.history == []


def test_detection_training_rejects_a_json_that_is_not_coco_shaped(tmp_path):
    """Discovery skips a metadata JSON, so the run stops with the layout hint."""
    data_dir, annotation = _write_coco_bundle(tmp_path)
    annotation.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    trainer = _trainer("object_detection")

    with pytest.raises(ValueError, match="COCO annotation JSON"):
        trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2)


def test_detection_training_with_a_missing_explicit_annotation_file(tmp_path):
    data_dir, _annotation = _write_coco_bundle(tmp_path)
    trainer = _trainer("pose_estimation")

    with pytest.raises(ValueError, match="COCO annotation JSON"):
        trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2,
                      annotation_file=str(tmp_path / "absent.json"))


def test_detection_training_explicit_annotation_file_is_used_over_discovery(tmp_path):
    data_dir, annotation = _write_coco_bundle(tmp_path)
    other = tmp_path / "empty.json"
    other.write_text(json.dumps({"images": [], "annotations": [], "categories": []}),
                     encoding="utf-8")
    trainer = _trainer("object_detection")
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd", annotation_file=str(annotation))

    assert trainer.class_names == ["background", "person", "bicycle"]
    assert len(trainer.history) == 1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task_type", ["object_detection", "pose_estimation"])
def test_predict_returns_boxes_scores_and_labels_in_source_pixels(task_type, tmp_path):
    data_dir, _annotation = _write_coco_bundle(
        tmp_path, with_keypoints=task_type == "pose_estimation")
    trainer = _trainer(task_type)
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd")

    sample = tmp_path / "unseen.png"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)).save(sample)
    result = trainer.predict(str(sample))

    assert set(result) >= {"boxes", "scores", "labels", "class_names"}
    assert result["boxes"].shape[1] == 4
    assert len(result["boxes"]) == len(result["scores"]) == len(result["labels"])
    assert (result["scores"] >= 0).all() and (result["scores"] <= 1).all()
    # Coordinates come back for a 40x30 image, not for the square the model saw.
    assert (result["boxes"][:, 0] >= 0).all() and (result["boxes"][:, 2] <= 40.01).all()
    assert (result["boxes"][:, 1] >= 0).all() and (result["boxes"][:, 3] <= 30.01).all()
    if task_type == "pose_estimation":
        assert result["keypoints"].shape[1:] == (trainer.num_keypoints, 3)


def test_predict_of_detection_result_is_repeatable(tmp_path):
    data_dir, _annotation = _write_coco_bundle(tmp_path, with_keypoints=False)
    trainer = _trainer("object_detection")
    trainer.train(data_dir=str(data_dir), n_epochs=1, batch_size=2, lr=0.01,
                  optimizer_name="sgd")
    sample = tmp_path / "again.png"
    Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8)).save(sample)

    first, second = trainer.predict(str(sample)), trainer.predict(str(sample))
    assert np.array_equal(first["boxes"], second["boxes"])
    assert np.array_equal(first["labels"], second["labels"])


# ---------------------------------------------------------------------------
# Overlay rendering (both Vision Studio and the registry page draw through this)
# ---------------------------------------------------------------------------
def _blank(width=64, height=48):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def test_draw_detections_keeps_only_boxes_above_the_threshold():
    result = {"boxes": np.array([[4.0, 4.0, 20.0, 20.0], [30.0, 30.0, 44.0, 44.0]]),
              "scores": np.array([0.9, 0.2]),
              "labels": np.array([1, 2]),
              "class_names": ["background", "person", "bicycle"]}

    overlay, kept = draw_detections(_blank(), result)
    pixels = np.asarray(overlay)

    assert kept == 1
    assert overlay.size == (64, 48)
    assert pixels[4:21, 4:21].max() > 0, "the surviving box must be visible"
    assert pixels[31:44, 31:44].max() == 0, "a box under the threshold must not be drawn"


def test_draw_detections_marks_visible_keypoints():
    result = {"boxes": np.array([[4.0, 4.0, 20.0, 20.0]]),
              "scores": np.array([0.9]),
              "labels": np.array([1]),
              "keypoints": np.array([[[10.0, 10.0, 2.0], [0.0, 0.0, 0.0]]])}

    overlay, kept = draw_detections(_blank(), result)

    assert kept == 1
    assert np.asarray(overlay)[9:12, 9:12].max() > 0, "the visible keypoint never landed"


def test_draw_detections_accepts_a_raw_torchvision_result():
    """The registry page passes the model's own dict, which has no class_names."""
    result = {"boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0]]),
              "scores": torch.tensor([0.8]),
              "labels": torch.tensor([1])}

    overlay, kept = draw_detections(_blank(), result)

    assert kept == 1
    assert np.asarray(overlay).max() > 0


def test_draw_detections_leaves_the_image_untouched_when_nothing_clears():
    image = _blank()
    result = {"boxes": np.array([[1.0, 1.0, 10.0, 10.0]]), "scores": np.array([0.1]),
              "labels": np.array([1])}

    overlay, kept = draw_detections(image, result)

    assert kept == 0
    assert np.array_equal(np.asarray(overlay), np.asarray(image))


def test_score_detections_counts_only_proposals_above_the_score_threshold():
    """The overlay hides what is below 0.50, so the metrics have to hide it too."""
    gt = [_sample([[0.0, 0.0, 10.0, 10.0]], [1])]
    pred = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0],
                                    [80.0, 80.0, 90.0, 90.0]]),
             "labels": torch.tensor([1, 1, 1]),
             "scores": torch.tensor([0.9, 0.3, 0.05])}]

    filtered = score_detections(gt, pred, score_threshold=0.5)
    assert filtered["tp"] == 1 and filtered["fp"] == 0 and filtered["fn"] == 0
    assert filtered["precision"] == filtered["recall"] == 1.0

    unfiltered = score_detections(gt, pred)
    assert unfiltered["fp"] == 2, "every emitted detection counts by default"


def test_score_detections_filters_keypoints_along_with_their_boxes():
    """A filtered box must take its keypoints with it, or the surviving instance is scored
    against the keypoints of the detection that was dropped."""
    gt = [_sample([[40.0, 40.0, 60.0, 60.0]], [1], [[[45.0, 45.0, 2.0]]])]
    pred = [{"boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0], [40.0, 40.0, 60.0, 60.0]]),
             "labels": torch.tensor([1, 1]),
             "scores": torch.tensor([0.2, 0.6]),
             "keypoints": torch.tensor([[[5.0, 5.0, 2.0]], [[45.0, 45.0, 2.0]]])}]

    metrics = score_detections(gt, pred, score_threshold=0.5)

    assert metrics["tp"] == 1 and metrics["fp"] == 0
    assert metrics["pck"] == 1.0
