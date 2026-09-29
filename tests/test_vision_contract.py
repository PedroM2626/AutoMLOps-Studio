"""Contract tests for the computer-vision engine (src/engines/vision.py).

Scope is the surface app.py drives: the Vision Studio wizard (CVAutoMLTrainer), the
"Architect Insight" text (get_cv_explanation) and the dataset/transform helpers.
Nothing here builds a torchvision backbone, because that downloads pretrained weights.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

import src.engines.vision as vision
from src.engines.vision import BACKBONE_REGISTRY, CVAutoMLTrainer, get_cv_explanation

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAINABLE_TASK_TYPES = {
    "image_classification",
    "image_multi_label",
    "image_segmentation",
    "image_anomaly_detection",
}
UNIMPLEMENTED_TASK_TYPES = ("object_detection", "pose_estimation")

RANDOM_TRANSFORMS = {
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "RandomResizedCrop",
}

CONFIG_FROM_UI = {"lr": 0.0035, "batch_size": 16}


def _app_literal(name):
    """Read a list literal out of app.py without importing the Streamlit script."""
    tree = ast.parse((REPO_ROOT / "app.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    found.append(ast.literal_eval(node.value))
    assert found, f"{name} is no longer defined in app.py - the UI/engine sync guard is stale"
    return found


def _transform_names(compose_obj):
    return [type(op).__name__ for op in compose_obj.transforms]


def _write_image(path, rng, size=16):
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).convert("RGB").save(path)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(11)


@pytest.fixture(scope="module")
def image_folder(tmp_path_factory, rng):
    root = tmp_path_factory.mktemp("cv_classification")
    for class_name in ("alpha", "beta"):
        class_dir = root / class_name
        class_dir.mkdir()
        for i in range(8):
            _write_image(class_dir / f"{i}.png", rng)
    return root


@pytest.fixture(scope="module")
def multi_label_bundle(tmp_path_factory, rng):
    images = tmp_path_factory.mktemp("cv_multilabel")
    rows = []
    for i in range(8):
        filename = f"img_{i}.png"
        _write_image(images / filename, rng)
        rows.append({"filename": filename, "tag_a": i % 2, "tag_b": int(i >= 4), "tag_c": 1})
    csv_path = images / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return images, csv_path


@pytest.fixture(scope="module")
def segmentation_dirs(tmp_path_factory, rng):
    images = tmp_path_factory.mktemp("seg_images")
    masks = tmp_path_factory.mktemp("seg_masks")
    for i in range(3):
        _write_image(images / f"{i}.png", rng, size=8)
        arr = np.full((8, 8), 255, dtype=np.uint8)
        Image.fromarray(arr).convert("L").save(masks / f"{i}.png")
    return images, masks


# ---------------------------------------------------------------------------
# The unsupported-task guard (the point of this file)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task_type", UNIMPLEMENTED_TASK_TYPES)
def test_unsupported_task_train_raises_not_implemented(task_type, tmp_path):
    trainer = CVAutoMLTrainer(task_type=task_type, num_classes=3, backbone="resnet18")
    with pytest.raises(NotImplementedError):
        trainer.train(data_dir=str(tmp_path), n_epochs=2, batch_size=2, lr=0.01)


@pytest.mark.parametrize("task_type", UNIMPLEMENTED_TASK_TYPES)
def test_unsupported_task_message_is_honest(task_type):
    trainer = CVAutoMLTrainer(task_type=task_type)
    with pytest.raises(NotImplementedError) as excinfo:
        trainer.train(data_dir="unused", n_epochs=1)

    message = str(excinfo.value)
    assert task_type in message, f"message must name the refused task: {message}"
    assert "not implemented" in message.lower(), f"message must say it is unsupported: {message}"
    named_alternatives = [t for t in TRAINABLE_TASK_TYPES if t in message]
    assert len(named_alternatives) >= 2, f"message must point at usable tasks: {message}"


@pytest.mark.parametrize("task_type", UNIMPLEMENTED_TASK_TYPES)
def test_unsupported_task_produces_no_fake_history_or_model(task_type):
    """The guard exists because the old code looped over epochs writing zeroed metrics."""
    epochs_reported = []
    trainer = CVAutoMLTrainer(task_type=task_type)
    with pytest.raises(NotImplementedError):
        trainer.train(
            data_dir="unused",
            n_epochs=3,
            callback=lambda *args: epochs_reported.append(args),
        )

    assert trainer.history == [], f"no epoch may be recorded: {trainer.history}"
    assert trainer.best_model is None, "an untrained model must not be handed back as champion"
    assert epochs_reported == [], "the epoch callback must never fire for an unsupported task"
    assert trainer.get_per_class_metrics() == {}


@pytest.mark.parametrize(
    "task_type,builder",
    [("object_detection", "fasterrcnn_resnet50_fpn"), ("pose_estimation", "keypointrcnn_resnet50_fpn")],
)
def test_unsupported_task_guard_fires_before_any_weight_download(
    task_type, builder, monkeypatch
):
    attempts = []

    def _boom(*args, **kwargs):
        attempts.append(builder)
        raise AssertionError(f"{builder} must not be constructed for {task_type}")

    monkeypatch.setattr(vision, builder, _boom)
    trainer = CVAutoMLTrainer(task_type=task_type)
    with pytest.raises(NotImplementedError):
        trainer.train(data_dir="unused", n_epochs=1)
    assert attempts == []


def test_orchestrator_propagates_the_guard_without_recording_metrics():
    """src/core/orchestrator.py:60 forwards train() to headless/API callers."""
    from src.core.orchestrator import AutoMLOrchestrator

    orchestrator = AutoMLOrchestrator(
        {"task_type": "object_detection", "selected_backbone": "resnet18", "dataset_path": None}
    )
    with pytest.raises(NotImplementedError):
        orchestrator.run_vision_training()


# ---------------------------------------------------------------------------
# UI / engine sync
# ---------------------------------------------------------------------------
def test_ui_offers_only_trainable_vision_tasks():
    ui_tasks = {
        entry[0] for literal in _app_literal("CV_TASKS") for entry in literal
    }
    assert ui_tasks, "app.py must still declare the Vision Studio task cards"
    forbidden = ui_tasks & set(UNIMPLEMENTED_TASK_TYPES)
    assert not forbidden, f"app.py would offer tasks that train() refuses: {sorted(forbidden)}"
    assert ui_tasks <= TRAINABLE_TASK_TYPES


def test_backbone_registry_matches_the_backbones_the_ui_offers():
    ui_backbones = {name for literal in _app_literal("backbones") for name in literal}
    assert ui_backbones == set(BACKBONE_REGISTRY), (
        "Vision Studio backbone list and BACKBONE_REGISTRY drifted apart"
    )


def test_backbone_registry_values_are_class_builders():
    assert BACKBONE_REGISTRY, "BACKBONE_REGISTRY must not be emptied"
    for name, builder in BACKBONE_REGISTRY.items():
        assert isinstance(name, str) and name
        assert callable(builder), f"{name} is not a callable builder"


def test_get_model_for_detection_tasks_requires_pretrained_weights():
    """train() refuses these tasks while get_model() can still build them, so the engine
    is only half-guarded.  Building either detection backbone needs a weights download,
    so the gap is documented here rather than exercised."""
    pytest.skip("get_model('object_detection'/'pose_estimation') downloads pretrained weights")


# ---------------------------------------------------------------------------
# get_cv_explanation, the "Architect Insight" string used by app.py:3695
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone", sorted(BACKBONE_REGISTRY))
def test_get_cv_explanation_returns_non_empty_text_for_every_ui_backbone(backbone):
    insight = get_cv_explanation(backbone, CONFIG_FROM_UI)
    assert isinstance(insight, str)
    assert insight.strip(), f"{backbone} produced an empty insight"
    assert len(insight.split()) >= 5, f"{backbone} insight is too thin to help a user"


def test_get_cv_explanation_is_specific_per_backbone():
    texts = [get_cv_explanation(name, CONFIG_FROM_UI) for name in sorted(BACKBONE_REGISTRY)]
    assert len(set(texts)) == len(texts), "two backbones share the same insight text"


@pytest.mark.parametrize(
    "key",
    ["deeplabv3", "faster_rcnn", "pose_estimation", "image_anomaly_detection"],
)
def test_get_cv_explanation_covers_task_and_segmentation_entries(key):
    assert get_cv_explanation(key, CONFIG_FROM_UI).strip()


def test_get_cv_explanation_interpolates_the_ui_config():
    lr_text = get_cv_explanation("lr", CONFIG_FROM_UI)
    batch_text = get_cv_explanation("batch_size", CONFIG_FROM_UI)
    assert str(CONFIG_FROM_UI["lr"]) in lr_text
    assert str(CONFIG_FROM_UI["batch_size"]) in batch_text


def test_get_cv_explanation_survives_missing_config_keys():
    """app.py fills the config from session state and may pass 'N/A' placeholders."""
    for params in ({}, {"lr": "N/A", "batch_size": "N/A"}):
        for key in ("lr", "batch_size"):
            text = get_cv_explanation(key, params)
            assert text.strip(), f"{key} with {params} produced no explanation"


def test_get_cv_explanation_falls_back_for_unknown_backbone():
    assert get_cv_explanation("not_a_real_backbone", CONFIG_FROM_UI).strip()


# ---------------------------------------------------------------------------
# Graceful input validation (no training involved)
# ---------------------------------------------------------------------------
def test_segmentation_without_mask_dir_returns_none(tmp_path):
    trainer = CVAutoMLTrainer(task_type="image_segmentation")
    assert trainer.train(data_dir=str(tmp_path), n_epochs=1) is None
    assert trainer.history == []


def test_multi_label_without_label_csv_returns_none(tmp_path):
    trainer = CVAutoMLTrainer(task_type="image_multi_label")
    assert trainer.train(data_dir=str(tmp_path), n_epochs=1, label_csv=None) is None
    assert trainer.history == []


def test_multi_label_with_missing_label_csv_returns_none(tmp_path):
    trainer = CVAutoMLTrainer(task_type="image_multi_label")
    missing = tmp_path / "absent.csv"
    assert trainer.train(data_dir=str(tmp_path), n_epochs=1, label_csv=str(missing)) is None


def test_classification_on_unreadable_folder_returns_none(tmp_path):
    trainer = CVAutoMLTrainer(task_type="image_classification")
    assert trainer.train(data_dir=str(tmp_path / "nope"), n_epochs=1) is None
    assert trainer.history == []
    assert trainer.best_model is None


def test_predict_before_training_returns_none(tmp_path, rng):
    trainer = CVAutoMLTrainer(task_type="image_classification")
    image_path = tmp_path / "sample.png"
    _write_image(image_path, rng)
    assert trainer.predict(str(image_path)) is None


# ---------------------------------------------------------------------------
# Transform assembly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "aug_config,expected",
    [
        ({"horizontal_flip": True}, "RandomHorizontalFlip"),
        ({"vertical_flip": True}, "RandomVerticalFlip"),
        ({"random_rotation": 15}, "RandomRotation"),
        ({"color_jitter": True}, "ColorJitter"),
        ({"random_crop": True}, "RandomResizedCrop"),
    ],
)
def test_augmentation_config_adds_the_matching_op(aug_config, expected):
    trainer = CVAutoMLTrainer()
    names = _transform_names(trainer._build_transforms(aug_config, train=True))
    assert expected in names


@pytest.mark.parametrize("aug_key", ["horizontal_flip", "vertical_flip", "random_crop", "color_jitter"])
def test_augmentation_config_off_by_default(aug_key):
    trainer = CVAutoMLTrainer()
    names = _transform_names(trainer._build_transforms({aug_key: False}, train=True))
    assert not (set(names) & RANDOM_TRANSFORMS), f"{aug_key}=False still injected a random op"
    names = _transform_names(trainer._build_transforms({"random_rotation": 0}, train=True))
    assert "RandomRotation" not in names


def test_training_transforms_normalise_after_tensor_conversion():
    trainer = CVAutoMLTrainer()
    names = _transform_names(trainer._build_transforms({"horizontal_flip": True}, train=True))
    assert names[0] == "Resize"
    assert names[-1] == "Normalize"
    assert names[-2] == "ToTensor"
    assert set(names[1:-2]) <= RANDOM_TRANSFORMS


def test_eval_transforms_are_deterministic():
    trainer = CVAutoMLTrainer()
    names = _transform_names(
        trainer._build_transforms({"horizontal_flip": True, "color_jitter": True}, train=False)
    )
    assert not (set(names) & RANDOM_TRANSFORMS), f"eval pipeline kept a random op: {names}"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def test_multi_label_dataset_reads_filename_and_label_columns(multi_label_bundle):
    images, csv_path = multi_label_bundle
    dataset = vision.MultiLabelImageDataset(str(images), str(csv_path))
    assert len(dataset) == 8
    assert dataset.label_names == ["tag_a", "tag_b", "tag_c"]
    assert dataset.labels.shape == (8, 3)
    assert dataset.labels.dtype == np.float32
    assert dataset.filenames[0].endswith(".png")


def test_multi_label_dataset_item_is_tensor_and_label_vector(multi_label_bundle):
    images, csv_path = multi_label_bundle
    trainer = CVAutoMLTrainer(task_type="image_multi_label")
    dataset = vision.MultiLabelImageDataset(
        str(images), str(csv_path), transform=trainer._build_transforms(train=False)
    )
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label.shape == (3,)
    assert label.dtype == torch.float32


def test_segmentation_dataset_pairs_by_filename(segmentation_dirs):
    images, masks = segmentation_dirs
    dataset = vision.SegmentationDataset(str(images), str(masks))
    assert len(dataset) == 3
    assert dataset.pairs == sorted(dataset.pairs)


def test_segmentation_dataset_drops_unpaired_files_instead_of_misaligning_them(tmp_path):
    """An image with no mask of the same name must not borrow a neighbour's mask."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    local_rng = np.random.default_rng(2)
    for i in range(3):
        _write_image(images_dir / f"{i}.png", local_rng, size=8)
        _write_image(masks_dir / f"{i}.png", local_rng, size=8)
    _write_image(images_dir / "orphan.png", local_rng, size=8)

    dataset = vision.SegmentationDataset(str(images_dir), str(masks_dir))

    assert len(dataset) == 3
    assert "orphan.png" not in dataset.pairs
    for index in range(len(dataset)):
        image, mask = dataset[index]
        assert image.size == (8, 8)
        assert mask.size == (8, 8)


def test_get_per_class_metrics_is_a_documented_placeholder():
    assert CVAutoMLTrainer().get_per_class_metrics() == {}


# ---------------------------------------------------------------------------
# Known product bugs: the validation split shares one transform object
# ---------------------------------------------------------------------------
def _capture_loaders(store):
    def _loop(self, model, train_loader, val_loader, criterion, optimizer, n_epochs, callback):
        store["train_transform"] = train_loader.dataset.dataset.transform
        store["val_transform"] = val_loader.dataset.dataset.transform
        return model

    return _loop


def test_classification_validation_subset_uses_eval_only_transforms(
    monkeypatch, image_folder
):
    store = {}
    monkeypatch.setattr(CVAutoMLTrainer, "get_model", lambda self: nn.Linear(3, 2))
    monkeypatch.setattr(CVAutoMLTrainer, "_run_classification_loop", _capture_loaders(store))

    trainer = CVAutoMLTrainer(task_type="image_classification", num_classes=2)
    trainer.train(
        data_dir=str(image_folder),
        n_epochs=1,
        batch_size=2,
        augmentation_config={"horizontal_flip": True, "color_jitter": True},
    )

    assert store, "train() never reached the classification loop"
    assert not (set(_transform_names(store["val_transform"])) & RANDOM_TRANSFORMS), (
        "validation images are augmented, so val_acc/val_loss are not comparable "
        "between epochs"
    )


def test_multi_label_augmentation_survives_the_validation_split(monkeypatch, multi_label_bundle):
    images, csv_path = multi_label_bundle
    store = {}
    monkeypatch.setattr(CVAutoMLTrainer, "get_model", lambda self: nn.Linear(3, 2))
    monkeypatch.setattr(CVAutoMLTrainer, "_run_multilabel_loop", _capture_loaders(store))

    trainer = CVAutoMLTrainer(task_type="image_multi_label")
    trainer.train(
        data_dir=str(images),
        n_epochs=1,
        batch_size=2,
        augmentation_config={"horizontal_flip": True},
        label_csv=str(csv_path),
    )

    assert store, "train() never reached the multi-label loop"
    assert "RandomHorizontalFlip" in _transform_names(store["train_transform"]), (
        "the val-transform assignment leaked into the training subset"
    )
