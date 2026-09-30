import os
import json
import time
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

logger = logging.getLogger(__name__)


def split_indices(total, val_split, seed=42):
    val_size = max(1, int(total * val_split))
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    return permutation[val_size:].tolist(), permutation[:val_size].tolist()


def _split_with_transforms(make_dataset, total, val_split, train_tf, val_tf, seed=42):
    """Split indices once, then build one dataset instance per side.

    random_split hands both subsets the SAME dataset object, so giving the validation
    subset a different transform also replaces the training view: augmentation was silently
    dropped from training, and where no val transform was applied at all, validation scored
    randomly augmented images and stopped being reproducible.
    """
    train_idx, val_idx = split_indices(total, val_split, seed)
    return (Subset(make_dataset(train_tf), train_idx),
            Subset(make_dataset(val_tf), val_idx))

# ---------------------------------------------------------------------------
# Supported backbones for classification / multi-label
# ---------------------------------------------------------------------------
BACKBONE_REGISTRY = {
    'resnet18':       lambda nc: _resnet18_head(nc),
    'resnet50':       lambda nc: _resnet50_head(nc),
    'mobilenet_v2':   lambda nc: _mobilenet_v2_head(nc),
    'efficientnet_b0': lambda nc: _efficientnet_b0_head(nc),
    'densenet121':    lambda nc: _densenet121_head(nc),
    'vgg16':          lambda nc: _vgg16_head(nc),
}

def _resnet18_head(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _resnet50_head(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _mobilenet_v2_head(num_classes):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def _efficientnet_b0_head(num_classes):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def _densenet121_head(num_classes):
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def _vgg16_head(num_classes):
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m


# ---------------------------------------------------------------------------
# Multi-label dataset: expects a CSV with columns [filename, label1, label2, ...]
# where each label column is 0 or 1.
# ---------------------------------------------------------------------------
class MultiLabelImageDataset(Dataset):
    """
    Multi-label classification dataset.

    CSV format (no header or with header):
        image_filename.jpg, 1, 0, 1, ...

    If a header row is detected, skip it.
    Images are loaded from `image_dir`.
    """
    def __init__(self, image_dir, label_csv_path, transform=None):
        import pandas as pd
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(label_csv_path)
        # First column is filename, rest are label columns
        self.filenames = df.iloc[:, 0].tolist()
        self.labels = df.iloc[:, 1:].values.astype(np.float32)
        self.label_names = list(df.columns[1:])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, str(self.filenames[idx]))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ---------------------------------------------------------------------------
# Segmentation dataset (unchanged from original)
# ---------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    """Custom Dataset for Image Segmentation."""
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        # Pair by filename: two independently sorted listings drift as soon as one folder
        # has an extra file, and the training then scores images against the wrong mask.
        images = {name for name in os.listdir(image_dir)
                  if os.path.isfile(os.path.join(image_dir, name))}
        masks = {name for name in os.listdir(mask_dir)
                 if os.path.isfile(os.path.join(mask_dir, name))}
        self.pairs = sorted(images & masks)
        skipped = sorted((images | masks) - set(self.pairs))
        if skipped:
            logger.warning(
                f"Segmentation skipped {len(skipped)} file(s) without a counterpart in the "
                f"other folder, e.g. {skipped[:3]}"
            )
        if not self.pairs:
            raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name = self.pairs[idx]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask * 255).long().squeeze(0)

        return image, mask


def _color_jitter():
    return transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)


def detection_transform(image_size):
    """The exact preprocessing detection and pose models are trained and scored with."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def find_coco_annotation(root):
    """First JSON under `root` that is shaped like a COCO annotation file.

    Datasets arrive as a ZIP whose layout nobody enforces: the accepted ones keep the JSON
    next to the images, one folder down, or inside the archive's own top-level folder, so the
    search is recursive rather than hardcoded to `annotations/instances.json`.
    """
    candidates = []
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith('.json'):
                candidates.append(os.path.join(dirpath, name))
    for path in sorted(candidates):
        try:
            with open(path, encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and 'images' in payload and 'annotations' in payload:
            return path
    return None


class CocoDetectionDataset(Dataset):
    """Images plus a COCO-style annotation file, used by detection and pose training.

    COCO stores boxes as [x, y, width, height] in the original pixel space; torchvision
    wants [x1, y1, x2, y2] in the space the model actually sees. The rescale therefore
    happens here, where the source size is still known - letting a standalone transform
    resize the image would leave boxes and keypoints pointing at the old geometry.
    """

    _IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, image_dir, annotation_file, image_size=224, require_keypoints=False,
                 augment=None):
        self.image_dir = image_dir
        self.image_size = int(image_size)
        self.require_keypoints = require_keypoints
        self.augment = augment
        self.normalize = detection_transform(self.image_size)

        with open(annotation_file, encoding="utf-8") as handle:
            coco = json.load(handle)
        if "images" not in coco or "annotations" not in coco:
            raise ValueError(
                f"{annotation_file} is not a COCO annotation file: it needs 'images' and "
                "'annotations' lists."
            )

        self.categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
        # torchvision reserves label 0 for the background, so COCO ids shift by one.
        self.label_by_id = {int(c["id"]): i + 1 for i, c in enumerate(self.categories)}
        self.class_names = ["background"] + [str(c.get("name", c["id"])) for c in self.categories]

        located = self._locate_images(coco["images"])
        by_image = {}
        for ann in coco["annotations"]:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)

        self.records = []
        self.num_keypoints = 0
        missing = []
        for entry in coco["images"]:
            image_id = int(entry["id"])
            path = located.get(image_id)
            if path is None:
                missing.append(entry.get("file_name"))
                continue
            annotations = by_image.get(image_id, [])
            if self.require_keypoints:
                annotations = [a for a in annotations if a.get("keypoints")]
            if not annotations:
                continue
            if self.require_keypoints:
                self.num_keypoints = max(
                    self.num_keypoints, max(len(a["keypoints"]) // 3 for a in annotations)
                )
            self.records.append({
                "path": path,
                "width": float(entry.get("width") or 0),
                "height": float(entry.get("height") or 0),
                "annotations": annotations,
                "image_id": image_id,
            })

        if missing:
            logger.warning(
                f"COCO skipped {len(missing)} annotated image(s) with no file on disk: {missing[:3]}"
            )
        if not self.records:
            raise ValueError(
                f"No usable {'keypoint' if self.require_keypoints else 'bounding box'} "
                f"annotations found for the images in {image_dir}."
            )

    def _locate_images(self, images):
        """Map COCO image ids to files, tolerating a folder prefix inside the archive."""
        index = {}
        for root, _dirs, files in os.walk(self.image_dir):
            for name in files:
                if name.lower().endswith(self._IMAGE_SUFFIXES):
                    index.setdefault(name, os.path.join(root, name))

        located = {}
        for entry in images:
            file_name = str(entry.get("file_name") or f"{entry['id']}")
            direct = file_name if os.path.isfile(file_name) else os.path.join(self.image_dir, file_name)
            if os.path.isfile(direct):
                located[int(entry["id"])] = direct
                continue
            base = os.path.basename(file_name)
            if base in index:
                located[int(entry["id"])] = index[base]
        return located

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        orig_w, orig_h = image.size
        width = record["width"] or orig_w
        height = record["height"] or orig_h
        # Only photometric ops belong here: a flip or a crop would move the pixels while the
        # box coordinates kept pointing at the original geometry.
        if self.augment:
            image = self.augment(image)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        scale_x = self.image_size / float(width or self.image_size)
        scale_y = self.image_size / float(height or self.image_size)

        boxes, labels, areas, keypoints = [], [], [], []
        for ann in record["annotations"]:
            x, y, w, h = [float(v) for v in ann["bbox"][:4]]
            x1, y1 = x * scale_x, y * scale_y
            x2, y2 = (x + w) * scale_x, (y + h) * scale_y
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.label_by_id.get(int(ann["category_id"]), 1))
            areas.append(max((x2 - x1) * (y2 - y1), 1.0))
            if self.require_keypoints:
                flat = ann.get("keypoints") or []
                points = []
                for i in range(0, len(flat) - 2, 3):
                    points.append([flat[i] * scale_x, flat[i + 1] * scale_y, float(flat[i + 2])])
                while len(points) < self.num_keypoints:
                    points.append([0.0, 0.0, 0.0])
                keypoints.append(points[:self.num_keypoints])

        target = {
            "image_id": torch.tensor([record["image_id"]], dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
        }
        if self.require_keypoints:
            target["num_keypoints"] = torch.tensor(
                [sum(1 for kp in kps if kp[2] > 0) for kps in keypoints], dtype=torch.int64
            )
            target["keypoints"] = torch.tensor(keypoints, dtype=torch.float32)

        return self.normalize(image), target


def detection_collate(batch):
    """Torchvision detection models take a list of images plus a list of targets."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def _keep_above_score(prediction, score_threshold):
    """Drop the proposals the UI will not draw, so the counted boxes are the shown boxes."""
    scores = prediction.get("scores")
    if scores is None or score_threshold is None:
        return prediction
    keep = [i for i in range(scores.shape[0]) if float(scores[i]) >= score_threshold]
    return {key: (value[keep] if torch.is_tensor(value) and value.dim() and
                  value.shape[0] == scores.shape[0] else value)
            for key, value in prediction.items()}


def score_detections(ground_truths, predictions, iou_threshold=0.5, score_threshold=None):
    """Greedy instance matching plus PCK, so the reported numbers are measured.

    A single scalar mAP would need pycocotools and 101 recall interpolation points; for a
    training curve, precision/recall/F1 at one IoU threshold and the percentage of correct
    keypoints tell the same story without a hidden dependency. Every detection the model
    emits counts by default - precision is then the pressure on its own noise floor, which
    is what improves epoch over epoch. `score_threshold` narrows it to an operating point.
    """
    from torchvision.ops import box_iou

    predictions = [_keep_above_score(p, score_threshold) if p is not None else p
                   for p in predictions]

    tp = fp = fn = 0
    kp_correct = kp_total = 0

    for target, prediction in zip(ground_truths, predictions):
        gt_boxes = target.get("boxes", torch.zeros((0, 4)))
        pred_boxes = prediction.get("boxes", torch.zeros((0, 4))) if prediction is not None             else torch.zeros((0, 4))
        matched_pred, matched_gt, matched_instance = set(), set(), {}

        if gt_boxes.numel() and pred_boxes.numel():
            iou = box_iou(pred_boxes.float(), gt_boxes.float())
            gt_labels = target.get("labels")
            pred_labels = prediction.get("labels")
            for gt_index in range(gt_boxes.shape[0]):
                best, best_value = None, iou_threshold
                for pred_index in range(pred_boxes.shape[0]):
                    if pred_index in matched_pred:
                        continue
                    # A box in the right place around the wrong class is a false positive, so
                    # a candidate only competes when it carries the ground-truth label.
                    if gt_labels is not None and pred_labels is not None:
                        if int(pred_labels[pred_index]) != int(gt_labels[gt_index]):
                            continue
                    value = float(iou[pred_index, gt_index])
                    if value >= best_value:
                        best, best_value = pred_index, value
                if best is not None:
                    matched_pred.add(best)
                    matched_gt.add(gt_index)
                    matched_instance[gt_index] = best
                    tp += 1

        fp += int(pred_boxes.shape[0]) - len(matched_pred)
        fn += int(gt_boxes.shape[0]) - len(matched_gt)

        gt_kps = target.get("keypoints")
        pred_kps = prediction.get("keypoints") if prediction is not None else None
        if gt_kps is not None and gt_boxes.shape[0]:
            # Annotated keypoints with no matching detection are wrong keypoints, not missing
            # ones: skipping them would report PCK over the instances that were detected.
            usable = pred_kps is not None and pred_kps.numel()
            for gt_index in range(gt_kps.shape[0]):
                visible = [k for k in range(gt_kps.shape[1]) if float(gt_kps[gt_index, k, 2]) > 0]
                if not visible:
                    continue
                # Counted against the instance this box matched: the model orders its output
                # by score while COCO keeps annotation order, so positions do not line up.
                pred_index = matched_instance.get(gt_index) if usable else None
                if pred_index is None or pred_index >= pred_kps.shape[0]:
                    kp_total += len(visible)
                    continue
                box = gt_boxes[gt_index]
                scale = float((box[2] - box[0]) * (box[3] - box[1]))
                threshold = 0.1 * (scale ** 0.5) if scale > 0 else 1.0
                for k in visible:
                    kp_total += 1
                    distance = float((pred_kps[pred_index, k, :2]
                                      - gt_kps[gt_index, k, :2]).norm())
                    if distance <= threshold:
                        kp_correct += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pck": (kp_correct / kp_total) if kp_total else None,
    }


def draw_detections(image, result, score_threshold=0.5):
    """Return a copy of `image` with the boxes (and keypoints) above the threshold drawn on it.

    Accepts either CVAutoMLTrainer.predict() output or a raw torchvision result dict, so the
    Vision Studio page and the registry page render detections the same way. Coordinates are
    expected to already be in the given image's pixel space.
    """
    from PIL import ImageDraw

    canvas = image.copy().convert('RGB')
    drawer = ImageDraw.Draw(canvas)
    boxes = np.asarray(result.get('boxes', []), dtype=float)
    scores = np.asarray(result.get('scores', []), dtype=float)
    labels = np.asarray(result.get('labels', []), dtype=int)
    names = result.get('class_names') or []
    keypoints = result.get('keypoints')

    kept = [i for i in range(len(scores)) if scores[i] >= score_threshold]
    for i in kept:
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        index = int(labels[i]) if i < len(labels) else -1
        name = names[index] if 0 <= index < len(names) else str(index)
        drawer.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=2)
        drawer.text((x1 + 2, max(0.0, y1 - 12)), f"{name} {scores[i]:.2f}", fill=(34, 197, 94))
        if keypoints is None or i >= len(keypoints):
            continue
        for point in np.asarray(keypoints[i], dtype=float):
            if len(point) > 2 and point[2] > 0:
                x, y = float(point[0]), float(point[1])
                drawer.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(239, 68, 68))

    return canvas, len(kept)


# ---------------------------------------------------------------------------
# Main CV Trainer
# ---------------------------------------------------------------------------
class CVAutoMLTrainer:
    def __init__(self, task_type='image_classification', num_classes=2,
                 backbone='resnet18', multilabel_threshold=0.5,
                 weights='DEFAULT', image_size=224):
        """
        Parameters
        ----------
        task_type : str
            One of: 'image_classification', 'image_multi_label',
                'image_segmentation', 'object_detection',
                'image_anomaly_detection', 'pose_estimation'
        num_classes : int
            Number of output classes / labels. For object_detection and pose_estimation
            train() overwrites it with the COCO category count plus the background row the
            Fast R-CNN head is built with, so the labels in the JSON always win.
        backbone : str
            Backbone key (see BACKBONE_REGISTRY).
        multilabel_threshold : float
            Sigmoid threshold for multi-label positive prediction.
        weights : str or None
            Torchvision weights enum name for the detection/segmentation models. None builds
            the same architecture from scratch, which is what the tests use to stay offline.
        image_size : int
            Side length detection and pose training resizes to.
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.backbone = backbone
        self.multilabel_threshold = multilabel_threshold
        self.weights = weights
        self.image_size = int(image_size)
        self.num_keypoints = 0
        self.class_names = []
        self.label_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model = None
        self.history = []  # list of epoch dicts

    def _load_weights(self, enum_class):
        """Resolve the weights argument to what torchvision expects, or None."""
        if self.weights is None or self.weights in ('none', 'None'):
            return None
        return enum_class.DEFAULT

    # ------------------------------------------------------------------
    def get_model(self):
        """Build the model head for the selected task and backbone."""
        if self.task_type == 'image_segmentation':
            model = deeplabv3_resnet50(weights=self._load_weights(DeepLabV3_ResNet50_Weights))
            in_channels = model.classifier[4].in_channels
            model.classifier[4] = nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
            if model.aux_classifier:
                aux_in = model.aux_classifier[4].in_channels
                model.aux_classifier[4] = nn.Conv2d(aux_in, self.num_classes, kernel_size=1)
            return model.to(self.device)

        elif self.task_type == 'object_detection':
            model = fasterrcnn_resnet50_fpn(weights=self._load_weights(FasterRCNN_ResNet50_FPN_Weights))
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # num_classes must include the background row the COCO head was trained with.
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model.to(self.device)

        elif self.task_type == 'pose_estimation':
            model = keypointrcnn_resnet50_fpn(weights=self._load_weights(KeypointRCNN_ResNet50_FPN_Weights))
            # Leaving the COCO heads in place trained 20 classes and 17 keypoints against a
            # dataset that has neither, so both heads are replaced to match the annotations.
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
                model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels,
                max(self.num_keypoints, 1),
            )
            return model.to(self.device)

        else:
            # Classification or Multi-label
            builder = BACKBONE_REGISTRY.get(self.backbone, BACKBONE_REGISTRY['resnet18'])
            model = builder(self.num_classes)
            return model.to(self.device)

    # ------------------------------------------------------------------
    def _safe_augmentation(self, augmentation_config):
        """Drop geometric augmentations for pixel- and box-level tasks.

        The mask is transformed separately with a resize-only pipeline, so flipping or
        rotating the image without the same op on the mask would train the model against
        shifted labels. Detection and pose carry the same hazard in a different shape: the
        boxes live in a JSON sidecar that no image transform touches.
        """
        if not augmentation_config:
            return augmentation_config
        geometric = ('horizontal_flip', 'vertical_flip', 'random_rotation', 'random_crop')
        if self.task_type in ('image_segmentation', 'object_detection', 'pose_estimation'):
            dropped = [key for key in geometric if augmentation_config.get(key)]
            if dropped:
                logger.warning(
                    f"{self.task_type} ignores {dropped}: the mask or the annotated "
                    "coordinates would not be transformed with the image."
                )
            return {k: v for k, v in augmentation_config.items() if k not in geometric}
        return augmentation_config

    def _build_transforms(self, augmentation_config=None, train=True):
        """Build torchvision transforms with optional augmentation."""
        aug = augmentation_config or {}
        base_ops = [transforms.Resize((224, 224))]

        if train:
            if aug.get('horizontal_flip', False):
                base_ops.append(transforms.RandomHorizontalFlip())
            if aug.get('vertical_flip', False):
                base_ops.append(transforms.RandomVerticalFlip())
            if aug.get('random_rotation', 0) > 0:
                base_ops.append(transforms.RandomRotation(aug['random_rotation']))
            if aug.get('color_jitter', False):
                base_ops.append(_color_jitter())
            if aug.get('random_crop', False):
                base_ops.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))

        base_ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transforms.Compose(base_ops)

    def _detection_augmentation(self, augmentation_config):
        """Photometric-only ops for detection/pose, run on the PIL image before rescaling."""
        aug = self._safe_augmentation(augmentation_config) or {}
        ops = []
        if aug.get('color_jitter', False):
            ops.append(_color_jitter())
        return transforms.Compose(ops) if ops else None

    # ------------------------------------------------------------------
    def train(self, data_dir, n_epochs=5, batch_size=8, lr=0.001,
              callback=None, mask_dir=None,
              augmentation_config=None, label_csv=None,
              val_split=0.2, optimizer_name='adam', annotation_file=None):
        """
        Train the CV model.

        Parameters
        ----------
        data_dir : str
            Path to dataset root.
        n_epochs : int
        batch_size : int
        lr : float
        callback : callable(epoch, acc, loss, duration, val_acc, val_loss)
        mask_dir : str
            Required for segmentation tasks.
        augmentation_config : dict
            Keys: horizontal_flip, vertical_flip, random_rotation (degrees),
                  color_jitter, random_crop.
        label_csv : str
            Path to multi-label CSV (required for 'image_multi_label').
        val_split : float
            Fraction of data to use as validation set.
        optimizer_name : str
            'adam', 'sgd', or 'rmsprop'.
        annotation_file : str
            COCO JSON for 'object_detection' / 'pose_estimation'. Discovered under
            data_dir when omitted.
        """
        train_tf = self._build_transforms(self._safe_augmentation(augmentation_config), train=True)
        val_tf = self._build_transforms(augmentation_config=None, train=False)

        # ------ Segmentation ------
        if self.task_type == 'image_segmentation':
            if not mask_dir:
                logger.error('mask_dir is required for segmentation')
                return None

            mask_tf = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
            reference = SegmentationDataset(
                data_dir, mask_dir, transform=train_tf, mask_transform=mask_tf)
            train_ds, val_ds = _split_with_transforms(
                lambda tf: SegmentationDataset(data_dir, mask_dir, transform=tf, mask_transform=mask_tf),
                len(reference), val_split, train_tf, val_tf)
            model = self.get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = self._make_optimizer(model, optimizer_name, lr)

            return self._run_training_loop(
                model, train_ds, val_ds, criterion, optimizer,
                n_epochs, batch_size, callback, segmentation=True)

        # ------ Object Detection / Pose Estimation ------
        elif self.task_type in ['object_detection', 'pose_estimation']:
            require_keypoints = self.task_type == 'pose_estimation'
            if not annotation_file:
                annotation_file = find_coco_annotation(data_dir) if data_dir else None
            if not annotation_file or not os.path.exists(annotation_file):
                raise ValueError(
                    f"{self.task_type} needs a COCO annotation JSON (with the 'images', "
                    f"'annotations' and 'categories' keys) somewhere under {data_dir}, whose "
                    "'file_name' entries match the image files. None was found."
                )

            # Two instances because training may jitter colours while validation must not;
            # both read the same JSON, so record i is the same image on either side.
            train_set = CocoDetectionDataset(data_dir, annotation_file, self.image_size,
                                             require_keypoints,
                                             self._detection_augmentation(augmentation_config))
            val_set = CocoDetectionDataset(data_dir, annotation_file, self.image_size,
                                           require_keypoints, None)
            # The keypoint head is built before the first epoch, so it has to cover the
            # widest annotation on either side of the split, not just the training side.
            self.num_keypoints = max(train_set.num_keypoints, val_set.num_keypoints)
            train_set.num_keypoints = val_set.num_keypoints = self.num_keypoints
            train_idx, val_idx = split_indices(len(train_set), val_split)

            # FastRCNNPredictor counts the background row, so class_names already carries it.
            self.num_classes = len(train_set.class_names)
            self.class_names = train_set.class_names

            train_ds, val_ds = Subset(train_set, train_idx), Subset(val_set, val_idx)
            model = self.get_model()
            optimizer = self._make_optimizer(model, optimizer_name, lr)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=False,
                                      collate_fn=detection_collate)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=False,
                                    collate_fn=detection_collate)

            return self._run_detection_loop(
                model, train_loader, val_loader, optimizer, n_epochs, callback)

        # ------ Multi-label ------
        elif self.task_type == 'image_multi_label':
            if not label_csv or not os.path.exists(label_csv):
                logger.error('label_csv is required for multi-label classification')
                return None

            reference = MultiLabelImageDataset(data_dir, label_csv, transform=train_tf)
            self.num_classes = len(reference.label_names)
            self.label_names = reference.label_names

            train_ds, val_ds = _split_with_transforms(
                lambda tf: MultiLabelImageDataset(data_dir, label_csv, transform=tf),
                len(reference), val_split, train_tf, val_tf)

            model = self.get_model()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = self._make_optimizer(model, optimizer_name, lr)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=False)

            return self._run_multilabel_loop(
                model, train_loader, val_loader, criterion, optimizer, n_epochs, callback)

        # ------ Standard Classification / Image Anomaly Detection ------
        else:
            try:
                reference = datasets.ImageFolder(data_dir, transform=train_tf)
                self.num_classes = len(reference.classes)
                self.class_names = reference.classes
            except Exception as e:
                logger.error(f'Error loading images: {e}')
                return None

            train_ds, val_ds = _split_with_transforms(
                lambda tf: datasets.ImageFolder(data_dir, transform=tf),
                len(reference), val_split, train_tf, val_tf)

            model = self.get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = self._make_optimizer(model, optimizer_name, lr)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=False)

            return self._run_classification_loop(
                model, train_loader, val_loader, criterion, optimizer, n_epochs, callback)

    # ------------------------------------------------------------------
    def _make_optimizer(self, model, name='adam', lr=1e-3):
        params = model.parameters()
        if name == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
        elif name == 'rmsprop':
            return optim.RMSprop(params, lr=lr)
        return optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------
    def _run_classification_loop(self, model, train_loader, val_loader,
                                  criterion, optimizer, n_epochs, callback):
        start_time = time.time()
        for epoch in range(n_epochs):
            # ----- Train -----
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss = running_loss / total if total else 0
            train_acc = correct / total if total else 0

            # ----- Validate -----
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, pred = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += pred.eq(targets).sum().item()

            val_loss = val_loss / val_total if val_total else 0
            val_acc = val_correct / val_total if val_total else 0
            duration = time.time() - start_time

            entry = {
                'epoch': epoch, 'acc': train_acc, 'loss': train_loss,
                'val_acc': val_acc, 'val_loss': val_loss
            }
            self.history.append(entry)

            if callback:
                callback(epoch, train_acc, train_loss, duration, val_acc, val_loss)

        self.best_model = model
        return model

    # ------------------------------------------------------------------
    def _run_multilabel_loop(self, model, train_loader, val_loader,
                              criterion, optimizer, n_epochs, callback):
        start_time = time.time()
        for epoch in range(n_epochs):
            # ----- Train -----
            model.train()
            running_loss, total = 0.0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

            train_loss = running_loss / total if total else 0

            # ----- Validate -----
            model.eval()
            val_loss, val_total = 0.0, 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    val_total += inputs.size(0)
                    preds = (torch.sigmoid(outputs) >= self.multilabel_threshold).float()
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

            val_loss = val_loss / val_total if val_total else 0
            # Subset accuracy (exact match)
            if all_preds:
                preds_cat = torch.cat(all_preds)
                tgts_cat = torch.cat(all_targets)
                val_acc = float((preds_cat == tgts_cat).all(dim=1).float().mean())
            else:
                val_acc = 0.0

            duration = time.time() - start_time
            train_acc = 0.0  # Not tracked per-batch for multi-label in train loop
            entry = {
                'epoch': epoch, 'acc': train_acc, 'loss': train_loss,
                'val_acc': val_acc, 'val_loss': val_loss
            }
            self.history.append(entry)

            if callback:
                callback(epoch, train_acc, train_loss, duration, val_acc, val_loss)

        self.best_model = model
        return model

    # ------------------------------------------------------------------
    def _to_device(self, images, targets):
        """Detection batches are lists of tensors plus dicts, so batch.to(device) does not fit."""
        return ([image.to(self.device) for image in images],
                [{key: value.to(self.device) for key, value in target.items()}
                 for target in targets])

    def _run_detection_loop(self, model, train_loader, val_loader, optimizer,
                            n_epochs, callback):
        """Faster / Keypoint R-CNN training: the model returns its own multi-part loss."""
        start_time = time.time()
        for epoch in range(n_epochs):
            model.train()
            running_loss, batches = 0.0, 0
            for images, targets in train_loader:
                images, targets = self._to_device(images, targets)
                optimizer.zero_grad()
                loss = sum(model(images, targets).values())
                loss.backward()
                # RPN, box, mask and keypoint heads are summed into one scalar while sitting
                # at very different scales, so an unclipped early step can wreck the backbone.
                clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += float(loss)
                batches += 1

            train_loss = running_loss / batches if batches else 0.0

            # Torchvision only returns the loss dict while the model is in training mode, so
            # the held-out loss costs its own pass; the boxes the metrics are measured on come
            # from an eval-mode pass, where the postprocessor runs and targets are ignored.
            # The backbone stays in eval during the loss pass, or validation images would move
            # the BatchNorm running statistics the model is scored with.
            val_loss_sum, val_batches = 0.0, 0
            predictions, ground_truths = [], []
            with torch.no_grad():
                model.train()
                model.backbone.eval()
                for images, targets in val_loader:
                    images, targets = self._to_device(images, targets)
                    val_loss_sum += float(sum(model(images, targets).values()))
                    val_batches += 1

                model.eval()
                for images, targets in val_loader:
                    batch = [image.to(self.device) for image in images]
                    predictions.extend(({k: v.cpu() for k, v in p.items()}
                                        for p in model(batch)))
                    ground_truths.extend(targets)

            val_loss = val_loss_sum / val_batches if val_batches else 0.0
            metrics = score_detections(ground_truths, predictions)
            # PCK is the headline metric when keypoints were annotated, F1 otherwise.
            quality = metrics['pck'] if metrics['pck'] is not None else metrics['f1']
            duration = time.time() - start_time

            entry = {
                'epoch': epoch, 'acc': 0.0, 'loss': train_loss,
                'val_acc': quality, 'val_loss': val_loss,
            }
            entry.update({k: v for k, v in metrics.items() if v is not None})
            self.history.append(entry)

            if callback:
                callback(epoch, 0.0, train_loss, duration, quality, val_loss)

        self.best_model = model
        return model

    # ------------------------------------------------------------------
    def _run_training_loop(self, model, train_ds, val_ds, criterion, optimizer,
                            n_epochs, batch_size, callback, segmentation=False):
        """Segmentation training loop with a held-out validation pass."""
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)
        start_time = time.time()
        for epoch in range(n_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if segmentation:
                    logits = outputs['out']
                else:
                    logits = outputs
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                if segmentation:
                    total += targets.nelement()
                    correct += predicted.eq(targets).sum().item()
                else:
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            epoch_loss = running_loss / len(train_ds)
            epoch_acc = correct / total if total else 0

            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    logits = outputs['out'] if segmentation else outputs
                    val_loss_sum += criterion(logits, targets).item() * inputs.size(0)
                    _, predicted = logits.max(1)
                    if segmentation:
                        val_total += targets.nelement()
                        val_correct += predicted.eq(targets).sum().item()
                    else:
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

            val_loss = val_loss_sum / max(1, len(val_ds))
            val_acc = val_correct / val_total if val_total else 0.0
            duration = time.time() - start_time
            entry = {'epoch': epoch, 'acc': epoch_acc, 'loss': epoch_loss,
                     'val_acc': val_acc, 'val_loss': val_loss}
            self.history.append(entry)
            if callback:
                callback(epoch, epoch_acc, epoch_loss, duration, val_acc, val_loss)

        self.best_model = model
        return model

    # ------------------------------------------------------------------
    def predict(self, image_path):
        """Run inference on a single image."""
        if self.best_model is None:
            return None

        self.best_model.eval()

        if self.task_type in ('object_detection', 'pose_estimation'):
            return self._predict_detections(image_path)

        transform = self._build_transforms(train=False)
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.best_model(input_tensor)
            if self.task_type == 'image_segmentation':
                logits = outputs['out']
                _, predicted = logits.max(1)
                return predicted.squeeze(0).cpu().numpy()
            elif self.task_type == 'image_multi_label':
                probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
                preds = (probs >= self.multilabel_threshold).astype(int)
                return {'probabilities': probs, 'predictions': preds,
                        'label_names': self.label_names}
            else:
                probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                predicted_class = int(probabilities.argmax())
                return {'class_id': predicted_class, 'probabilities': probabilities,
                        'class_names': self.class_names}

    # ------------------------------------------------------------------
    def _predict_detections(self, image_path):
        """Boxes, scores, labels and (for pose) keypoints in the original image's pixels.

        The model works on a square resize, so returning its raw coordinates would make the
        overlay land somewhere else as soon as the source image was not square.
        """
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        tensor = detection_transform(self.image_size)(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.best_model(tensor)[0]

        scale = torch.tensor([width / self.image_size, height / self.image_size,
                              width / self.image_size, height / self.image_size])
        result = {
            'boxes': (output['boxes'].cpu() * scale).numpy(),
            'scores': output['scores'].cpu().numpy(),
            'labels': output['labels'].cpu().numpy(),
            'class_names': self.class_names,
        }
        if 'keypoints' in output:
            keypoints = output['keypoints'].cpu()
            keypoints[..., 0] *= width / self.image_size
            keypoints[..., 1] *= height / self.image_size
            result['keypoints'] = keypoints.numpy()
        return result

    # ------------------------------------------------------------------
    def get_per_class_metrics(self):
        """
        Return per-class accuracy from training history if available.
        For multi-label: requires a dedicated evaluation pass.
        """
        return {}  # Placeholder; computed in app.py during evaluation.


# ---------------------------------------------------------------------------
# Explainability helper
# ---------------------------------------------------------------------------
def get_cv_explanation(model_name, params):
    explanations = {
        'resnet18':       'ResNet-18 uses skip connections (residual connections) that prevent gradient vanishing, allowing deeper networks to be trained efficiently.',
        'resnet50':       'ResNet-50 is a deeper and more powerful version of ResNet, with 3 internal layers per residual block (Bottleneck), excellent for larger datasets.',
        'mobilenet_v2':   'MobileNetV2 uses depthwise separable convolutions to drastically reduce parameters, ideal for resource-constrained devices.',
        'efficientnet_b0':'EfficientNet-B0 scales width, depth, and resolution in a balanced way, achieving high accuracy with lower computational cost.',
        'densenet121':    'DenseNet-121 connects each layer to all previous ones, promoting feature reuse and richer gradients during backprop.',
        'vgg16':          'VGG16 uses a simple and deep sequential architecture with 3x3 kernels, easy to understand but heavier in parameters.',
        'deeplabv3':      'DeepLabV3 uses Atrous Spatial Pyramid Pooling (ASPP) to capture objects at multiple scales in semantic segmentation.',
        'faster_rcnn':    'Faster R-CNN uses an integrated Region Proposal Network (RPN) to locate and classify objects simultaneously.',
        'pose_estimation':'Pose estimation predicts keypoints (joints) for each detected person/object to describe spatial body structure.',
        'image_anomaly_detection': 'Image anomaly detection learns visual normality patterns and flags samples that diverge from expected structure.',
        'lr':       f"The learning rate of {params.get('lr', 'N/A')} controls the weight adjustment speed. Too high causes divergence; too low makes training slow.",
        'batch_size': f"Batch size of {params.get('batch_size', 'N/A')} defines how many images the model sees before updating weights."
    }
    return explanations.get(model_name, 'Robust model for visual feature extraction.')
