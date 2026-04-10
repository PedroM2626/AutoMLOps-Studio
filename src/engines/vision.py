import os
import time
import logging
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

logger = logging.getLogger(__name__)

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
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask * 255).long().squeeze(0)

        return image, mask


# ---------------------------------------------------------------------------
# Main CV Trainer
# ---------------------------------------------------------------------------
class CVAutoMLTrainer:
    def __init__(self, task_type='image_classification', num_classes=2,
                 backbone='resnet18', multilabel_threshold=0.5):
        """
        Parameters
        ----------
        task_type : str
            One of: 'image_classification', 'image_multi_label',
                'image_segmentation', 'object_detection',
                'image_anomaly_detection', 'pose_estimation'
        num_classes : int
            Number of output classes / labels.
        backbone : str
            Backbone key (see BACKBONE_REGISTRY).
        multilabel_threshold : float
            Sigmoid threshold for multi-label positive prediction.
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.backbone = backbone
        self.multilabel_threshold = multilabel_threshold
        self.class_names = []
        self.label_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model = None
        self.history = []  # list of epoch dicts

    # ------------------------------------------------------------------
    def get_model(self):
        """Build the model head for the selected task and backbone."""
        if self.task_type == 'image_segmentation':
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            in_channels = model.classifier[4].in_channels
            model.classifier[4] = nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
            if model.aux_classifier:
                aux_in = model.aux_classifier[4].in_channels
                model.aux_classifier[4] = nn.Conv2d(aux_in, self.num_classes, kernel_size=1)
            return model.to(self.device)

        elif self.task_type == 'object_detection':
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model.to(self.device)

        elif self.task_type == 'pose_estimation':
            model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            return model.to(self.device)

        else:
            # Classification or Multi-label
            builder = BACKBONE_REGISTRY.get(self.backbone, BACKBONE_REGISTRY['resnet18'])
            model = builder(self.num_classes)
            return model.to(self.device)

    # ------------------------------------------------------------------
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
                base_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                        saturation=0.2, hue=0.1))
            if aug.get('random_crop', False):
                base_ops.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))

        base_ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transforms.Compose(base_ops)

    # ------------------------------------------------------------------
    def train(self, data_dir, n_epochs=5, batch_size=8, lr=0.001,
              callback=None, mask_dir=None,
              augmentation_config=None, label_csv=None,
              val_split=0.2, optimizer_name='adam'):
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
        """
        train_tf = self._build_transforms(augmentation_config, train=True)
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
            full_dataset = SegmentationDataset(
                data_dir, mask_dir, transform=train_tf, mask_transform=mask_tf)
            model = self.get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = self._make_optimizer(model, optimizer_name, lr)

            return self._run_training_loop(
                model, full_dataset, full_dataset, criterion, optimizer,
                n_epochs, batch_size, callback, segmentation=True)

        # ------ Object Detection (demo stub) ------
        elif self.task_type in ['object_detection', 'pose_estimation']:
            model = self.get_model()
            optimizer = self._make_optimizer(model, optimizer_name, lr)
            for epoch in range(n_epochs):
                if callback:
                    callback(epoch, 0.0, 0.0, 0.0, 0.0, 0.0)
                self.history.append({'epoch': epoch, 'acc': 0.0, 'loss': 0.0,
                                     'val_acc': 0.0, 'val_loss': 0.0})
            self.best_model = model
            return model

        # ------ Multi-label ------
        elif self.task_type == 'image_multi_label':
            if not label_csv or not os.path.exists(label_csv):
                logger.error('label_csv is required for multi-label classification')
                return None

            full_dataset = MultiLabelImageDataset(data_dir, label_csv, transform=train_tf)
            self.num_classes = len(full_dataset.label_names)
            self.label_names = full_dataset.label_names

            val_size = max(1, int(len(full_dataset) * val_split))
            train_size = len(full_dataset) - val_size
            train_ds, val_ds = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42))
            # Apply val transforms to val subset
            val_ds.dataset.transform = val_tf

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
                full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
                self.num_classes = len(full_dataset.classes)
                self.class_names = full_dataset.classes
            except Exception as e:
                logger.error(f'Error loading images: {e}')
                return None

            val_size = max(1, int(len(full_dataset) * val_split))
            train_size = len(full_dataset) - val_size
            train_ds, val_ds = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42))

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
    def _run_training_loop(self, model, train_ds, val_ds, criterion, optimizer,
                            n_epochs, batch_size, callback, segmentation=False):
        """Segmentation training loop (no validation split)."""
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
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
                if segmentation:
                    _, predicted = logits.max(1)
                    total += targets.nelement()
                    correct += predicted.eq(targets).sum().item()
                else:
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            epoch_loss = running_loss / len(train_ds)
            epoch_acc = correct / total if total else 0
            duration = time.time() - start_time
            entry = {'epoch': epoch, 'acc': epoch_acc, 'loss': epoch_loss,
                     'val_acc': 0.0, 'val_loss': 0.0}
            self.history.append(entry)
            if callback:
                callback(epoch, epoch_acc, epoch_loss, duration, 0.0, 0.0)

        self.best_model = model
        return model

    # ------------------------------------------------------------------
    def predict(self, image_path):
        """Run inference on a single image."""
        if self.best_model is None:
            return None

        self.best_model.eval()
        transform = self._build_transforms(train=False)
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.task_type in ['object_detection', 'pose_estimation']:
                outputs = self.best_model([input_tensor.squeeze(0)])
            else:
                outputs = self.best_model(input_tensor)
            if self.task_type == 'image_segmentation':
                logits = outputs['out']
                _, predicted = logits.max(1)
                return predicted.squeeze(0).cpu().numpy()
            elif self.task_type == 'object_detection':
                return outputs[0]
            elif self.task_type == 'pose_estimation':
                return outputs[0]
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
