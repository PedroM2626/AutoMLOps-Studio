import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Subset, Dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import time
import logging

logger = logging.getLogger(__name__)

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
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # Convert mask to long tensor for CrossEntropyLoss
            mask = (mask * 255).long().squeeze(0)
            
        return image, mask

class CVAutoMLTrainer:
    def __init__(self, task_type='image_classification', num_classes=2):
        self.task_type = task_type
        self.num_classes = num_classes
        self.class_names = [] # Store names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.history = []

    def get_model(self, model_name='resnet18'):
        if self.task_type == 'image_segmentation':
            # Use DeepLabV3 for segmentation
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            in_channels = model.classifier[4].in_channels
            model.classifier[4] = nn.Conv2d(in_channels, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            if model.aux_classifier:
                in_channels_aux = model.aux_classifier[4].in_channels
                model.aux_classifier[4] = nn.Conv2d(in_channels_aux, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            return model.to(self.device)
            
        elif self.task_type == 'object_detection':
            # Use Faster R-CNN for detection
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model.to(self.device)

        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        return model.to(self.device)

    def train(self, data_dir, n_epochs=5, batch_size=8, lr=0.001, callback=None, mask_dir=None):
        """Training for Classification, Segmentation or Detection."""
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.task_type == 'image_segmentation':
            if not mask_dir:
                logger.error("mask_dir is required for segmentation")
                return None
            
            mask_transforms = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
            
            full_dataset = SegmentationDataset(data_dir, mask_dir, transform=data_transforms, mask_transform=mask_transforms)
            dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
            model = self.get_model()
            criterion = nn.CrossEntropyLoss()
        elif self.task_type == 'object_detection':
            # Simplified for demo: detection usually requires custom dataset with boxes
            # Here we mock a training loop or use a simple classification structure if data is missing
            st.warning("Object Detection requires COCO-style JSON or VOC-style XML. Using pre-trained weights with custom head.")
            model = self.get_model()
            # Object detection in torchvision handles its own loss during forward in train mode
            optimizer = optim.Adam(model.parameters(), lr=lr)
            # Dummy training loop as we don't have box annotations in a simple folder structure
            for epoch in range(n_epochs):
                if callback: callback(epoch, 0.0, 0.0, 0.0)
            self.best_model = model
            return model
        else:
            try:
                full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
                self.num_classes = len(full_dataset.classes)
                self.class_names = full_dataset.classes # Save class names
            except Exception as e:
                logger.error(f"Error loading images: {e}")
                return None

            dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
            model = self.get_model('resnet18')
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if self.task_type == 'image_segmentation':
                    logits = outputs['out']
                    loss = criterion(logits, targets)
                else:
                    logits = outputs
                    loss = criterion(logits, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
                if self.task_type == 'image_segmentation':
                    _, predicted = logits.max(1)
                    total += targets.nelement()
                    correct += predicted.eq(targets).sum().item()
                else:
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            epoch_loss = running_loss / len(full_dataset)
            epoch_acc = correct / total
            duration = time.time() - start_time
            
            if callback:
                callback(epoch, epoch_acc, epoch_loss, duration)
            
            self.history.append({'epoch': epoch, 'acc': epoch_acc, 'loss': epoch_loss})

        self.best_model = model
        return model

    def predict(self, image_path):
        if self.best_model is None:
            return None
            
        self.best_model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.best_model(input_tensor)
            if self.task_type == 'image_segmentation':
                logits = outputs['out']
                _, predicted = logits.max(1)
                return predicted.squeeze(0).cpu().numpy()
            elif self.task_type == 'object_detection':
                # Returns list of dicts with boxes, labels, scores
                return outputs[0]
            else:
                _, predicted = outputs.max(1)
                return predicted.item()

def get_cv_explanation(model_name, params):
    explanations = {
        'resnet18': "ResNet-18 foi escolhido por seu equilíbrio entre profundidade e custo computacional. Utiliza 'skip connections' para mitigar o problema do desaparecimento do gradiente.",
        'mobilenet_v2': "MobileNetV2 é otimizado para dispositivos móveis, utilizando convoluções separáveis em profundidade para reduzir drasticamente o número de parâmetros.",
        'deeplabv3': "DeepLabV3 utiliza Atrous Spatial Pyramid Pooling (ASPP) para capturar objetos em múltiplas escalas e refinar os limites na segmentação semântica.",
        'faster_rcnn': "Faster R-CNN utiliza uma Region Proposal Network (RPN) integrada para identificar e localizar objetos simultaneamente com alta precisão.",
        'lr': f"O learning rate de {params.get('lr', 'N/A')} controla o quão rápido o modelo ajusta seus pesos. Um valor muito alto pode causar divergência, enquanto um muito baixo torna o treino lento.",
        'batch_size': f"Batch size de {params.get('batch_size', 'N/A')} define quantas imagens o modelo vê antes de atualizar os pesos, impactando a estabilidade do gradiente e o uso de memória GPU."
    }
    return explanations.get(model_name, "Modelo robusto para extração de características visuais.")
