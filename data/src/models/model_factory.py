import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower().strip()

    if name == "faster_rcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

   
    elif name == "ssd":
        model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
        in_channels = [out_ch for out_ch in model.head.classification_head.module_list][0].in_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        return model

    elif name == "yolo" or name == "yolo11":
        from src.models.yolo_model import OriginalYOLO11Wrapper
        return OriginalYOLO11Wrapper(num_classes=num_classes)

    elif name == "efficientdet" or name == "fcos":

        model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )
        return model

    elif name == "detr" or name == "faster_rcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Модель {model_name} не поддерживается.")