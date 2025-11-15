import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4  # glioma, meningioma, notumor, pituitary


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Define the same architecture used during training.
    For inference we don't need pretrained weights, just the structure.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(weights_path: str, device: str = "cpu") -> nn.Module:
    """
    Load model weights from disk and prepare for inference.
    """
    model = build_model()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
