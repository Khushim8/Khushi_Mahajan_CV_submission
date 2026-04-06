import logging
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)


def build_model(num_classes: int, dropout: float) -> nn.Module:
    # EfficientNet-B0: good balance of size vs accuracy for ~12k images.
    # Swap out the default 1000-class head for our 5-class problem.
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    logger.info("model: EfficientNet-B0, head in_features=%d, classes=%d", in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("backbone frozen")


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = True
    logger.info("backbone unfrozen for fine-tuning")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("using device: %s", device)
    return device


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
