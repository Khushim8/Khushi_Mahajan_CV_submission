import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import freeze_backbone, unfreeze_backbone, count_trainable_params

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
) -> nn.Module:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(cfg["output"]["model_checkpoint"])
    checkpoint_path = base.with_stem(f"{base.stem}_{ts}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = cfg["training"]["epochs"]
    fine_tune_ep = cfg["training"]["fine_tune_epoch"]
    weight_decay = cfg["training"]["weight_decay"]

    criterion = nn.CrossEntropyLoss()

    # start with backbone frozen, only train the head
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if epoch == fine_tune_ep:
            # unfreeze and drop lr significantly to avoid wrecking pretrained weights
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["fine_tune_lr"],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - fine_tune_ep + 1
            )

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = _run_epoch(model, test_loader, criterion, optimizer, device, training=False)
        scheduler.step()

        logger.info(
            "epoch %02d/%02d  train_loss=%.4f acc=%.1f%%  val_loss=%.4f acc=%.1f%%",
            epoch, epochs, train_loss, train_acc * 100, val_loss, val_acc * 100,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  saved checkpoint (val_acc=%.1f%%)", best_acc * 100)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info("done. best val_acc=%.1f%%", best_acc * 100)
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total
