import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def generate_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    idx_to_class: Dict[int, str],
    dataset_dir: Path,
    output_path: str | Path,
) -> None:
    model.eval()
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    # grab paths from ImageFolder in the same order the loader will process them
    img_paths = [
        Path(p).relative_to(dataset_dir).as_posix()
        for p, _ in test_loader.dataset.samples
    ]

    preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(next(model.parameters()).device)
            batch_preds = model(images).argmax(dim=1).cpu().tolist()
            preds.extend(idx_to_class[p] for p in batch_preds)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_label"])
        writer.writerows(zip(img_paths, preds))

    logger.info("wrote %d predictions to %s", len(preds), output_path)
    logger.info("label distribution: %s", dict(Counter(preds)))
