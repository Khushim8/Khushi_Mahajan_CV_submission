import logging
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

CLASSES = ["crack", "hole", "normal", "rust", "scratch"]
CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# ImageNet mean/std since we're using a pretrained backbone
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def build_dataloaders(
    dataset_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {d}. Check dataset_dir in config.yaml."
            )

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_transforms(image_size, "train"),
    )
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=get_transforms(image_size, "test"),
    )

    if set(train_dataset.classes) != set(CLASSES):
        logger.warning("unexpected classes found: %s", train_dataset.classes)

    logger.info(
        "loaded %d train / %d test images across %d classes",
        len(train_dataset), len(test_dataset), len(train_dataset.classes),
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    return train_loader, test_loader, train_dataset.class_to_idx
