"""
main.py - trains a defect classifier on the Rigetti dataset and outputs predictions.csv

usage:
    python main.py                   # full run
    python main.py --skip-train      # skip to eval (needs outputs/best_model.pth)
    python main.py --config path/to/config.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data import build_dataloaders
from src.model import build_model, get_device
from src.train import train
from src.evaluate import evaluate
from src.predict import generate_predictions


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--retrain", metavar="DIR", default=None,
                        help="path to folder of new labeled images for fine-tuning from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["output"]["log_file"])
    logger = logging.getLogger(__name__)
    logger.info("starting pipeline")

    set_seed(cfg["training"]["seed"])
    device = get_device()

    train_loader, test_loader, class_to_idx = build_dataloaders(
        dataset_dir=cfg["data"]["dataset_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(cfg["model"]["num_classes"])]

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["dropout"]).to(device)

    if args.skip_train:
        ckpt = cfg["output"]["model_checkpoint"]
        logger.info("loading checkpoint: %s", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    elif args.retrain:
        ckpt = cfg["output"]["model_checkpoint"]
        logger.info("retraining from checkpoint: %s on new data: %s", ckpt, args.retrain)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        retrain_loader, _, _ = build_dataloaders(
            dataset_dir=args.retrain,
            image_size=cfg["data"]["image_size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )
        cfg_retrain = dict(cfg)
        cfg_retrain["training"] = dict(cfg["training"])
        cfg_retrain["training"]["epochs"] = 5
        cfg_retrain["training"]["learning_rate"] = cfg["training"]["fine_tune_lr"]
        cfg_retrain["training"]["fine_tune_epoch"] = 999  # skip re-freeze phase
        model = train(model, retrain_loader, test_loader, cfg_retrain, device)
    else:
        model = train(model, train_loader, test_loader, cfg, device)

    results = evaluate(model, test_loader, class_names, device)

    generate_predictions(
        model=model,
        test_loader=test_loader,
        idx_to_class=idx_to_class,
        dataset_dir=Path(cfg["data"]["dataset_dir"]),
        output_path=cfg["output"]["predictions_csv"],
    )

    logger.info("final accuracy: %.2f%%", results["accuracy"] * 100)


if __name__ == "__main__":
    main()
