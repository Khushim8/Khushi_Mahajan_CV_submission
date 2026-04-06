import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    n = len(class_names)
    cm = [[0] * n for _ in range(n)]
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    accuracy = sum(cm[i][i] for i in range(n)) / len(all_labels)
    per_class = {
        cls: cm[i][i] / sum(cm[i]) if sum(cm[i]) > 0 else 0.0
        for i, cls in enumerate(class_names)
    }

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for i, cls in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision[cls] = prec
        per_class_recall[cls]    = rec
        per_class_f1[cls]        = f1

    results = {"accuracy": accuracy, "per_class_acc": per_class, "confusion_matrix": cm, "class_names": class_names}
    results.update({
        "per_class_precision": per_class_precision,
        "per_class_recall":    per_class_recall,
        "per_class_f1":        per_class_f1,
    })
    _print_results(results)
    return results


def _print_results(results: Dict) -> None:
    class_names = results["class_names"]
    cm = results["confusion_matrix"]
    n = len(class_names)

    logger.info("overall accuracy: %.2f%%", results["accuracy"] * 100)
    logger.info("per-class accuracy:")
    for cls, acc in results["per_class_acc"].items():
        logger.info("  %s: %.1f%%", cls, acc * 100)

    logger.info("per-class F1:")
    for cls, f1 in results["per_class_f1"].items():
        logger.info("  %s: %.1f%%", cls, f1 * 100)

    logger.info("confusion matrix (rows=true, cols=pred):")
    logger.info("         " + "".join(f"{c[:6]:>8}" for c in class_names))
    for i, cls in enumerate(class_names):
        logger.info("  %-8s" + "".join(f"{cm[i][j]:>8}" for j in range(n)), cls[:8])
