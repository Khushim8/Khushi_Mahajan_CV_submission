## Setup

```bash
pip install -r requirements.txt
```

Dataset should be at `dataset/` path relative to the project folder structure.

## Running
```bash
python main.py
```

outputs:
- `predictions.csv` found in the output folder
- `outputs/best_model.pth` 
- `outputs/training.log`

## Structure

```
main.py         
config.yaml     hyperparameters
src/
  data.py       dataset loading + augmentation
  model.py      model definition 
  train.py      training loop
  evaluate.py   metrics: accuracy, confusion matrix, per-class
  predict.py    inference -> generates predictions.csv
```

## Results

| | accuracy | F1 |
|--|--|--|
| overall | 98.70% | |
| crack | 97.8% | 98.1% |
| hole | 99.8% | 98.8% |
| normal | 98.2% | 97.8% |
| rust | 98.5% | 99.2% |
| scratch | 99.2% | 99.6% |

Trained for 15 epochs on CPU. full confusion matrix in `outputs/training.log`.
---

## Approach

**Model:** EfficientNet-B0 pretrained on ImageNet. I considered starting with a simpler baseline (ResNet-18 or even a small custom CNN) but the dataset is only 12k images — enough to fine-tune but probably not enough to learn good texture features from scratch. B0 is also small enough that it trains reasonably fast on CPU.

**Training:** two phases. First ~7 epochs with the backbone frozen so the randomly initialized head converges without immediately destroying the pretrained weights. Then unfreeze everything and continue at a much lower lr (1e-4 vs 1e-3). This is pretty standard for fine-tuning pretrained vision models.

**Augmentation:** kept it minimal — flips, small rotation, slight brightness/contrast jitter. Didn't want to go heavier because the defect geometry actually matters here: a crack has directional texture that shouldn't be distorted too aggressively.

**config.yaml:** all hyperparameters live there so you can tweak and re-run without touching any code.

**What I skipped:** the metadata.csv has per-image features like lighting angle and noise strength, but using them would require a multimodal model (CNN + tabular features via a late-fusion head) which felt like over-engineering for this scope. The visual features alone should be sufficient given the class differences are visually obvious. A late-fusion extension would be straightforward if accuracy needed a further boost.

---

