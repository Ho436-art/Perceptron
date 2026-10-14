# Perceptron Decision Boundary & Linear Separability Challenge

Single-Layer Perceptron (step activation, perceptron learning rule) on the
[Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication).
Features: variance, skewness, curtosis, entropy. Target: 0 = genuine, 1 = forged.
Features are min-max normalized.

## Project structure

| File | Purpose |
|---|---|
| `perceptron.py` | Perceptron class (step activation, perceptron learning rule) |
| `common.py` | Dataset loading, train/test split, normalization, helpers |
| `task1_banknote_classification.py` | 1. Banknote Classification using Perceptron |
| `task2_decision_boundary.py` | 2. Decision Boundary Explorer |
| `task3_feature_pairs.py` | 3. Feature Pair Separability Challenge |
| `task4_sample_order.py` | 4. Training Sample Order Challenge |
| `task5_borderline.py` | 5. Borderline Sample Detector |
| `task6_xor.py` | 6. XOR Limitation Demonstrator |
| `task7_feature_removal.py` | 7. Perceptron Feature Removal Challenge |
| `task8_report.py` | 8. Perceptron Classification Report |
| `main_menu.py` | Menu-driven program (options 1-9) |
| `perceptron_banknote.ipynb` | Same code as a single Jupyter notebook |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

By default the data is downloaded from the UCI repository when a script runs.
To use the Kaggle copy instead, place `BankNote_Authentication.csv` in this
folder (see https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data).

## Run

```bash
python main_menu.py                     # full menu
python task3_feature_pairs.py           # or run any task on its own
```

Training is capped at 100 epochs per run. The real dataset is not perfectly
linearly separable, so some runs report "not converged".
