# Shared data loading, normalization and helpers used by all task files
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
FEATURES = ['variance', 'skewness', 'curtosis', 'entropy']

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
LOCAL_FILE = "BankNote_Authentication.csv"   # Kaggle file name; put it next to the notebook if you downloaded it

if os.path.exists(LOCAL_FILE):
    df = pd.read_csv(LOCAL_FILE)
    df.columns = FEATURES + ['class']
else:
    df = pd.read_csv(UCI_URL, header=None, names=FEATURES + ['class'])

print(df.shape)
print(df['class'].value_counts())

X_raw = df[FEATURES].values.astype(float)
y = df['class'].values.astype(int)          # 0 = genuine, 1 = forged

Xtr_raw, Xte_raw, ytr, yte = train_test_split(
    X_raw, y, test_size=0.2, random_state=SEED, stratify=y)

# Min-max normalization (statistics computed on the training set only)
mn, mx = Xtr_raw.min(axis=0), Xtr_raw.max(axis=0)
def normalize(X):
    return (X - mn) / (mx - mn)

Xtr, Xte = normalize(Xtr_raw), normalize(Xte_raw)
print("Train:", Xtr.shape, " Test:", Xte.shape)
from perceptron import Perceptron, accuracy

def train_on(cols, order=None, lr=0.1, max_epochs=100):
    # train on the selected feature columns (list of indices), optionally with a custom sample order (index array)
    idx = np.arange(len(ytr)) if order is None else order
    m = Perceptron(lr=lr, max_epochs=max_epochs)
    m.fit(Xtr[idx][:, cols], ytr[idx])
    return m

ALL = list(range(4))
BORDER_EPS = 0.05    # a sample is "borderline" if its distance to the boundary is below this (normalized units)

def distance_to_boundary(model, X):
    norm = np.linalg.norm(model.w)
    return np.abs(model.weighted_sum(X)) / (norm if norm > 0 else 1.0)