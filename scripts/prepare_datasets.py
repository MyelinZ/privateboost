"""Download and prepare all datasets for PrivateBoost benchmarks.

Outputs clean numeric CSVs with a 'target' column to data/.
"""
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def prepare_heart_disease():
    """UCI Heart Disease (Cleveland). 297 samples, 13 features."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    path = os.path.join(DATA_DIR, "heart_disease_raw.csv")
    out = os.path.join(DATA_DIR, "heart_disease.csv")

    if not os.path.exists(path):
        print("Downloading Heart Disease...")
        urllib.request.urlretrieve(url, path)

    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]
    df = pd.read_csv(path, names=cols, na_values="?")
    df = df.dropna()
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(out, index=False)
    print(f"  Heart Disease: {len(df)} rows, {len(df.columns)} cols -> {out}")


def prepare_breast_cancer():
    """sklearn Breast Cancer Wisconsin. 569 samples, 30 features."""
    out = os.path.join(DATA_DIR, "breast_cancer.csv")

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(out, index=False)
    print(f"  Breast Cancer: {len(df)} rows, {len(df.columns)} cols -> {out}")


def prepare_pima_diabetes():
    """Pima Indians Diabetes (OpenML/UCI). 768 samples, 8 features."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    path = os.path.join(DATA_DIR, "pima_raw.csv")
    out = os.path.join(DATA_DIR, "pima_diabetes.csv")

    if not os.path.exists(path):
        print("Downloading Pima Diabetes...")
        urllib.request.urlretrieve(url, path)

    cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "target",
    ]
    df = pd.read_csv(path, names=cols)
    df.to_csv(out, index=False)
    print(f"  Pima Diabetes: {len(df)} rows, {len(df.columns)} cols -> {out}")


def prepare_cdc_brfss():
    """CDC BRFSS Diabetes Health Indicators. ~70k balanced samples, 21 features."""
    out = os.path.join(DATA_DIR, "cdc_diabetes.csv")

    if os.path.exists(out):
        df = pd.read_csv(out)
        print(f"  CDC BRFSS: {len(df)} rows, {len(df.columns)} cols -> {out} (already exists)")
        return

    from ucimlrepo import fetch_ucirepo

    print("Downloading CDC BRFSS...")
    dataset = fetch_ucirepo(id=891)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    df = X.copy()
    df["target"] = y.astype(int)

    # Balance: downsample majority class
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    n_min = min(len(pos), len(neg))
    rng = np.random.RandomState(42)
    neg_sampled = neg.sample(n=n_min, random_state=rng)
    df = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=rng).reset_index(drop=True)

    # Move target to last column
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    df.to_csv(out, index=False)
    print(f"  CDC BRFSS: {len(df)} rows, {len(df.columns)} cols -> {out}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Preparing datasets...\n")
    prepare_heart_disease()
    prepare_breast_cancer()
    prepare_pima_diabetes()
    prepare_cdc_brfss()
    print("\nDone.")


if __name__ == "__main__":
    main()
