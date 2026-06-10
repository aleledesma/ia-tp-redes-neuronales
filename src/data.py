"""Carga del dataset de HuggingFace y preprocesamiento.

Dataset: Breast Cancer Wisconsin (Original) -- 'mstz/breast'
Tarea  : clasificacion binaria  ->  is_cancer = 1 (maligno) vs 0 (benigno).
9 features numericas (escala 1-10) obtenidas de biopsias.
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_NAME = "mstz/breast"
LABEL_COL = "is_cancer"
CLASS_NAMES = ["Benigno (0)", "Maligno (1)"]


def _to_dataframe():
    ds = load_dataset(DATASET_NAME)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split].to_pandas()


def load_data(test_size=0.2, seed=42, return_frame=False):
    """Devuelve X_train, X_test, y_train, y_test escalados + metadatos.

    El positivo (1) es el diagnostico Maligno (clase clinicamente relevante).
    """
    df = _to_dataframe()

    y = df[LABEL_COL].astype(int).values
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean()).values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    if return_frame:
        return X_train, X_test, y_train, y_test, feat_cols, df
    return X_train, X_test, y_train, y_test, feat_cols


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, cols, df = load_data(return_frame=True)
    print("Dataset:", DATASET_NAME)
    print("Total muestras:", len(df))
    print("Features:", len(cols), "->", cols)
    print("Train/Test:", Xtr.shape, Xte.shape)
    print("Balance train (pos=Maligno):", round(float(ytr.mean()), 3),
          "| test:", round(float(yte.mean()), 3))
