"""Experimento base (consignas 3 y 6).

Entrena un MLP secuencial razonable, evalua en test y produce:
  - curva de entrenamiento (loss y accuracy)        -> results/01_baseline_curvas.png
  - matriz de confusion                              -> results/01_baseline_confusion.png
  - metricas (accuracy, precision, recall, F1)       -> results/01_baseline_metrics.json
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

from data import load_data, CLASS_NAMES
from model import build_mlp
from utils import set_seed, fig_path, save_metrics


def main():
    set_seed(42)
    X_train, X_test, y_train, y_test, cols = load_data()

    model = build_mlp(input_dim=X_train.shape[1],
                      hidden_units=(16, 8), dropout=0.2, l2=0.0,
                      learning_rate=1e-3)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=100, batch_size=32, verbose=0)

    # --- Curvas de entrenamiento ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoca"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoca"); ax[1].legend()
    fig.suptitle("Modelo base - curvas de entrenamiento")
    fig.tight_layout()
    fig.savefig(fig_path("01_baseline_curvas.png"), dpi=130)
    plt.close(fig)

    # --- Evaluacion en test ---
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig2, ax2 = plt.subplots(figsize=(5, 4.2))
    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title("Modelo base - matriz de confusion (test)")
    fig2.tight_layout()
    fig2.savefig(fig_path("01_baseline_confusion.png"), dpi=130)
    plt.close(fig2)

    out = {"metrics": metrics, "confusion_matrix": cm.tolist(),
           "n_test": int(len(y_test))}
    save_metrics("01_baseline_metrics.json", out)

    print("== Modelo base ==")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("  matriz de confusion [[TN, FP],[FN, TP]] =", cm.tolist())


if __name__ == "__main__":
    main()
