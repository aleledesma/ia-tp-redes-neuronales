"""Simulacion de overfitting (consigna 4).

Se fuerza el sobreajuste con: red sobredimensionada, SIN regularizacion,
subconjunto de entrenamiento pequenio y muchas epocas. Se compara contra el
modelo base regularizado para evidenciar la brecha train/val.

Salida:
  - results/02_overfitting_curvas.png  (loss y accuracy train vs val)
  - results/02_overfitting_metrics.json
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data
from model import build_mlp
from utils import set_seed, fig_path, save_metrics


def main():
    set_seed(42)
    X_train, X_test, y_train, y_test, cols = load_data()

    # Subconjunto pequenio de entrenamiento para favorecer el sobreajuste.
    n_small = 60
    X_small, y_small = X_train[:n_small], y_train[:n_small]

    # Red grande y sin regularizacion -> mucha capacidad.
    model = build_mlp(input_dim=X_train.shape[1],
                      hidden_units=(256, 256, 128), dropout=0.0, l2=0.0,
                      learning_rate=1e-3)

    history = model.fit(X_small, y_small,
                        validation_data=(X_test, y_test),
                        epochs=300, batch_size=8, verbose=0)

    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["loss"], label="train")
    ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoca"); ax[0].legend()
    ax[1].plot(h["accuracy"], label="train")
    ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoca"); ax[1].legend()
    fig.suptitle(f"Overfitting forzado (n_train={n_small}, red 256-256-128, sin regularizacion)")
    fig.tight_layout()
    fig.savefig(fig_path("02_overfitting_curvas.png"), dpi=130)
    plt.close(fig)

    out = {
        "n_train_small": n_small,
        "final_train_loss": float(h["loss"][-1]),
        "final_val_loss": float(h["val_loss"][-1]),
        "final_train_acc": float(h["accuracy"][-1]),
        "final_val_acc": float(h["val_accuracy"][-1]),
        "min_val_loss": float(np.min(h["val_loss"])),
        "epoch_min_val_loss": int(np.argmin(h["val_loss"])),
    }
    save_metrics("02_overfitting_metrics.json", out)

    print("== Overfitting ==")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
