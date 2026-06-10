"""Experimentos variando hiperparametros (consigna 5).

Se varia UN hiperparametro por vez dejando el resto en el valor base.
Hiperparametros estudiados (>= 4):
  1. learning_rate
  2. batch_size
  3. capacidad de la red (neuronas / capas ocultas)
  4. dropout

Para cada variante se mide en test: accuracy, precision, recall, F1.
Salida:
  - results/03_hp_<nombre>.png   (un grafico por hiperparametro)
  - results/03_hyperparams.json  (tabla completa de resultados)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

from data import load_data
from model import build_mlp
from utils import set_seed, fig_path, save_metrics

# Configuracion base comun a todos los experimentos.
BASE = dict(hidden_units=(16, 8), dropout=0.2, l2=0.0,
            learning_rate=1e-3, batch_size=32, epochs=100)


def train_eval(X_train, y_train, X_test, y_test, **cfg):
    set_seed(42)  # misma inicializacion -> comparacion justa
    c = {**BASE, **cfg}
    model = build_mlp(input_dim=X_train.shape[1],
                      hidden_units=c["hidden_units"], dropout=c["dropout"],
                      l2=c["l2"], learning_rate=c["learning_rate"])
    model.fit(X_train, y_train, validation_split=0.2,
              epochs=c["epochs"], batch_size=c["batch_size"], verbose=0)
    y_pred = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def bar_plot(labels, results, title, fname):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, [r[m] for r in results], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05); ax.set_ylabel("score (test)")
    ax.set_title(title); ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path(fname), dpi=130)
    plt.close(fig)


def main():
    X_train, X_test, y_train, y_test, cols = load_data()
    all_results = {}

    experiments = [
        ("learning_rate", "learning_rate",
         [3.0, 1e-1, 1e-2, 1e-3, 1e-5], "Variacion del learning rate",
         "03_hp_learning_rate.png"),
        ("batch_size", "batch_size",
         [8, 32, 128, len(X_train)], "Variacion del batch size",
         "03_hp_batch_size.png"),
        ("hidden_units", "hidden_units",
         [(4,), (16, 8), (64, 32), (256, 128, 64)],
         "Variacion de la capacidad de la red", "03_hp_capacidad.png"),
        ("dropout", "dropout",
         [0.0, 0.3, 0.6, 0.9], "Variacion del dropout", "03_hp_dropout.png"),
    ]

    for key, arg, values, title, fname in experiments:
        results, labels = [], []
        for v in values:
            res = train_eval(X_train, y_train, X_test, y_test, **{arg: v})
            results.append(res)
            labels.append(str(v if arg != "batch_size" or v != len(X_train)
                              else f"{v} (full)"))
            print(f"[{key}={v}] " +
                  " ".join(f"{m}={res[m]:.3f}" for m in res))
        bar_plot(labels, results, title, fname)
        all_results[key] = {"values": labels, "results": results}

    save_metrics("03_hyperparams.json", all_results)
    print("\nExperimentos de hiperparametros completados.")


if __name__ == "__main__":
    main()
