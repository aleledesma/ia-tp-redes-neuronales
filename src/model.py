"""Construccion del MLP secuencial de Keras."""
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_mlp(input_dim, hidden_units=(16, 8), dropout=0.0, l2=0.0,
              learning_rate=1e-3, activation="relu"):
    """Crea un MLP secuencial para clasificacion binaria.

    input_dim     : numero de features de entrada
    hidden_units  : tupla con la cantidad de neuronas por capa oculta
    dropout       : tasa de dropout aplicada despues de cada capa oculta (0 = sin dropout)
    l2            : coeficiente de regularizacion L2 en las capas ocultas (0 = sin L2)
    learning_rate : learning rate del optimizador Adam
    activation    : funcion de activacion de las capas ocultas
    """
    reg = regularizers.l2(l2) if l2 > 0 else None
    model = keras.Sequential(name="mlp")
    model.add(keras.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation=activation, kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    # Salida: 1 neurona sigmoide -> probabilidad de la clase positiva
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
