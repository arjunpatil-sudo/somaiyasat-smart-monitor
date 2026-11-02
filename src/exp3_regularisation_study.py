"""Exp 3 (supporting) - regularisation study for the 128-64-32 DNN.

Trains the same network with different dropout / BatchNorm / class-weight
settings. Its results are recorded in REG_STUDY inside exp3_dnn.py.
"""
import keras
from keras import layers

from common import HEALTH_CLASSES, set_seed
from keras_utils import evaluate_classifier
from telemetry_data import classification_data, class_weights


def main():
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = classification_data()
    cw = class_weights(ytr)
    for drop, use_cw, bn in [(0.2, True, True), (0.1, True, True), (0.0, True, True), (0.1, True, False), (0.1, False, True)]:
        set_seed()
        m = keras.Sequential([keras.Input((7,))])
        for u in (128, 64, 32):
            if bn:
                m.add(layers.Dense(u, use_bias=False)); m.add(layers.BatchNormalization()); m.add(layers.Activation("relu"))
            else:
                m.add(layers.Dense(u, activation="relu"))
            if drop:
                m.add(layers.Dropout(drop))
        m.add(layers.Dense(3, activation="softmax"))
        m.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=256, class_weight=cw if use_cw else None, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
        e = evaluate_classifier(m, Xte, yte, HEALTH_CLASSES)
        print({"dropout": drop, "class_weight": use_cw, "batchnorm": bn,
               "test_accuracy": e["test_accuracy"], "test_macro_f1": e["test_macro_f1"]})


if __name__ == "__main__":
    main()
