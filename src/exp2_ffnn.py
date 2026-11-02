"""
Experiment 2 - Feed-forward neural network (Keras)

Single-hidden-layer FFNN (7 -> 16 ReLU -> 3 softmax) classifying SomaiyaSat
telemetry into Normal / Warning / Critical. Same data split as Exp 1, so the
hand-written NumPy network and the Keras FFNN can be compared directly.
"""
import time

import keras
from keras import layers

from common import HEALTH_CLASSES, METRICS, MODELS, PLOTS, plot_confusion, plot_history, save_json, set_seed, model_size_kb
from keras_utils import evaluate_classifier
from telemetry_data import classification_data, class_weights


def build_ffnn(n_in, hidden=16):
    return keras.Sequential([
        keras.Input((n_in,)),
        layers.Dense(hidden, activation="relu", name="hidden"),
        layers.Dense(3, activation="softmax", name="output"),
    ], name="ffnn")


def main():
    set_seed()
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = classification_data()
    model = build_ffnn(Xtr.shape[1])
    model.compile(optimizer=keras.optimizers.SGD(0.05, momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=128,
                     class_weight=class_weights(ytr), verbose=2)
    train_time = time.time() - t0
    res = {"experiment": "Exp 2 - Feed-forward neural network",
           "architecture": "7 -> Dense(16, ReLU) -> Dense(3, softmax)",
           "optimizer": "SGD + momentum 0.9, lr 0.05, batch 128, 40 epochs, class-weighted CE",
           "params": model.count_params(), "train_time_s": round(train_time, 1),
           **evaluate_classifier(model, Xte, yte, HEALTH_CLASSES), "history": hist.history}
    path = MODELS / "exp2_ffnn.keras"; model.save(path)
    res["model_size_kb"] = model_size_kb(path)
    save_json(res, METRICS / "exp2_ffnn.json")
    plot_history(hist, "Exp 2 FFNN", PLOTS / "exp2_training.png")
    plot_confusion(res["confusion_matrix"], HEALTH_CLASSES, "Exp 2 - FFNN (test)", PLOTS / "exp2_confusion.png")
    print({k: res[k] for k in ("params", "test_accuracy", "test_macro_f1", "model_size_kb")})


if __name__ == "__main__":
    main()
