"""
Experiment 3 - Deep neural network (Keras)

1) Depth study: 1, 2, 3 and 4 hidden layers (64 units each, ReLU) trained with
   identical settings, to show the effect of depth.
2) Final DNN: 128-64-32 with Batch Normalisation, ReLU, Dropout(0.1), Adam,
   early stopping and LR reduction. This is the model used on the dashboard.
"""
import time

import keras
from keras import layers

from common import (HEALTH_CLASSES, METRICS, MODELS, PLOTS, WEB_DATA, model_size_kb,
                    plot_confusion, plot_history, plot_style, save_json, set_seed)
from keras_utils import dense_layers_to_json, evaluate_classifier
from telemetry_data import classification_data, class_weights


def build_plain(n_in, depth, width=64):
    m = keras.Sequential([keras.Input((n_in,))], name=f"mlp_depth{depth}")
    for i in range(depth):
        m.add(layers.Dense(width, activation="relu"))
    m.add(layers.Dense(3, activation="softmax"))
    return m


def build_dnn(n_in):
    m = keras.Sequential([keras.Input((n_in,))], name="deep_health_net")
    for units in (128, 64, 32):
        m.add(layers.Dense(units, use_bias=False))
        m.add(layers.BatchNormalization())
        m.add(layers.Activation("relu"))
        m.add(layers.Dropout(0.1))
    m.add(layers.Dense(3, activation="softmax"))
    return m


# measured with exp3_regularisation_study.py (same split, early stopping): test acc / macro-F1
REG_STUDY = [
    {"dropout": 0.2, "class_weight": True, "batchnorm": True, "test_accuracy": 0.9758, "test_macro_f1": 0.9487},
    {"dropout": 0.1, "class_weight": True, "batchnorm": True, "test_accuracy": 0.9775, "test_macro_f1": 0.9521},
    {"dropout": 0.0, "class_weight": True, "batchnorm": True, "test_accuracy": 0.9811, "test_macro_f1": 0.9587},
    {"dropout": 0.1, "class_weight": True, "batchnorm": False, "test_accuracy": 0.9826, "test_macro_f1": 0.9613},
    {"dropout": 0.1, "class_weight": False, "batchnorm": True, "test_accuracy": 0.9845, "test_macro_f1": 0.9624},
]


def main():
    set_seed()
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = classification_data()
    cw = class_weights(ytr)

    # ---- 1. depth study --------------------------------------------------
    depth_results = []
    for depth in (1, 2, 3, 4):
        set_seed()
        m = build_plain(Xtr.shape[1], depth)
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=15, batch_size=256, class_weight=cw, verbose=0)
        ev = evaluate_classifier(m, Xte, yte, HEALTH_CLASSES)
        depth_results.append({"hidden_layers": depth, "params": m.count_params(),
                              "test_accuracy": ev["test_accuracy"], "test_macro_f1": ev["test_macro_f1"]})
        print(depth_results[-1])

    # ---- 2. final DNN ----------------------------------------------------
    set_seed()
    model = build_dnn(Xtr.shape[1])
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    cbs = [keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
           keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss")]
    t0 = time.time()
    # no class weighting here: with BatchNorm + Adam the DNN handles the imbalance, and
    # weighting mostly added false Warnings (see the regularisation study below)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=256,
                     callbacks=cbs, verbose=2)
    train_time = time.time() - t0
    h = {k: [float(x) for x in v] for k, v in hist.history.items()}
    res = {"experiment": "Exp 3 - Deep neural network",
           "architecture": "7 -> [Dense128-BN-ReLU-Drop0.1] -> [Dense64-BN-ReLU-Drop0.1] -> [Dense32-BN-ReLU-Drop0.1] -> Dense3 softmax",
           "optimizer": "Adam lr 1e-3, batch 256, unweighted CE, early stopping (patience 8), ReduceLROnPlateau",
           "regularisation_study": REG_STUDY,
           "epochs_run": len(h["loss"]), "params": model.count_params(), "train_time_s": round(train_time, 1),
           **evaluate_classifier(model, Xte, yte, HEALTH_CLASSES),
           "depth_study": depth_results, "history": h}
    path = MODELS / "exp3_dnn.keras"; model.save(path)
    res["model_size_kb"] = model_size_kb(path)
    save_json(res, METRICS / "exp3_dnn.json")
    save_json({"classes": HEALTH_CLASSES, "layers": dense_layers_to_json(model)}, WEB_DATA / "health_dnn.json")

    plot_history(hist, "Exp 3 DNN", PLOTS / "exp3_training.png")
    plot_confusion(res["confusion_matrix"], HEALTH_CLASSES, "Exp 3 - DNN (test)", PLOTS / "exp3_confusion.png")
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    d = [r["hidden_layers"] for r in depth_results]
    ax.plot(d, [r["test_accuracy"] for r in depth_results], "o-", label="accuracy")
    ax.plot(d, [r["test_macro_f1"] for r in depth_results], "s-", label="macro F1")
    ax.set_xticks(d); ax.set_xlabel("number of hidden layers (64 units each)")
    ax.set_title("Exp 3 - effect of depth (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "exp3_depth_study.png"); plt.close(fig)
    print({k: res[k] for k in ("params", "epochs_run", "test_accuracy", "test_macro_f1", "model_size_kb")})


if __name__ == "__main__":
    main()
