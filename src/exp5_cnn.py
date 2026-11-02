"""
Experiment 5 - Convolutional Neural Network (from scratch) on EuroSAT RGB

A compact VGG-style CNN classifies 64x64 Sentinel-2 patches into 10 land-cover
classes. It is designed to be small enough for on-board / edge use
(~0.29 M parameters, ~1.1 MB of float32 weights) and runs in the browser on
the project dashboard.

Usage:  python src/exp5_cnn.py [epochs]
"""
import sys
import time

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from common import (EUROSAT_CLASSES, METRICS, MODELS, PLOTS, model_size_kb, plot_confusion,
                    plot_history, save_json, set_seed)
from eurosat_data import eurosat_splits
from keras_utils import evaluate_classifier


def conv_bn(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


# per-channel mean / std of the EuroSAT RGB training split (0-255 scale)
RGB_MEAN = [87.69, 96.89, 103.93]
RGB_STD = [51.62, 34.85, 29.43]


def build_cnn():
    inp = keras.Input((64, 64, 3), name="image")
    x = layers.Normalization(mean=RGB_MEAN, variance=[s ** 2 for s in RGB_STD], name="standardize")(inp)
    x = conv_bn(x, 32, "b1c1"); x = conv_bn(x, 32, "b1c2"); x = layers.MaxPooling2D(name="b1_pool")(x)
    x = conv_bn(x, 64, "b2c1"); x = conv_bn(x, 64, "b2c2"); x = layers.MaxPooling2D(name="b2_pool")(x)
    x = conv_bn(x, 128, "b3c1"); x = layers.MaxPooling2D(name="b3_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(10, activation="softmax", name="classifier")(x)
    return keras.Model(inp, out, name="somaiyasat_cnn")


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))   # satellite views have no "up"
    return img, label


def main(epochs=25):
    set_seed()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = eurosat_splits()
    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr.astype(np.float32), ytr)).shuffle(4096, seed=42)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((Xva.astype(np.float32), yva)).batch(256)

    model = build_cnn()
    model.summary()
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [keras.callbacks.ReduceLROnPlateau(factor=0.4, patience=2, monitor="val_loss", verbose=1),
           keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss")]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
    train_time = time.time() - t0

    path = MODELS / "exp5_cnn.keras"; model.save(path)
    ev = evaluate_classifier(model, Xte.astype(np.float32), yte, EUROSAT_CLASSES)
    t1 = time.time(); model.predict(Xte[:512].astype(np.float32), verbose=0, batch_size=1); lat = (time.time() - t1) / 512
    res = {"experiment": "Exp 5 - CNN from scratch (EuroSAT RGB)",
           "architecture": "2x[Conv32-BN-ReLU] Pool, 2x[Conv64-BN-ReLU] Pool, [Conv128-BN-ReLU] Pool, GAP, Dropout0.3, Dense10",
           "optimizer": "Adam lr 1e-3, batch 64, ReduceLROnPlateau, early stopping; flip + rot90 augmentation",
           "split": "stratified 70/15/15: 18900 train / 4050 val / 4050 test",
           "epochs_run": len(hist.history["loss"]), "params": model.count_params(),
           "train_time_s": round(train_time, 1), "cpu_latency_ms_per_image": round(lat * 1000, 2),
           "model_size_kb": model_size_kb(path), **ev,
           "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()}}
    save_json(res, METRICS / "exp5_cnn.json")
    plot_history(hist, "Exp 5 CNN", PLOTS / "exp5_training.png")
    plot_confusion(ev["confusion_matrix"], EUROSAT_CLASSES, "Exp 5 - CNN (test)", PLOTS / "exp5_confusion.png")
    print({k: res[k] for k in ("params", "epochs_run", "test_accuracy", "test_macro_f1", "model_size_kb", "train_time_s")})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
