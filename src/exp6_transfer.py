"""
Experiment 6 - Transfer learning with a pretrained CNN (MobileNetV2, ImageNet)

Two width multipliers are compared, to trade accuracy for size on low-power
satellite hardware:
    alpha = 1.0   (2.26 M backbone params)
    alpha = 0.35  (0.41 M backbone params)

Phase A - feature extraction: backbone frozen, only a new classifier head is
          trained (backbone features are computed once and cached).
Phase B - fine-tuning: blocks 10-16 + the last 1x1 conv are unfrozen and the
          whole model is trained end-to-end with a small learning rate.

EuroSAT 64x64 patches are resized to 128x128 inside the model.
ImageNet weights: keras-applications release files (see README), placed in
data/pretrained/. If missing, keras downloads them automatically.

Usage: python src/exp6_transfer.py [finetune_epochs]
"""
import sys
import time

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from common import (DATA, EUROSAT_CLASSES, METRICS, MODELS, PLOTS, model_size_kb, plot_confusion,
                    plot_style, save_json, set_seed)
from eurosat_data import eurosat_splits
from exp5_cnn import augment
from keras_utils import evaluate_classifier

IMG = 128
WEIGHTS = {
    1.0: DATA / "pretrained" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
    0.35: DATA / "pretrained" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96_no_top.h5",
}


def build(alpha):
    w = WEIGHTS[alpha]
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), alpha=alpha, include_top=False,
                                          weights=str(w) if w.exists() else "imagenet")
    base.trainable = False
    inp = keras.Input((64, 64, 3), name="image")
    x = layers.Resizing(IMG, IMG, name="resize")(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(10, activation="softmax", name="classifier")(x)
    return keras.Model(inp, out, name=f"mobilenetv2_{alpha}"), base


def batches(X, y=None, bs=128):
    """uint8 arrays -> float32 batches on the fly (keeps RAM low)."""
    ds = tf.data.Dataset.from_tensor_slices(X if y is None else (X, y))
    cast = (lambda a: tf.cast(a, tf.float32)) if y is None else (lambda a, b: (tf.cast(a, tf.float32), b))
    return ds.batch(bs).map(cast).prefetch(2)


def features(model, X):
    """Output of GAP layer (frozen backbone) for cached feature extraction."""
    f = keras.Model(model.input, model.get_layer("gap").output)
    return f.predict(batches(X), verbose=0)


def run(alpha, ft_epochs, data):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    set_seed()
    model, base = build(alpha)
    backbone_params = base.count_params()

    # ---------- Phase A: feature extraction ----------
    t0 = time.time()
    Ftr, Fva = features(model, Xtr), features(model, Xva)
    head_in = keras.Input((Ftr.shape[1],))
    h = model.get_layer("dropout")(head_in); h = model.get_layer("classifier")(h)
    head = keras.Model(head_in, h)
    head.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    hA = head.fit(Ftr, ytr, validation_data=(Fva, yva), epochs=30, batch_size=128, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    timeA = time.time() - t0
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])   # head layers are shared
    evA = evaluate_classifier(model, Xte, yte, EUROSAT_CLASSES)
    print(f"alpha {alpha} feature-extraction test acc {evA['test_accuracy']}")

    # ---------- Phase B: fine-tuning ----------
    base.trainable = True
    n_layers = len(base.layers)
    fine_tune_at = [i for i, l in enumerate(base.layers) if l.name.startswith("block_10")][0]
    for l in base.layers[:fine_tune_at]:
        l.trainable = False
    for l in base.layers:                       # keep BN statistics frozen (standard practice)
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    model.compile(keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(4096, seed=42)
                .map(lambda a, b: augment(tf.cast(a, tf.float32), b), num_parallel_calls=2).batch(64).prefetch(2))
    t0 = time.time()
    hB = model.fit(train_ds, validation_data=batches(Xva, yva), epochs=ft_epochs, verbose=2,
                   callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                              keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)])
    timeB = time.time() - t0
    evB = evaluate_classifier(model, Xte, yte, EUROSAT_CLASSES)
    path = MODELS / f"exp6_mobilenetv2_a{alpha}.keras"; model.save(path)
    t1 = time.time(); model.predict(Xte[:256].astype(np.float32), verbose=0, batch_size=1); lat = (time.time() - t1) / 256
    print(f"alpha {alpha} fine-tuned test acc {evB['test_accuracy']}")
    return {
        "alpha": alpha, "input_size": IMG, "backbone_params": int(backbone_params),
        "total_params": int(model.count_params()), "backbone_layers": n_layers,
        "fine_tune_from_layer": int(fine_tune_at), "trainable_params_finetune": trainable,
        "feature_extraction": {"test_accuracy": evA["test_accuracy"], "test_macro_f1": evA["test_macro_f1"],
                               "time_s": round(timeA, 1), "epochs": len(hA.history["loss"]),
                               "history": {k: [float(v) for v in vals] for k, vals in hA.history.items()}},
        "fine_tuned": {"test_accuracy": evB["test_accuracy"], "test_macro_f1": evB["test_macro_f1"],
                       "time_s": round(timeB, 1), "epochs": len(hB.history["loss"]),
                       "confusion_matrix": evB["confusion_matrix"], "report": evB["report"],
                       "history": {k: [float(v) for v in vals] for k, vals in hB.history.items()}},
        "model_size_kb": model_size_kb(path), "cpu_latency_ms_per_image": round(lat * 1000, 2),
    }


def main(ft_epochs=12):
    data = eurosat_splits()                    # kept as uint8; cast per batch
    results = [run(a, ft_epochs, data) for a in (1.0, 0.35)]
    save_json({"experiment": "Exp 6 - Transfer learning (MobileNetV2, ImageNet)", "runs": results},
              METRICS / "exp6_transfer.json")

    plt = plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for r in results:
        hA, hB = r["feature_extraction"]["history"], r["fine_tuned"]["history"]
        va = hA["val_accuracy"] + hB["val_accuracy"]
        ax[0].plot(range(1, len(va) + 1), va, label=f"alpha {r['alpha']}")
        ax[0].axvline(len(hA["val_accuracy"]) + 0.5, ls=":", c="grey")
    ax[0].set_title("Exp 6 - validation accuracy\n(head training | fine-tuning)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    names = [f"a={r['alpha']}\n{p}" for r in results for p in ("frozen", "fine-tuned")]
    accs = [r[p]["test_accuracy"] for r in results for p in ("feature_extraction", "fine_tuned")]
    ax[1].bar(names, accs, color=["#8fb3d9", "#2f6db5", "#f2b880", "#d9772b"])
    for i, a in enumerate(accs):
        ax[1].text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=9)
    ax[1].set_ylim(min(accs) - 0.05, 1.0); ax[1].set_title("Exp 6 - test accuracy")
    fig.tight_layout(); fig.savefig(PLOTS / "exp6_transfer.png"); plt.close(fig)
    best = max(results, key=lambda r: r["fine_tuned"]["test_accuracy"])
    plot_confusion(best["fine_tuned"]["confusion_matrix"], EUROSAT_CLASSES,
                   f"Exp 6 - MobileNetV2 a={best['alpha']} fine-tuned (test)", PLOTS / "exp6_confusion.png")
    for r in results:
        print({k: r[k] for k in ("alpha", "total_params", "model_size_kb", "cpu_latency_ms_per_image")},
              r["feature_extraction"]["test_accuracy"], r["fine_tuned"]["test_accuracy"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
