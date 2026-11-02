"""
Experiment 4 - Visualisation of feature maps in a CNN

Uses the CNN trained in Exp 5 (run exp5_cnn.py first) and shows what it learns:
  1. first-layer 3x3 RGB filters
  2. activation (feature) maps at increasing depth for sample Earth images
  3. Grad-CAM heat-maps: which pixels drove each class decision

Outputs: results/plots/exp4_*.png  (+ copies for the web dashboard)
"""
import shutil

import keras
import numpy as np
import tensorflow as tf

from common import EUROSAT_CLASSES, METRICS, MODELS, PLOTS, WEB_IMG, plot_style, save_json, set_seed
from eurosat_data import eurosat_splits

LAYERS = ["b1c1_relu", "b1c2_relu", "b2c2_relu", "b3c1_relu"]


def pick_samples(X, y, classes):
    rng = np.random.default_rng(7)
    return [int(rng.choice(np.where(y == EUROSAT_CLASSES.index(c))[0])) for c in classes]


def plot_filters(model, path):
    plt = plot_style()
    W = model.get_layer("b1c1_conv").get_weights()[0]          # (3,3,3,32)
    fig, axes = plt.subplots(4, 8, figsize=(8, 4.3))
    for k, ax in enumerate(axes.flat):
        f = W[..., k]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f, interpolation="nearest"); ax.axis("off")
    fig.suptitle("Exp 4 - the 32 learned first-layer filters (3x3 RGB)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_feature_maps(model, img, cname, path, n_ch=8):
    plt = plot_style()
    probe = keras.Model(model.input, [model.get_layer(n).output for n in LAYERS])
    acts = probe.predict(img[None].astype(np.float32), verbose=0)
    fig, axes = plt.subplots(len(LAYERS), n_ch + 1, figsize=(1.35 * (n_ch + 1), 1.5 * len(LAYERS)))
    for r, (name, a) in enumerate(zip(LAYERS, acts)):
        a = a[0]
        top = np.argsort(a.mean(axis=(0, 1)))[::-1][:n_ch]      # most active channels
        axes[r, 0].imshow(img); axes[r, 0].set_title("input" if r == 0 else "", fontsize=8)
        axes[r, 0].set_ylabel(f"{name.split('_')[0]}\n{a.shape[0]}x{a.shape[1]}x{a.shape[2]}", fontsize=8)
        for c, ch in enumerate(top):
            axes[r, c + 1].imshow(a[..., ch], cmap="viridis")
            axes[r, c + 1].set_title(f"ch {ch}", fontsize=7)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"Exp 4 - feature maps for a '{cname}' image (8 most active channels per layer)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def grad_cam(model, img, layer="b3c1_relu"):
    grad_model = keras.Model(model.input, [model.get_layer(layer).output, model.output])
    x = tf.convert_to_tensor(img[None].astype(np.float32))
    with tf.GradientTape() as tape:
        conv, pred = grad_model(x)
        cls = tf.argmax(pred[0])
        score = pred[:, cls]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], (64, 64)).numpy()[..., 0]
    return cam, int(cls), float(pred[0, cls])


def main():
    set_seed()
    model = keras.models.load_model(MODELS / "exp5_cnn.keras")
    _, _, (Xte, yte) = eurosat_splits()

    plot_filters(model, PLOTS / "exp4_filters.png")
    for cname in ("Forest", "River", "Residential"):
        i = pick_samples(Xte, yte, [cname])[0]
        plot_feature_maps(model, Xte[i], cname, PLOTS / f"exp4_feature_maps_{cname.lower()}.png")

    plt = plot_style()
    idx = pick_samples(Xte, yte, EUROSAT_CLASSES)
    fig, axes = plt.subplots(2, 10, figsize=(16, 4.9), gridspec_kw={"hspace": 0.55})
    cams = []
    for k, i in enumerate(idx):
        cam, cls, p = grad_cam(model, Xte[i])
        axes[0, k].imshow(Xte[i]); axes[0, k].set_title(EUROSAT_CLASSES[yte[i]], fontsize=8)
        axes[1, k].imshow(Xte[i]); axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, k].set_title(f"pred: {EUROSAT_CLASSES[cls]}\np = {p:.2f}", fontsize=7.5,
                             color="green" if cls == yte[i] else "red")
        cams.append({"true": EUROSAT_CLASSES[yte[i]], "pred": EUROSAT_CLASSES[cls], "prob": round(p, 3)})
        for ax in axes[:, k]:
            ax.axis("off")
    fig.suptitle("Exp 4 - Grad-CAM on the last conv layer: where the CNN looks (top: input + true class, bottom: heat-map + prediction)")
    fig.savefig(PLOTS / "exp4_gradcam.png"); plt.close(fig)

    # activation statistics per layer (sparsity grows with depth)
    probe = keras.Model(model.input, [model.get_layer(n).output for n in LAYERS])
    acts = probe.predict(Xte[:256].astype(np.float32), verbose=0)
    stats = [{"layer": n, "shape": list(a.shape[1:]), "mean_activation": round(float(a.mean()), 4),
              "fraction_zero": round(float((a <= 0).mean()), 4)} for n, a in zip(LAYERS, acts)]
    save_json({"experiment": "Exp 4 - Feature map visualisation", "layer_stats": stats, "gradcam_samples": cams},
              METRICS / "exp4_feature_maps.json")
    for f in ("exp4_filters.png", "exp4_gradcam.png", "exp4_feature_maps_forest.png",
              "exp4_feature_maps_river.png", "exp4_feature_maps_residential.png"):
        shutil.copy(PLOTS / f, WEB_IMG / f)
    print(stats)


if __name__ == "__main__":
    main()
