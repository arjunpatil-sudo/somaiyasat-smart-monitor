"""
Export trained models + results to the static web dashboard (web/).

* CNN (Exp 5): BatchNorm folded into conv weights -> web/data/cnn_manifest.json + cnn_weights.bin
* Test-image gallery sprite (3 per class)          -> web/img/gallery.png + web/data/gallery.json
* All metric files summarised                     -> web/data/summary.json
* Plots copied                                    -> web/img/
(health_dnn.json, lstm_forecaster.json, replay_orbit.json are written by Exp 3 / Exp 8.)
"""
import shutil

import keras
import numpy as np
from PIL import Image

from common import EUROSAT_CLASSES, METRICS, MODELS, PLOTS, WEB_DATA, WEB_IMG, load_json, save_json
from eurosat_data import eurosat_splits
from exp5_cnn import RGB_MEAN, RGB_STD


def export_cnn():
    model = keras.models.load_model(MODELS / "exp5_cnn.keras")
    blobs, layers, off = [], [], 0

    def add(arr):
        nonlocal off
        a = np.asarray(arr, np.float32).ravel()
        blobs.append(a); start = off; off += a.size
        return start, a.size

    for l in model.layers:
        n = l.name
        if n.endswith("_conv"):
            W = l.get_weights()[0]
            bn = model.get_layer(n.replace("_conv", "_bn"))
            g, beta, mean, var = bn.get_weights()
            s = g / np.sqrt(var + bn.epsilon)
            Wf, bf = W * s, beta - mean * s                   # fold BN into conv
            wo, ws = add(Wf); bo, bs = add(bf)
            layers.append({"type": "conv", "name": n.replace("_conv", ""), "cout": int(W.shape[-1]),
                           "w_offset": wo, "w_size": ws, "b_offset": bo, "b_size": bs})
        elif n.endswith("_pool"):
            layers.append({"type": "pool"})
        elif n == "gap":
            layers.append({"type": "gap"})
        elif n == "classifier":
            W, b = l.get_weights(); wo, ws = add(W); bo, bs = add(b)
            layers.append({"type": "dense", "w_offset": wo, "w_size": ws, "b_offset": bo, "b_size": bs})
    np.concatenate(blobs).astype("<f4").tofile(WEB_DATA / "cnn_weights.bin")
    save_json({"classes": EUROSAT_CLASSES, "input": [64, 64, 3], "rgb_mean": RGB_MEAN, "rgb_std": RGB_STD, "layers": layers}, WEB_DATA / "cnn_manifest.json")
    return model


def export_gallery(model):
    _, _, (Xte, yte) = eurosat_splits()
    rng = np.random.default_rng(3)
    idx = [int(i) for c in range(10) for i in rng.choice(np.where(yte == c)[0], 3, replace=False)]
    sprite = np.zeros((3 * 64, 10 * 64, 3), np.uint8)
    items = []
    probs = model.predict(Xte[idx].astype(np.float32), verbose=0)
    for k, i in enumerate(idx):
        r, c = k % 3, k // 3
        sprite[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = Xte[i]
        items.append({"x": c * 64, "y": r * 64, "label": EUROSAT_CLASSES[yte[i]],
                      "keras_probs": np.round(probs[k], 5).tolist()})
    Image.fromarray(sprite).save(WEB_IMG / "gallery.png")
    save_json({"tile": 64, "items": items}, WEB_DATA / "gallery.json")


def summary():
    m = {p.stem: load_json(p) for p in METRICS.glob("*.json")}
    strip = lambda d: {k: v for k, v in d.items() if k not in ("history", "report")}
    out = {}
    for k, v in m.items():
        if k == "exp6_transfer":
            out[k] = {"experiment": v["experiment"], "runs": [
                {**{kk: vv for kk, vv in r.items() if kk not in ("feature_extraction", "fine_tuned")},
                 "feature_extraction": strip(r["feature_extraction"]), "fine_tuned": strip(r["fine_tuned"])}
                for r in v["runs"]]}
        else:
            out[k] = strip(v)
    save_json(out, WEB_DATA / "summary.json")


def copy_plots():
    for p in PLOTS.glob("*.png"):
        shutil.copy(p, WEB_IMG / p.name)


if __name__ == "__main__":
    model = export_cnn()
    export_gallery(model)
    summary()
    copy_plots()
    print("web assets exported")
