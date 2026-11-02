"""Shared paths, seeds and plotting helpers for the SomaiyaSat Smart Monitor project."""
import json
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
METRICS = RESULTS / "metrics"
MODELS = ROOT / "models"
WEB_DATA = ROOT / "web" / "data"
WEB_IMG = ROOT / "web" / "img"

for p in (DATA, PLOTS, METRICS, MODELS, WEB_DATA, WEB_IMG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
HEALTH_CLASSES = ["Normal", "Warning", "Critical"]
TELEMETRY_FEATURES = [
    "battery_voltage",   # V
    "battery_current",   # mA, + = charging
    "battery_temp",      # deg C
    "board_temp",        # deg C
    "solar_current",     # mA
    "load_current",      # mA
]
# inputs to the health classifier / forecasters (6 sensors + sun-sensor flag)
MODEL_FEATURES = TELEMETRY_FEATURES + ["sunlit"]
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
    })
    return plt


def plot_history(history, title, path, metric="accuracy"):
    """Loss + metric curves for a Keras History (or dict)."""
    plt = plot_style()
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h["loss"], label="train")
    if "val_loss" in h:
        ax[0].plot(h["val_loss"], label="validation")
    ax[0].set_title(f"{title} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    if metric in h:
        ax[1].plot(h[metric], label="train")
        if f"val_{metric}" in h:
            ax[1].plot(h[f"val_{metric}"], label="validation")
        ax[1].set_title(f"{title} - {metric}"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_confusion(cm, labels, title, path, normalize=True):
    plt = plot_style()
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2.6, 0.55 * n + 2.2))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = shown[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else int(v), ha="center", va="center",
                    color="white" if v > 0.6 * shown.max() else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def model_size_kb(path):
    return round(Path(path).stat().st_size / 1024, 1)
