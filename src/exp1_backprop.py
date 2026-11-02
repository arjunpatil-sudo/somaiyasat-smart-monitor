"""
Experiment 1 - Back-propagation algorithm (from scratch, NumPy only)

A 7-16-3 multilayer perceptron (tanh hidden layer, softmax output) is trained
on SomaiyaSat telemetry to classify spacecraft health. Every gradient is
derived and coded by hand (chain rule), then verified against a numerical
(finite-difference) gradient before training.

Outputs: results/metrics/exp1_backprop.json, results/plots/exp1_*.png
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from common import (HEALTH_CLASSES, METRICS, PLOTS, SEED, plot_confusion,
                    plot_style, save_json, set_seed)
from telemetry_data import classification_data, class_weights


class MLP:
    def __init__(self, n_in, n_hidden, n_out, rng):
        # Xavier/Glorot initialisation
        self.W1 = rng.normal(0, np.sqrt(2 / (n_in + n_hidden)), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / (n_hidden + n_out)), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    # ---------- forward pass ----------
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1          # (N, H)
        self.a1 = np.tanh(self.z1)               # hidden activation
        self.z2 = self.a1 @ self.W2 + self.b2    # (N, C) logits
        z = self.z2 - self.z2.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.p = e / e.sum(axis=1, keepdims=True)   # softmax
        return self.p

    def loss(self, y, sw=None):
        n = len(y)
        sw = np.ones(n) if sw is None else sw
        return float(-(sw * np.log(self.p[np.arange(n), y] + 1e-12)).sum() / sw.sum())

    # ---------- backward pass (chain rule) ----------
    def backward(self, y, sw=None):
        n = len(y)
        sw = np.ones(n) if sw is None else sw
        Y = np.zeros_like(self.p); Y[np.arange(n), y] = 1
        dz2 = (self.p - Y) * sw[:, None] / sw.sum()   # dL/dz2  (softmax + CE)
        dW2 = self.a1.T @ dz2                          # dL/dW2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T                          # back through W2
        dz1 = da1 * (1 - self.a1 ** 2)                 # tanh'(z) = 1 - tanh^2
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2]


def gradient_check(model, X, y, eps=1e-5, n_checks=25, rng=None):
    """Compare analytic gradients with central finite differences."""
    model.forward(X); grads = model.backward(y)
    rel_errors = []
    for P, G in zip(model.params(), grads):
        for _ in range(n_checks // 4 + 1):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps; model.forward(X); lp = model.loss(y)
            P[idx] = old - eps; model.forward(X); lm = model.loss(y)
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            rel_errors.append(abs(num - G[idx]) / max(1e-12, abs(num) + abs(G[idx])))
    return float(np.max(rel_errors)), float(np.mean(rel_errors))


def main():
    set_seed()
    rng = np.random.default_rng(SEED)
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = classification_data()
    cw = class_weights(ytr)
    sw_all = np.array([cw[c] for c in ytr])

    model = MLP(Xtr.shape[1], 16, 3, rng)
    max_err, mean_err = gradient_check(model, Xtr[:64], ytr[:64], rng=rng)
    print(f"gradient check: max rel err {max_err:.2e}, mean {mean_err:.2e}")

    lr, momentum, epochs, bs = 0.01, 0.9, 40, 128
    vel = [np.zeros_like(p) for p in model.params()]
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            model.forward(Xtr[b])
            grads = model.backward(ytr[b], sw_all[b])
            for p, g, v in zip(model.params(), grads, vel):
                v *= momentum; v -= lr * g; p += v       # SGD + momentum update
        model.forward(Xtr); hist["loss"].append(model.loss(ytr, sw_all))
        hist["accuracy"].append(accuracy_score(ytr, model.p.argmax(1)))
        model.forward(Xva)
        hist["val_loss"].append(model.loss(yva, np.array([cw[c] for c in yva])))
        hist["val_accuracy"].append(accuracy_score(yva, model.p.argmax(1)))
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep+1:2d}  loss {hist['loss'][-1]:.4f}  val_acc {hist['val_accuracy'][-1]:.4f}")
    train_time = time.time() - t0

    pred = model.forward(Xte).argmax(1)
    cm = confusion_matrix(yte, pred)
    res = {
        "experiment": "Exp 1 - Backpropagation from scratch (NumPy)",
        "architecture": "7-16-3 MLP, tanh hidden, softmax output",
        "optimizer": f"mini-batch SGD + momentum (lr={lr}, m={momentum}, batch={bs})",
        "epochs": epochs,
        "gradient_check_max_rel_error": max_err,
        "gradient_check_mean_rel_error": mean_err,
        "params": int(sum(p.size for p in model.params())),
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "train_time_s": round(train_time, 1),
        "confusion_matrix": cm.tolist(),
        "history": hist,
    }
    save_json(res, METRICS / "exp1_backprop.json")
    np.savez(METRICS / "exp1_weights.npz", *model.params())

    plt = plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="validation")
    ax[0].set_title("Exp 1 - weighted cross-entropy loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="validation")
    ax[1].set_title("Exp 1 - accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(PLOTS / "exp1_training.png"); plt.close(fig)
    plot_confusion(cm, HEALTH_CLASSES, "Exp 1 - NumPy MLP (test)", PLOTS / "exp1_confusion.png")
    print({k: v for k, v in res.items() if k not in ("history",)})


if __name__ == "__main__":
    main()
