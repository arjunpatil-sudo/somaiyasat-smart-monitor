"""Shared training / evaluation for the recurrent forecasters (Exp 7 RNN, Exp 8 LSTM)."""
import time

import keras
import numpy as np
from keras import layers

from common import METRICS, MODELS, PLOTS, model_size_kb, plot_style, save_json, set_seed
from forecast_data import HORIZON, LOOKBACK, TARGETS, forecast_data, to_physical


def build(kind, units=64):
    rnn = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM}[kind]
    return keras.Sequential([
        keras.Input((LOOKBACK, 7)),
        rnn(units, name=f"{kind}_layer"),
        layers.Dense(HORIZON * len(TARGETS), name="forecast"),
        layers.Reshape((HORIZON, len(TARGETS))),
    ], name=f"{kind}_forecaster")


def metrics(Yt, Yp, sc):
    t, p = to_physical(Yt, sc), to_physical(Yp, sc)
    err = np.abs(t - p)
    out = {}
    for j, name in enumerate(TARGETS):
        out[name] = {"mae": round(float(err[..., j].mean()), 4),
                     "rmse": round(float(np.sqrt(((t - p)[..., j] ** 2).mean())), 4),
                     "mae_t+1": round(float(err[:, 0, j].mean()), 4),
                     "mae_t+15": round(float(err[:, -1, j].mean()), 4),
                     "mae_by_horizon": [round(float(v), 4) for v in err[..., j].mean(0)]}
    return out


def persistence(X):
    """Baseline: future = last observed value."""
    last = X[:, -1, [0, 2]]                       # battery_voltage, battery_temp columns
    return np.repeat(last[:, None, :], HORIZON, axis=1)


def train_and_eval(kind, title, epochs=30):
    set_seed()
    ((Xtr, Ytr), (Xva, Yva), (Xte, Yte)), sc, test_df = forecast_data()
    model = build(kind)
    model.compile(keras.optimizers.Adam(2e-3), loss="mse", metrics=["mae"])
    model.summary()
    cbs = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)]
    t0 = time.time()
    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs, batch_size=128, callbacks=cbs, verbose=2)
    train_time = time.time() - t0
    Yp = model.predict(Xte, verbose=0, batch_size=1024)
    path = MODELS / f"{'exp7' if kind == 'rnn' else 'exp8'}_{kind}.keras"; model.save(path)
    res = {"experiment": title, "model": kind.upper(), "lookback_min": LOOKBACK, "horizon_min": HORIZON,
           "targets": TARGETS, "params": model.count_params(), "epochs_run": len(hist.history["loss"]),
           "train_time_s": round(train_time, 1), "model_size_kb": model_size_kb(path),
           "windows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
           "test": metrics(Yte, Yp, sc), "persistence_baseline": metrics(Yte, persistence(Xte), sc),
           "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()}}
    save_json(res, METRICS / f"{'exp7' if kind == 'rnn' else 'exp8'}_{kind}.json")
    return model, res, sc, test_df, hist
