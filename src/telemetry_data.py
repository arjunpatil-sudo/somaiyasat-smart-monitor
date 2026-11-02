"""Loading / splitting / scaling helpers for the telemetry experiments.

Split is by simulated run (not by row) so test data comes from orbits the
models never saw:  run % 6 == 5 -> test,  run % 6 == 4 -> validation.
"""
import numpy as np
import pandas as pd

from common import DATA, MODEL_FEATURES, save_json, WEB_DATA


def load_telemetry():
    path = DATA / "telemetry.csv"
    if not path.exists():
        from telemetry_sim import generate
        generate().round(4).to_csv(path, index=False)
    return pd.read_csv(path)


def split_runs(df):
    test = df[df.run % 6 == 5]
    val = df[df.run % 6 == 4]
    train = df[df.run % 6 < 4]
    return train, val, test


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def to_dict(self, names):
        return {"features": names, "mean": self.mean.tolist(), "std": self.std.tolist()}


def classification_data():
    df = load_telemetry()
    tr, va, te = split_runs(df)
    X = [d[MODEL_FEATURES].to_numpy(np.float32) for d in (tr, va, te)]
    y = [d["health"].to_numpy(np.int64) for d in (tr, va, te)]
    sc = Standardizer().fit(X[0])
    X = [sc.transform(x).astype(np.float32) for x in X]
    save_json(sc.to_dict(MODEL_FEATURES), WEB_DATA / "health_scaler.json")
    return X, y, sc


def class_weights(y):
    counts = np.bincount(y)
    w = len(y) / (len(counts) * counts)
    return {i: float(v) for i, v in enumerate(w)}
