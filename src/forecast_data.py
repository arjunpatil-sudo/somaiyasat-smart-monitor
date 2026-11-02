"""Sliding-window datasets for battery-voltage / temperature forecasting (Exp 7 & 8).

Input : last LOOKBACK minutes of all 7 telemetry channels (standardised)
Output: next HORIZON minutes of battery_voltage and battery_temp (standardised)
Windows never cross run boundaries; split by run as in telemetry_data.py.
"""
import numpy as np

from common import MODEL_FEATURES, WEB_DATA, save_json
from telemetry_data import Standardizer, load_telemetry, split_runs

LOOKBACK = 60
HORIZON = 15
TARGETS = ["battery_voltage", "battery_temp"]
STRIDE = 3


def make_windows(df, sc, stride=STRIDE):
    X, Y = [], []
    tgt_idx = [MODEL_FEATURES.index(t) for t in TARGETS]
    for _, run in df.groupby("run"):
        a = sc.transform(run[MODEL_FEATURES].to_numpy(np.float32))
        for s in range(0, len(a) - LOOKBACK - HORIZON + 1, stride):
            X.append(a[s:s + LOOKBACK])
            Y.append(a[s + LOOKBACK:s + LOOKBACK + HORIZON][:, tgt_idx])   # (H, 2)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def forecast_data():
    df = load_telemetry()
    tr, va, te = split_runs(df)
    sc = Standardizer().fit(tr[MODEL_FEATURES].to_numpy(np.float32))
    save_json(sc.to_dict(MODEL_FEATURES), WEB_DATA / "forecast_scaler.json")
    return [make_windows(d, sc) for d in (tr, va, te)], sc, te


def to_physical(Y, sc):
    """(N, H, 2) standardised -> physical units (V, degC)."""
    idx = [MODEL_FEATURES.index(t) for t in TARGETS]
    return Y * sc.std[idx] + sc.mean[idx]
