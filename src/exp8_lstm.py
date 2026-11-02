"""
Experiment 8 - Long Short-Term Memory network

Same forecasting task as Exp 7 with an LSTM layer of the same width, then a
side-by-side comparison of RNN vs LSTM vs persistence, error-vs-horizon curves
and an example forecast through an eclipse. Exports the LSTM weights and a
test-orbit replay for the in-browser forecaster on the dashboard.
Run exp7_rnn.py first (for the comparison plot).
"""
import keras
import numpy as np

from common import METRICS, MODEL_FEATURES, MODELS, PLOTS, WEB_DATA, load_json, plot_history, plot_style, save_json
from forecast_data import HORIZON, LOOKBACK, TARGETS, to_physical
from forecasting import train_and_eval


def export_lstm(model, sc, test_df):
    lstm = model.get_layer("lstm_layer"); dense = model.get_layer("forecast")
    k, rk, b = lstm.get_weights(); W, bd = dense.get_weights()
    r = lambda a: np.round(a, 6).tolist()
    # one test run with a fault for the dashboard replay
    run_id = test_df[test_df.run_fault == "battery_ageing"].run.iloc[0]
    run = test_df[test_df.run == run_id].iloc[300:1500]
    save_json({"lookback": LOOKBACK, "horizon": HORIZON, "targets": TARGETS, "features": MODEL_FEATURES,
               "units": int(k.shape[1] // 4), "kernel": r(k), "recurrent_kernel": r(rk), "bias": r(b),
               "dense_W": r(W), "dense_b": r(bd)}, WEB_DATA / "lstm_forecaster.json")
    save_json({"run": int(run_id), "fault": "battery_ageing", "minute": run.minute.tolist(),
               "rows": np.round(run[MODEL_FEATURES].to_numpy(), 4).tolist(), "health": run.health.tolist()},
              WEB_DATA / "replay_orbit.json")


def main():
    model, res, sc, test_df, hist = train_and_eval("lstm", "Exp 8 - LSTM forecaster")
    plot_history(hist, "Exp 8 LSTM", PLOTS / "exp8_training.png", metric="mae")
    export_lstm(model, sc, test_df)

    plt = plot_style()
    rnn_path = METRICS / "exp7_rnn.json"
    rnn = load_json(rnn_path) if rnn_path.exists() else None
    # ---- error vs horizon ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    units = {"battery_voltage": "V", "battery_temp": "degC"}
    for j, t in enumerate(TARGETS):
        h = range(1, HORIZON + 1)
        ax[j].plot(h, res["persistence_baseline"][t]["mae_by_horizon"], "k:", label="persistence")
        if rnn:
            ax[j].plot(h, rnn["test"][t]["mae_by_horizon"], "o-", ms=3, label="SimpleRNN")
        ax[j].plot(h, res["test"][t]["mae_by_horizon"], "s-", ms=3, label="LSTM")
        ax[j].set_title(f"{t} - MAE vs horizon"); ax[j].set_xlabel("minutes ahead"); ax[j].set_ylabel(units[t]); ax[j].legend()
    fig.tight_layout(); fig.savefig(PLOTS / "exp8_horizon_comparison.png"); plt.close(fig)

    # ---- rolling 15-min-ahead forecast through a test orbit ----
    run_id = test_df[test_df.run_fault == "battery_ageing"].run.iloc[0]
    run = test_df[test_df.run == run_id]
    a = sc.transform(run[MODEL_FEATURES].to_numpy(np.float32))
    starts = np.arange(600, 1100, 1)
    Xw = np.stack([a[s - LOOKBACK:s] for s in starts])
    pl = to_physical(model.predict(Xw, verbose=0, batch_size=512), sc)
    models = [("LSTM", pl)]
    rpath = MODELS / "exp7_rnn.keras"
    if rpath.exists():
        models.insert(0, ("SimpleRNN", to_physical(keras.models.load_model(rpath).predict(Xw, verbose=0, batch_size=512), sc)))
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    mins = run.minute.to_numpy()
    for j, t in enumerate(TARGETS):
        seg = slice(600, 1100 + HORIZON)
        ax[j].plot(mins[seg], run[t].to_numpy()[seg], c="k", lw=1, label="actual")
        for name, P in models:
            ax[j].plot(mins[starts + HORIZON - 1], P[:, -1, j], lw=1.2, label=f"{name} (15 min ahead)")
        sun = run.sunlit.to_numpy()[seg]
        ax[j].fill_between(mins[seg], *ax[j].get_ylim(), where=sun == 0, color="grey", alpha=0.15, label="eclipse")
        ax[j].set_ylabel(units[t]); ax[j].legend(loc="upper right", fontsize=8, ncol=4)
    ax[0].set_title(f"Exp 8 - 15-minute-ahead forecasts on unseen test run {run_id} (battery-ageing fault)")
    ax[1].set_xlabel("mission minute")
    fig.tight_layout(); fig.savefig(PLOTS / "exp8_forecast_example.png"); plt.close(fig)

    if rnn:
        save_json({"rnn": {t: rnn["test"][t] for t in TARGETS}, "lstm": {t: res["test"][t] for t in TARGETS},
                   "persistence": {t: res["persistence_baseline"][t] for t in TARGETS},
                   "params": {"rnn": rnn["params"], "lstm": res["params"]}}, METRICS / "exp7_8_comparison.json")
    for t, m in res["test"].items():
        print(f"{t}: LSTM MAE {m['mae']} (t+15 {m['mae_t+15']}) | persistence {res['persistence_baseline'][t]['mae']}"
              + (f" | RNN {rnn['test'][t]['mae']}" if rnn else ""))


if __name__ == "__main__":
    main()
