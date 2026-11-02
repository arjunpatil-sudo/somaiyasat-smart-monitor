"""
Experiment 7 - Recurrent Neural Network (SimpleRNN)

Forecasts the next 15 minutes of SomaiyaSat battery voltage and battery
temperature from the previous 60 minutes of all telemetry channels.
Compared with a persistence baseline (future = last value).
"""
from common import PLOTS, plot_history
from forecasting import train_and_eval


def main():
    model, res, *_ , hist = train_and_eval("rnn", "Exp 7 - Recurrent Neural Network forecaster")
    plot_history(hist, "Exp 7 SimpleRNN", PLOTS / "exp7_training.png", metric="mae")
    for t, m in res["test"].items():
        b = res["persistence_baseline"][t]
        print(f"{t}: RNN MAE {m['mae']}  (t+1 {m['mae_t+1']}, t+15 {m['mae_t+15']})  | persistence {b['mae']}")


if __name__ == "__main__":
    main()
