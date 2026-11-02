"""Dataset figures for the report: sample telemetry and class distribution."""
import numpy as np

from common import HEALTH_CLASSES, PLOTS, TELEMETRY_FEATURES, EUROSAT_CLASSES, plot_style
from telemetry_data import load_telemetry, split_runs

UNITS = {"battery_voltage": "V", "battery_current": "mA", "battery_temp": "degC",
         "board_temp": "degC", "solar_current": "mA", "load_current": "mA"}


def main():
    plt = plot_style()
    df = load_telemetry()
    run = df[df.run == df[df.run_fault == "heater_failure"].run.iloc[0]].iloc[:600]
    fig, axes = plt.subplots(len(TELEMETRY_FEATURES), 1, figsize=(10, 9), sharex=True)
    for ax, f in zip(axes, TELEMETRY_FEATURES):
        ax.plot(run.minute, run[f], lw=0.9, c="#2f6db5")
        lo, hi = ax.get_ylim()
        ax.fill_between(run.minute, lo, hi, where=run.sunlit == 0, color="grey", alpha=0.18, lw=0)
        ax.set_ylim(lo, hi); ax.set_ylabel(f"{f}\n({UNITS[f]})", fontsize=8)
    axes[0].set_title("Simulated SomaiyaSat housekeeping telemetry - first 10 h of a run (grey = eclipse)")
    axes[-1].set_xlabel("mission minute")
    fig.tight_layout(); fig.savefig(PLOTS / "data_telemetry_sample.png"); plt.close(fig)

    tr, va, te = split_runs(df)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    w = 0.27
    for k, (name, d) in enumerate([("train", tr), ("validation", va), ("test", te)]):
        c = np.bincount(d.health, minlength=3)
        ax[0].bar(np.arange(3) + (k - 1) * w, c, w, label=f"{name} ({len(d):,})")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(HEALTH_CLASSES); ax[0].set_yscale("log")
    ax[0].set_title("Telemetry health labels per split"); ax[0].legend(fontsize=8)
    fc = df.groupby("run_fault").health.value_counts(normalize=True).unstack().fillna(0)
    fc.columns = HEALTH_CLASSES
    fc.plot.barh(stacked=True, ax=ax[1], color=["#3ccf8e", "#ffc247", "#ff5c6c"], width=0.7)
    ax[1].set_title("Label mix by injected fault type"); ax[1].set_xlabel("fraction of minutes"); ax[1].set_ylabel("")
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(PLOTS / "data_telemetry_labels.png"); plt.close(fig)

    from eurosat_data import eurosat_splits
    (X, y), _, _ = eurosat_splits()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 10, figsize=(13, 4.3))
    for c in range(10):
        for r, i in enumerate(rng.choice(np.where(y == c)[0], 3, replace=False)):
            axes[r, c].imshow(X[i]); axes[r, c].axis("off")
        axes[0, c].set_title(EUROSAT_CLASSES[c], fontsize=8)
    fig.suptitle("EuroSAT RGB samples (64x64 Sentinel-2 patches)")
    fig.tight_layout(); fig.savefig(PLOTS / "data_eurosat_samples.png"); plt.close(fig)


if __name__ == "__main__":
    main()
