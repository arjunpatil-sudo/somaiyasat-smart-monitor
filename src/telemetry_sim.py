"""
SomaiyaSat housekeeping-telemetry simulator.

Real SomaiyaSat downlink logs are not public, so this module generates
physically-motivated telemetry for a 1S Li-ion PocketQube in a ~500 km LEO:

* ~94.6 min orbit with ~35 min eclipse per orbit
* body-mounted solar panels on a slowly tumbling satellite
* battery state of charge (SOC) integrated from (solar - load) current
* terminal voltage = open-circuit voltage(SOC) + I * R_internal
* first-order thermal models for battery and OBC board, with a survival heater
* scheduled payload activity: camera captures and SSTV transmissions

Fault episodes are injected so the health classifier and the forecasters
see abnormal behaviour: battery ageing, heater failure, solar-panel/string
loss, latch-up over-current, and a stuck-on transmitter.

Health label (Normal / Warning / Critical) is computed from the TRUE state
using operator thresholds; sensor noise is then added to the readings, so
the classes overlap near the thresholds (as on a real satellite).

Run:  python src/telemetry_sim.py      -> data/telemetry.csv
"""
import numpy as np
import pandas as pd

from common import DATA, SEED, TELEMETRY_FEATURES

ORBIT_MIN = 94.6
ECLIPSE_FRAC = 0.37
DT_MIN = 1.0                      # one sample per minute

FAULTS = ["none", "battery_ageing", "heater_failure", "solar_loss",
          "latch_up", "stuck_tx"]


def ocv(soc):
    """Open-circuit voltage of a 1S Li-ion cell vs state of charge."""
    soc = np.clip(soc, 0.0, 1.0)
    return 3.0 + 0.75 * soc + 0.45 * soc ** 2 - 0.35 * np.exp(-12 * soc) + 0.02


def health_label(v, t_bat, t_board, load, solar, sunlit):
    """Operator rule-set applied to the true (noise-free) state."""
    critical = (v < 3.40) | (t_bat > 52) | (t_bat < -6) | (t_board > 70) | (load > 430)
    warning = ((v < 3.62) | (t_bat > 40) | (t_bat < 2) | (t_board > 58) | (load > 300)
               | (sunlit & (solar < 35)))
    return np.where(critical, 2, np.where(warning, 1, 0))


def simulate_run(minutes, rng, fault="none"):
    t = np.arange(minutes) * DT_MIN
    phase = (t % ORBIT_MIN) / ORBIT_MIN
    phase0 = rng.uniform(0, 1)
    phase = (phase + phase0) % 1.0
    sunlit = phase < (1 - ECLIPSE_FRAC)

    capacity = rng.uniform(900, 1100)          # mAh
    r_int = rng.uniform(0.10, 0.16)            # ohm
    panel_eff = rng.uniform(0.9, 1.1)
    tumble = 2 * np.pi / rng.uniform(5, 11)    # rad / min

    # fault window
    f_start = int(rng.uniform(0.25, 0.55) * minutes)
    f_len = int(rng.uniform(0.15, 0.35) * minutes) if fault in ("battery_ageing", "solar_loss", "heater_failure") \
        else int(rng.uniform(25, 90))
    in_fault = np.zeros(minutes, bool)
    if fault != "none":
        in_fault[f_start:f_start + f_len] = True

    soc = rng.uniform(0.55, 0.95)
    t_bat = rng.uniform(5, 20)
    t_board = t_bat + 5
    tx_on_until, cam_on_until = -1, -1
    out = {k: np.zeros(minutes) for k in TELEMETRY_FEATURES}
    true_v = np.zeros(minutes); true_tb = np.zeros(minutes); true_tbd = np.zeros(minutes)
    true_load = np.zeros(minutes); true_solar = np.zeros(minutes)

    for i in range(minutes):
        f = in_fault[i]
        # --- solar input -------------------------------------------------
        attitude = 0.55 + 0.45 * np.abs(np.sin(tumble * t[i]))
        solar = 230 * panel_eff * attitude if sunlit[i] else 0.0
        if f and fault == "solar_loss":
            solar *= 0.12
        # --- loads ---------------------------------------------------------
        load = 65 + rng.normal(0, 4)
        if rng.random() < 0.012 and cam_on_until < i:
            cam_on_until = i + 2
        if rng.random() < 0.006 and tx_on_until < i:
            tx_on_until = i + rng.integers(6, 10)          # SSTV image ~8 min
        if f and fault == "stuck_tx":
            tx_on_until = i + 1
            load += 70                                      # PA stuck at high power
        if i <= cam_on_until:
            load += 85
        if i <= tx_on_until:
            load += 210
        heater_on = (t_bat < 3) and not (f and fault == "heater_failure")
        if heater_on:
            load += 95
        if f and fault == "latch_up":
            load += 290 + rng.normal(0, 15)
        # --- battery -------------------------------------------------------
        cap = capacity * (0.45 if (f and fault == "battery_ageing") else 1.0)
        r = r_int * (2.6 if (f and fault == "battery_ageing") else 1.0)
        i_bat = solar - load                                   # mA
        if f and fault == "battery_ageing":
            i_bat -= 115                                       # internal leakage of a degraded cell
        if soc >= 0.995 and i_bat > 0:                          # charge regulator tapers
            i_bat = 0.0
        soc = float(np.clip(soc + 0.97 * i_bat * DT_MIN / 60 / cap, 0.0, 1.0))
        v = ocv(soc) + (i_bat / 1000) * r
        # --- thermal -------------------------------------------------------
        t_env = 30 if sunlit[i] else -19
        heat_bat = 0.004 * (i_bat / 10) ** 2 / 10 + (32 if heater_on else 0)
        heat_board = 0.045 * load
        t_bat += DT_MIN / 24 * (t_env + heat_bat - t_bat)
        t_board += DT_MIN / 9 * (t_env + 8 + heat_board - t_board)
        true_v[i], true_tb[i], true_tbd[i], true_load[i], true_solar[i] = v, t_bat, t_board, load, solar
        out["battery_current"][i] = i_bat

    labels = health_label(true_v, true_tb, true_tbd, true_load, true_solar, sunlit)
    # sensor noise (ADC + thermistor)
    out["battery_voltage"] = true_v + rng.normal(0, 0.018, minutes)
    out["battery_current"] = out["battery_current"] + rng.normal(0, 6, minutes)
    out["battery_temp"] = true_tb + rng.normal(0, 0.9, minutes)
    out["board_temp"] = true_tbd + rng.normal(0, 1.1, minutes)
    out["solar_current"] = np.clip(true_solar + rng.normal(0, 7, minutes), 0, None)
    out["load_current"] = true_load + rng.normal(0, 8, minutes)

    df = pd.DataFrame(out)
    df.insert(0, "minute", t.astype(int))
    df["sunlit"] = sunlit.astype(int)
    df["fault"] = np.where(in_fault, fault, "none")
    df["health"] = labels
    return df


def generate(n_runs=48, minutes=2880, seed=SEED):
    """n_runs independent 2-day passes; ~1/4 nominal, rest with one fault each."""
    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_runs):
        fault = "none" if r % 4 == 0 else FAULTS[1 + (r % (len(FAULTS) - 1))]
        df = simulate_run(minutes, rng, fault)
        df.insert(0, "run", r)
        df["run_fault"] = fault
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    path = DATA / "telemetry.csv"
    df.round(4).to_csv(path, index=False)
    print(f"saved {len(df):,} rows -> {path}")
    print(df["health"].value_counts().sort_index().rename({0: "Normal", 1: "Warning", 2: "Critical"}))
    print(df.groupby("run_fault")["health"].value_counts(normalize=True).unstack().round(3))
    print(df[TELEMETRY_FEATURES].describe().round(2))
