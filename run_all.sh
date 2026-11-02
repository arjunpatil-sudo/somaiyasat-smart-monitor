#!/usr/bin/env bash
# Re-runs every experiment in order and refreshes the dashboard assets.
# Needs data/EuroSAT_RGB.zip (https://zenodo.org/records/7711810) for Exp 4-6.
set -e
cd "$(dirname "$0")/src"
python telemetry_sim.py      # synthetic SomaiyaSat telemetry -> data/telemetry.csv
python exp1_backprop.py      # Exp 1  backprop from scratch (NumPy)
python exp2_ffnn.py          # Exp 2  feed-forward NN
python exp3_regularisation_study.py  # Exp 3  dropout / BN / class-weight study
python exp3_dnn.py           # Exp 3  deep NN
python exp5_cnn.py           # Exp 5  CNN on EuroSAT   (run before Exp 4)
python exp4_feature_maps.py  # Exp 4  feature maps + Grad-CAM
python exp6_transfer.py      # Exp 6  MobileNetV2 transfer learning
python exp7_rnn.py           # Exp 7  SimpleRNN forecaster
python exp8_lstm.py          # Exp 8  LSTM forecaster + comparison
python plot_telemetry.py     # dataset figures for the report
python export_web.py         # weights + results -> web/
python web_parity.py         # JavaScript vs Keras check
node ../report/build_report.js  # rebuild the report (DOCX)
echo "done - open web/index.html via a local server: python -m http.server -d web 8000"
