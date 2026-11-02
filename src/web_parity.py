"""Checks that the JavaScript inference in web/js/nn.js matches Keras (CNN, DNN, LSTM)."""
import json
import subprocess

import keras
import numpy as np

from common import METRICS, MODELS, MODEL_FEATURES, ROOT, WEB_DATA, save_json
from eurosat_data import eurosat_splits
from forecast_data import forecast_data
from telemetry_data import classification_data

N = 40


def main():
    _, _, (Xte, yte) = eurosat_splits()
    imgs = Xte[:N]
    cnn = keras.models.load_model(MODELS / "exp5_cnn.keras")
    k_cnn = cnn.predict(imgs.astype(np.float32), verbose=0)

    (_, _, Xh), _, sc = classification_data()
    raw_h = Xh[:200] * sc.std + sc.mean                       # back to physical units, as the web app uses
    k_dnn = keras.models.load_model(MODELS / "exp3_dnn.keras").predict(Xh[:200], verbose=0)

    (_, _, (Xf, _)), fsc, _ = forecast_data()
    raw_f = Xf[:20] * fsc.std + fsc.mean
    k_lstm = keras.models.load_model(MODELS / "exp8_lstm.keras").predict(Xf[:20], verbose=0).reshape(20, -1)

    rgba = np.concatenate([imgs, np.full(imgs.shape[:3] + (1,), 255, np.uint8)], -1).reshape(N, -1)
    case = {"rgba": rgba.tolist(), "health": raw_h.tolist(), "seqs": raw_f.tolist()}
    (ROOT / "web" / "data" / "_parity_case.json").write_text(json.dumps(case))
    js = r"""
    globalThis.window = globalThis; globalThis.fetch = async (u) => { const fs = require('fs'); const p = require('path').join(process.argv[1], u);
      return { json: async () => JSON.parse(fs.readFileSync(p)), arrayBuffer: async () => { const b = fs.readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); } }; };
    require(process.argv[1] + '/js/nn.js');
    (async () => {
      const fs = require('fs'), d = p => JSON.parse(fs.readFileSync(process.argv[1] + '/data/' + p));
      const c = d('_parity_case.json'), net = await NN.loadCNN('data/cnn_manifest.json', 'data/cnn_weights.bin');
      const dnn = d('health_dnn.json'), hs = d('health_scaler.json'), lstm = d('lstm_forecaster.json'), fs2 = d('forecast_scaler.json');
      const st = (v, s) => v.map((x, i) => (x - s.mean[i]) / s.std[i]);
      const out = { cnn: c.rgba.map(r => NN.runCNN(net, r).probs), dnn: c.health.map(r => NN.runDense(dnn.layers, st(r, hs))),
                    lstm: c.seqs.map(s => NN.runLSTM(lstm, s.map(r => st(r, fs2)))) };
      console.log(JSON.stringify(out));
    })();"""
    res = subprocess.run(["node", "-e", js, "--", str(ROOT / "web")], capture_output=True, text=True, check=True)
    j = json.loads(res.stdout.strip().splitlines()[-1])
    (ROOT / "web" / "data" / "_parity_case.json").unlink()
    out = {"cnn_n": N, "cnn_max_abs_diff": float(np.abs(np.array(j["cnn"]) - k_cnn).max()),
           "cnn_top1_agree": float((np.array(j["cnn"]).argmax(1) == k_cnn.argmax(1)).mean()),
           "dnn_max_abs_diff": float(np.abs(np.array(j["dnn"]) - k_dnn).max()),
           "lstm_max_abs_diff": float(np.abs(np.array(j["lstm"]) - k_lstm).max())}
    save_json(out, METRICS / "web_parity.json")
    print(out)


if __name__ == "__main__":
    main()
