// SomaiyaSat Smart Monitor dashboard - all inference runs client-side (see nn.js)
const $ = (s) => document.querySelector(s);
const getJSON = (u) => fetch(u).then((r) => r.json());
const COLORS = { Normal: "#3ccf8e", Warning: "#ffc247", Critical: "#ff5c6c" };

const FEATURES = [
  { key: "battery_voltage", label: "Battery voltage", unit: "V", min: 2.8, max: 4.3, step: 0.01 },
  { key: "battery_current", label: "Battery current", unit: "mA", min: -600, max: 250, step: 1 },
  { key: "battery_temp", label: "Battery temp", unit: "°C", min: -20, max: 60, step: 0.5 },
  { key: "board_temp", label: "OBC board temp", unit: "°C", min: -20, max: 80, step: 0.5 },
  { key: "solar_current", label: "Solar current", unit: "mA", min: 0, max: 280, step: 1 },
  { key: "load_current", label: "Load current", unit: "mA", min: 30, max: 650, step: 1 },
  { key: "sunlit", label: "Sunlit (sun sensor)", unit: "", min: 0, max: 1, step: 1 },
];
const PRESETS = {
  "Nominal · sunlit": [4.12, 120, 22, 35, 200, 70, 1],
  "Nominal · eclipse": [3.95, -70, 6, 2, 0, 70, 0],
  "SSTV downlink": [3.98, -280, 10, 15, 0, 280, 0],
  "Latch-up": [3.72, -420, 16, 48, 0, 470, 0],
  "Solar panel loss": [3.72, -45, 22, 30, 18, 68, 1],
  "Heater failure": [3.9, -70, -9, -10, 0, 65, 0],
  "Battery ageing": [3.36, -190, 6, 6, 0, 75, 0],
};

let HEALTH, HSCALE, LSTM, FSCALE, REPLAY, CNN, GALLERY, SUMMARY;

// ---------------------------------------------------------------- health
const standardize = (vals, sc) => vals.map((v, i) => (v - sc.mean[i]) / sc.std[i]);
const classifyHealth = (vals) => NN.runDense(HEALTH.layers, standardize(vals, HSCALE));

function renderBars(el, labels, probs, colorFn) {
  el.innerHTML = labels.map((l, i) => `
    <div class="bar"><span>${l}</span><div class="track"><div class="fill" style="width:${(probs[i] * 100).toFixed(1)}%;background:${colorFn(l, i)}"></div></div>
    <span class="mono">${(probs[i] * 100).toFixed(1)}%</span></div>`).join("");
}

function updateHealth() {
  const vals = FEATURES.map((f) => parseFloat($(`#s-${f.key}`).value));
  FEATURES.forEach((f, i) => ($(`#v-${f.key}`).textContent = f.key === "sunlit" ? (vals[i] ? "yes" : "no") : `${vals[i]} ${f.unit}`));
  const p = classifyHealth(vals);
  const k = p.indexOf(Math.max(...p));
  const cls = HEALTH.classes[k];
  const b = $("#healthBadge");
  b.textContent = cls; b.className = `badge b-${cls}`;
  renderBars($("#healthBars"), HEALTH.classes, p, (l) => COLORS[l]);
}

function setupHealth() {
  $("#sliders").innerHTML = FEATURES.map((f) => `
    <div class="slider"><label for="s-${f.key}">${f.label}</label>
    <input type="range" id="s-${f.key}" min="${f.min}" max="${f.max}" step="${f.step}">
    <span class="val" id="v-${f.key}"></span></div>`).join("");
  $("#presets").innerHTML = Object.keys(PRESETS).map((n) => `<button data-p="${n}">${n}</button>`).join("");
  const apply = (n) => { PRESETS[n].forEach((v, i) => ($(`#s-${FEATURES[i].key}`).value = v)); updateHealth(); };
  $("#presets").addEventListener("click", (e) => e.target.dataset.p && apply(e.target.dataset.p));
  FEATURES.forEach((f) => $(`#s-${f.key}`).addEventListener("input", updateHealth));
  apply("Nominal · sunlit");
}

// ---------------------------------------------------------------- charts
function drawChart(canvas, { actual, forecast, sunlit, t0, now, unit, title }) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight || 210;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr);
  const pad = { l: 52, r: 14, t: 24, b: 26 };
  const n = actual.length + 0;
  const all = actual.concat(forecast.map((f) => f[1]));
  let lo = Math.min(...all), hi = Math.max(...all);
  const m = (hi - lo) * 0.12 || 0.1; lo -= m; hi += m;
  const X = (i) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  g.clearRect(0, 0, W, H);
  // eclipse shading
  g.fillStyle = "#2a3456";
  for (let i = 0; i < n; i++) if (!sunlit[i]) g.fillRect(X(i) - 0.5, pad.t, (W - pad.l - pad.r) / (n - 1) + 1, H - pad.t - pad.b);
  // grid + labels
  g.strokeStyle = "#24304f"; g.fillStyle = "#93a0bd"; g.font = "11px system-ui"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo), y = Y(v);
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillText(v.toFixed(unit === "V" ? 2 : 1), 6, y + 4);
  }
  for (let i = 0; i < n; i += 30) g.fillText(`${t0 + i}`, X(i) - 8, H - 8);
  g.fillStyle = "#e7ecf7"; g.font = "600 12px system-ui"; g.fillText(title, pad.l, 15);
  // "now" line
  const nowI = now - t0;
  g.strokeStyle = "#5fb3ff"; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(nowI), pad.t); g.lineTo(X(nowI), H - pad.b); g.stroke(); g.setLineDash([]);
  // actual: past solid, future faded
  const line = (pts, color, w, alpha = 1) => {
    g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.beginPath();
    pts.forEach(([i, v], k) => (k ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)))); g.stroke(); g.globalAlpha = 1;
  };
  line(actual.slice(0, nowI + 1).map((v, i) => [i, v]), "#e7ecf7", 1.4);
  line(actual.slice(nowI).map((v, i) => [i + nowI, v]), "#e7ecf7", 1.2, 0.3);
  line(forecast, "#ffb86b", 2.4);
}

// ---------------------------------------------------------------- replay
let playing = false, timer = null;
function updateReplay() {
  const t = parseInt($("#timeSlider").value);
  const rows = REPLAY.rows, L = LSTM.lookback, Hh = LSTM.horizon;
  const seq = rows.slice(t - L, t).map((r) => standardize(r, FSCALE));
  const out = NN.runLSTM(LSTM, seq);
  const iv = FEATURES.findIndex((f) => f.key === "battery_voltage"), it = FEATURES.findIndex((f) => f.key === "battery_temp");
  const fv = [], ft = [];
  for (let h = 0; h < Hh; h++) {
    fv.push(out[h * 2] * FSCALE.std[iv] + FSCALE.mean[iv]);
    ft.push(out[h * 2 + 1] * FSCALE.std[it] + FSCALE.mean[it]);
  }
  const a = Math.max(0, t - 120), b = Math.min(rows.length, t + 30);
  const win = rows.slice(a, b), sun = win.map((r) => r[6]);
  const t0 = REPLAY.minute[a], now = REPLAY.minute[t - 1] + 1;
  const fpts = (arr) => [[t - 1 - a, rows[t - 1][arr === fv ? iv : it]]].concat(arr.map((v, h) => [t - a + h, v]));
  drawChart($("#chartV"), { actual: win.map((r) => r[iv]), forecast: fpts(fv), sunlit: sun, t0, now: now - 1, unit: "V", title: "Battery voltage (V)" });
  drawChart($("#chartT"), { actual: win.map((r) => r[it]), forecast: fpts(ft), sunlit: sun, t0, now: now - 1, unit: "C", title: "Battery temperature (°C)" });
  $("#timeLabel").textContent = `t = ${REPLAY.minute[t - 1]} min`;
  const p = classifyHealth(rows[t - 1]);
  const cls = HEALTH.classes[p.indexOf(Math.max(...p))];
  const pill = $("#replayHealth");
  pill.textContent = `DNN: ${cls} (${(Math.max(...p) * 100).toFixed(0)}%) · truth: ${HEALTH.classes[REPLAY.health[t - 1]]}`;
  pill.style.background = COLORS[cls] + "33"; pill.style.color = COLORS[cls];
}
function setupReplay() {
  const s = $("#timeSlider");
  s.max = REPLAY.rows.length; s.min = LSTM.lookback; s.value = 610;
  s.addEventListener("input", updateReplay);
  $("#playBtn").addEventListener("click", () => {
    playing = !playing; $("#playBtn").textContent = playing ? "❚❚ Pause" : "▶ Play";
    if (playing) timer = setInterval(() => {
      let v = parseInt(s.value) + 2; if (v > parseInt(s.max)) v = parseInt(s.min);
      s.value = v; updateReplay();
    }, 60);
    else clearInterval(timer);
  });
  window.addEventListener("resize", updateReplay);
  updateReplay();
}

// ---------------------------------------------------------------- vision
const VIR = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function viridis(v) {
  v = Math.max(0, Math.min(1, v)) * (VIR.length - 1);
  const i = Math.min(VIR.length - 2, Math.floor(v)), f = v - i;
  return VIR[i].map((c, k) => c + f * (VIR[i + 1][k] - c));
}
let galleryImg;
function classifyCanvas(src, trueLabel) {
  const c = $("#picked"), g = c.getContext("2d");
  g.drawImage(src, 0, 0, 64, 64);
  const px = g.getImageData(0, 0, 64, 64).data;
  const t = performance.now();
  const { probs, maps } = NN.runCNN(CNN, px);
  const ms = performance.now() - t;
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const best = CNN.classes[order[0][1]];
  $("#visionOut").innerHTML = `
    <div style="font-size:20px;font-weight:700">${best}</div>
    ${trueLabel ? `<div class="small" style="color:${trueLabel === best ? "#3ccf8e" : "#ff5c6c"}">true label: ${trueLabel} ${trueLabel === best ? "✓" : "✗"}</div>` : `<div class="small muted">uploaded image</div>`}
    <div id="topk"></div><div class="small muted">inference ${ms.toFixed(0)} ms in your browser</div>`;
  renderBars($("#topk"), order.map((o) => CNN.classes[o[1]]), order.map((o) => o[0]), (l, i) => (i ? "#3d5a8a" : "#5fb3ff"));
  const fm = maps.b1c1, holder = $("#fmaps");
  holder.innerHTML = "";
  for (let ch = 0; ch < 16; ch++) {
    const cv = document.createElement("canvas"); cv.width = fm.W; cv.height = fm.H;
    const cg = cv.getContext("2d"), im = cg.createImageData(fm.W, fm.H);
    let mx = 1e-6; for (let i = 0; i < fm.H * fm.W; i++) mx = Math.max(mx, fm.data[i * fm.C + ch]);
    for (let i = 0; i < fm.H * fm.W; i++) {
      const [r, gg, bb] = viridis(fm.data[i * fm.C + ch] / mx);
      im.data[i * 4] = r; im.data[i * 4 + 1] = gg; im.data[i * 4 + 2] = bb; im.data[i * 4 + 3] = 255;
    }
    cg.putImageData(im, 0, 0); cv.title = `channel ${ch}`; holder.appendChild(cv);
  }
}
function setupVision() {
  const gc = $("#gallery"), g = gc.getContext("2d");
  galleryImg = new Image();
  galleryImg.onload = () => {
    g.imageSmoothingEnabled = false; g.drawImage(galleryImg, 0, 0, gc.width, gc.height);
    pickTile(GALLERY.items[3]);
  };
  galleryImg.src = "img/gallery.png";
  const tile = document.createElement("canvas"); tile.width = tile.height = 64;
  const pickTile = (it) => {
    tile.getContext("2d").drawImage(galleryImg, it.x, it.y, 64, 64, 0, 0, 64, 64);
    classifyCanvas(tile, it.label);
  };
  gc.addEventListener("click", (e) => {
    const r = gc.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * galleryImg.width, y = ((e.clientY - r.top) / r.height) * galleryImg.height;
    const it = GALLERY.items.find((i) => x >= i.x && x < i.x + 64 && y >= i.y && y < i.y + 64);
    if (it) pickTile(it);
  });
  $("#upload").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    const img = new Image();
    img.onload = () => {
      const s = Math.min(img.width, img.height), c = document.createElement("canvas"); c.width = c.height = 64;
      c.getContext("2d").drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, 64, 64);
      classifyCanvas(c, null);
    };
    img.src = URL.createObjectURL(f);
  });
}

// ---------------------------------------------------------------- experiments + results
const pct = (x) => (x * 100).toFixed(2) + "%";
function setupExperiments() {
  const S = SUMMARY;
  const e6 = S.exp6_transfer.runs, c = S.exp7_8_comparison;
  const cards = [
    ["Exp 1", "Back-propagation", "A 7-16-3 MLP written from scratch in NumPy, with hand-derived gradients checked against finite differences.", `test acc ${pct(S.exp1_backprop.test_accuracy)} · grad-check err ${S.exp1_backprop.gradient_check_max_rel_error.toExponential(1)}`],
    ["Exp 2", "Feed-forward NN", "A single-hidden-layer Keras FFNN that classifies telemetry as Normal, Warning or Critical.", `test acc ${pct(S.exp2_ffnn.test_accuracy)} · macro-F1 ${S.exp2_ffnn.test_macro_f1}`],
    ["Exp 3", "Deep NN", "A depth study (1 to 4 layers) and a 128-64-32 DNN with BatchNorm and Dropout. The live health monitor uses this DNN.", `test acc ${pct(S.exp3_dnn.test_accuracy)} · ${S.exp3_dnn.params.toLocaleString()} params`],
    ["Exp 4", "Feature maps", "Shows the CNN's first-layer filters, its activation maps at each depth, and Grad-CAM heat-maps.", `${S.exp4_feature_maps.layer_stats.length} layers visualised`],
    ["Exp 5", "CNN", "A compact CNN trained from scratch on EuroSAT RGB (27,000 images, 10 classes).", `test acc ${pct(S.exp5_cnn.test_accuracy)} · ${Math.round(S.exp5_cnn.params * 4 / 1024)} KB`],
    ["Exp 6", "Transfer learning", "An ImageNet MobileNetV2 at α = 1.0 and α = 0.35, used first as a frozen feature extractor and then fine-tuned.", e6.map((r) => `α${r.alpha}: ${pct(r.fine_tuned.test_accuracy)}`).join(" · ")],
    ["Exp 7", "RNN", "A SimpleRNN that forecasts battery voltage and temperature 15 minutes ahead from 60 minutes of history.", `V MAE ${(c.rnn.battery_voltage.mae * 1000).toFixed(1)} mV · T MAE ${c.rnn.battery_temp.mae.toFixed(2)} °C`],
    ["Exp 8", "LSTM", "The same task with an LSTM, compared against the RNN and a persistence baseline. The orbit replay uses this LSTM.", `V MAE ${(c.lstm.battery_voltage.mae * 1000).toFixed(1)} mV · T MAE ${c.lstm.battery_temp.mae.toFixed(2)} °C`],
  ];
  $("#expCards").innerHTML = cards.map(([n, t, d, m]) => `<div class="exp"><div class="n">${n}</div><h3>${t}</h3><p>${d}</p><div class="metric">${m}</div></div>`).join("");

  const row = (cells) => `<tr>${cells.map((x, i) => `<td class="${i ? "num" : ""}">${x}</td>`).join("")}</tr>`;
  const cls = [["NumPy MLP (Exp 1)", S.exp1_backprop], ["FFNN (Exp 2)", S.exp2_ffnn], ["DNN (Exp 3)", S.exp3_dnn]];
  const wkb = (p) => Math.round(p * 4 / 1024);
  const img = [["CNN from scratch (Exp 5)", S.exp5_cnn.params, S.exp5_cnn.test_accuracy, S.exp5_cnn.test_macro_f1, wkb(S.exp5_cnn.params)]]
    .concat(e6.flatMap((r) => [
      [`MobileNetV2 α${r.alpha} frozen (Exp 6)`, r.total_params, r.feature_extraction.test_accuracy, r.feature_extraction.test_macro_f1, wkb(r.total_params)],
      [`MobileNetV2 α${r.alpha} fine-tuned (Exp 6)`, r.total_params, r.fine_tuned.test_accuracy, r.fine_tuned.test_macro_f1, wkb(r.total_params)]]));
  $("#resultsTables").innerHTML = `
    <h4>Telemetry health classification (held-out runs)</h4>
    <table><tr><th>Model</th><th>Params</th><th>Accuracy</th><th>Macro-F1</th></tr>
    ${cls.map(([n, r]) => row([n, r.params.toLocaleString(), pct(r.test_accuracy), r.test_macro_f1])).join("")}</table>
    <h4>EuroSAT land-cover classification (4,050 test images)</h4>
    <table><tr><th>Model</th><th>Params</th><th>Accuracy</th><th>Macro-F1</th><th>Weights (KB)</th></tr>
    ${img.map(([n, p, a, f, s]) => row([n, p.toLocaleString(), pct(a), f, s])).join("")}</table>
    <h4>15-minute forecasting (held-out runs) · mean absolute error</h4>
    <table><tr><th>Model</th><th>Params</th><th>Voltage (mV)</th><th>Voltage @ t+15 (mV)</th><th>Temp (°C)</th><th>Temp @ t+15 (°C)</th></tr>
    ${[["Persistence baseline", "-", c.persistence], ["SimpleRNN (Exp 7)", c.params.rnn, c.rnn], ["LSTM (Exp 8)", c.params.lstm, c.lstm]]
      .map(([n, p, m]) => row([n, p.toLocaleString(), (m.battery_voltage.mae * 1000).toFixed(1), (m.battery_voltage["mae_t+15"] * 1000).toFixed(1), m.battery_temp.mae.toFixed(2), m.battery_temp["mae_t+15"].toFixed(2)])).join("")}</table>`;

  const plots = [
    ["exp1_training.png", "Exp 1 · NumPy backprop training"], ["exp3_depth_study.png", "Exp 3 · effect of depth"],
    ["exp3_confusion.png", "Exp 3 · DNN confusion matrix"], ["exp4_filters.png", "Exp 4 · first-layer filters"],
    ["exp4_feature_maps_river.png", "Exp 4 · feature maps (River)"], ["exp4_gradcam.png", "Exp 4 · Grad-CAM"],
    ["exp5_training.png", "Exp 5 · CNN training"], ["exp5_confusion.png", "Exp 5 · CNN confusion matrix"],
    ["exp6_transfer.png", "Exp 6 · transfer learning"], ["exp8_horizon_comparison.png", "Exp 7-8 · error vs horizon"],
    ["exp8_forecast_example.png", "Exp 8 · forecasts through an eclipse"],
  ];
  $("#plots").innerHTML = plots.map(([f, c]) => `<figure><a href="img/${f}" target="_blank"><img loading="lazy" src="img/${f}" alt="${c}"></a><figcaption>${c}</figcaption></figure>`).join("");
}

// ---------------------------------------------------------------- boot
(async () => {
  [HEALTH, HSCALE, LSTM, FSCALE, REPLAY, GALLERY, SUMMARY] = await Promise.all([
    "data/health_dnn.json", "data/health_scaler.json", "data/lstm_forecaster.json", "data/forecast_scaler.json",
    "data/replay_orbit.json", "data/gallery.json", "data/summary.json"].map(getJSON));
  setupHealth();
  setupReplay();
  setupExperiments();
  CNN = await NN.loadCNN("data/cnn_manifest.json", "data/cnn_weights.bin");
  setupVision();
})();
