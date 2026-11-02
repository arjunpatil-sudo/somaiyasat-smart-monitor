// Minimal neural-network inference in plain JavaScript (no libraries).
// Weights are exported from the trained Keras models by src/export_web.py.

const NN = {};

NN.act = {
  relu: (x) => (x > 0 ? x : 0),
  tanh: Math.tanh,
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  linear: (x) => x,
};

NN.softmax = (v) => {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};

// ---------- Dense / BatchNorm stack (health DNN, Exp 3) ----------
NN.runDense = (layers, input) => {
  let x = input.slice();
  for (const L of layers) {
    if (L.type === "dense") {
      const out = L.b.slice();
      for (let i = 0; i < x.length; i++) {
        const xi = x[i], row = L.W[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      x = L.activation === "softmax" ? NN.softmax(out) : out.map(NN.act[L.activation] || NN.act.linear);
    } else if (L.type === "bn") {
      x = x.map((v, i) => L.gamma[i] * (v - L.mean[i]) / Math.sqrt(L.var[i] + L.eps) + L.beta[i]);
    } else if (L.type === "activation") {
      x = L.activation === "softmax" ? NN.softmax(x) : x.map(NN.act[L.activation]);
    }
  }
  return x;
};

// ---------- LSTM forecaster (Exp 8) ----------
// Keras gate order in kernel columns: input, forget, cell, output.
NN.runLSTM = (m, seq) => {
  const u = m.units;
  let h = new Float64Array(u), c = new Float64Array(u);
  for (const x of seq) {
    const z = Float64Array.from(m.bias);
    for (let i = 0; i < x.length; i++) {
      const row = m.kernel[i], xi = x[i];
      for (let j = 0; j < 4 * u; j++) z[j] += xi * row[j];
    }
    for (let i = 0; i < u; i++) {
      const row = m.recurrent_kernel[i], hi = h[i];
      if (hi === 0) continue;
      for (let j = 0; j < 4 * u; j++) z[j] += hi * row[j];
    }
    const hn = new Float64Array(u);
    for (let k = 0; k < u; k++) {
      const ig = NN.act.sigmoid(z[k]);
      const fg = NN.act.sigmoid(z[u + k]);
      const cg = Math.tanh(z[2 * u + k]);
      const og = NN.act.sigmoid(z[3 * u + k]);
      c[k] = fg * c[k] + ig * cg;
      hn[k] = og * Math.tanh(c[k]);
    }
    h = hn;
  }
  const out = Array.from(m.dense_b);
  for (let i = 0; i < u; i++) {
    const row = m.dense_W[i];
    for (let j = 0; j < out.length; j++) out[j] += h[i] * row[j];
  }
  return out; // flat [h0_v, h0_t, h1_v, h1_t, ...]
};

// ---------- CNN (Exp 5), HWC layout, BN folded into conv weights ----------
NN.conv3x3 = (inp, H, W, Cin, w, b, Cout) => {
  const out = new Float32Array(H * W * Cout);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * Cout;
      for (let co = 0; co < Cout; co++) out[o + co] = b[co];
      for (let ky = 0; ky < 3; ky++) {
        const iy = y + ky - 1;
        if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 3; kx++) {
          const ix = x + kx - 1;
          if (ix < 0 || ix >= W) continue;
          const ib = (iy * W + ix) * Cin;
          const wb = (ky * 3 + kx) * Cin * Cout;
          for (let ci = 0; ci < Cin; ci++) {
            const v = inp[ib + ci];
            if (v === 0) continue;
            const wr = wb + ci * Cout;
            for (let co = 0; co < Cout; co++) out[o + co] += v * w[wr + co];
          }
        }
      }
      for (let co = 0; co < Cout; co++) if (out[o + co] < 0) out[o + co] = 0; // ReLU
    }
  }
  return out;
};

NN.maxpool2 = (inp, H, W, C) => {
  const h = H >> 1, w = W >> 1, out = new Float32Array(h * w * C);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < C; c++) {
        const a = inp[((2 * y) * W + 2 * x) * C + c], b2 = inp[((2 * y) * W + 2 * x + 1) * C + c];
        const d = inp[((2 * y + 1) * W + 2 * x) * C + c], e = inp[((2 * y + 1) * W + 2 * x + 1) * C + c];
        out[(y * w + x) * C + c] = Math.max(a, b2, d, e);
      }
  return out;
};

// rgba: Uint8ClampedArray 64*64*4 from a canvas
NN.runCNN = (net, rgba) => {
  let H = 64, W = 64, C = 3;
  let x = new Float32Array(H * W * 3);
  const mu = net.rgb_mean, sd = net.rgb_std;
  for (let i = 0; i < H * W; i++) for (let c = 0; c < 3; c++) x[i * 3 + c] = (rgba[i * 4 + c] - mu[c]) / sd[c];
  const maps = {};
  for (const L of net.layers) {
    if (L.type === "conv") {
      x = NN.conv3x3(x, H, W, C, L.w, L.b, L.cout); C = L.cout;
      maps[L.name] = { data: x, H, W, C };
    } else if (L.type === "pool") {
      x = NN.maxpool2(x, H, W, C); H >>= 1; W >>= 1;
    } else if (L.type === "gap") {
      const g = new Float32Array(C);
      for (let i = 0; i < H * W; i++) for (let c = 0; c < C; c++) g[c] += x[i * C + c];
      x = g.map((v) => v / (H * W));
    } else if (L.type === "dense") {
      const out = Array.from(L.b);
      for (let i = 0; i < x.length; i++) for (let j = 0; j < out.length; j++) out[j] += x[i] * L.w[i * out.length + j];
      x = NN.softmax(out);
    }
  }
  return { probs: Array.from(x), maps };
};

// Loads manifest + binary weights produced by export_web.py
NN.loadCNN = async (manifestUrl, binUrl) => {
  const [man, buf] = await Promise.all([
    fetch(manifestUrl).then((r) => r.json()),
    fetch(binUrl).then((r) => r.arrayBuffer()),
  ]);
  const f = new Float32Array(buf);
  const layers = man.layers.map((L) => {
    const o = { ...L };
    if (L.w_offset !== undefined) o.w = f.subarray(L.w_offset, L.w_offset + L.w_size);
    if (L.b_offset !== undefined) o.b = f.subarray(L.b_offset, L.b_offset + L.b_size);
    return o;
  });
  return { classes: man.classes, rgb_mean: man.rgb_mean, rgb_std: man.rgb_std, layers };
};

window.NN = NN;
