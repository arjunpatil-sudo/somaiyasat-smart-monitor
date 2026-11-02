// Builds report/SomaiyaSat_Smart_Monitor_Report.docx from results/metrics/*.json and plots.
// Run: node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Header, Footer, PageNumber, TableOfContents,
  ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const M = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics", f + ".json")));
const PLOT = (f) => path.join(ROOT, "results/plots", f);
const FIG = (f) => path.join(ROOT, "report/figs", f);
const LINKS = JSON.parse(fs.readFileSync(path.join(__dirname, "links.json")));

const e1 = M("exp1_backprop"), e2 = M("exp2_ffnn"), e3 = M("exp3_dnn"), e4 = M("exp4_feature_maps");
const e5 = M("exp5_cnn"), e6 = M("exp6_transfer"), e7 = M("exp7_rnn"), e8 = M("exp8_lstm");
const a10 = e6.runs.find((r) => r.alpha === 1), a035 = e6.runs.find((r) => r.alpha === 0.35);
const pct = (x, d = 2) => (x * 100).toFixed(d) + "%";
const mv = (x) => (x * 1000).toFixed(1) + " mV";
const fmt = (n) => Number(n).toLocaleString("en-US");
const wkb = (params) => (params * 4 / 1024).toFixed(params < 1000 ? 1 : 0); // float32 weight size

// ------------------------------------------------------------------ helpers
const FONT = "Times New Roman";
const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 },
  children: runs(text, opts),
});
function runs(text, opts = {}) {
  // **bold** segments
  return String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) => s.startsWith("**")
    ? new TextRun({ text: s.slice(2, -2), bold: true, size: opts.size || 24, font: FONT })
    : new TextRun({ text: s, size: opts.size || 24, font: FONT, italics: opts.italics, bold: opts.bold }));
}
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, spacing: { after: 200 }, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t, font: FONT })] });
const T = (t, pageBreak = false) => new Paragraph({ pageBreakBefore: pageBreak, spacing: { before: 200, after: 120 },
  children: [new TextRun({ text: t, bold: true, size: 27, font: FONT, color: "2E5597" })] });   // titled but not in TOC
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 }, children: [new TextRun({ text: t, font: FONT })] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60, line: 288 }, children: runs(t) });
const N = (t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60, line: 288 }, children: runs(t) });
let figNo = 0, tabNo = 0;
function imgSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), b };
}
function Figure(file, caption, widthIn = 6.2) {
  const { w, h, b } = imgSize(file);
  const W = widthIn * 96, Hh = (W * h) / w;
  figNo++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: b, transformation: { width: W, height: Hh } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}: ${caption}`, italics: true, size: 20, font: FONT })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const borders = { top: border, bottom: border, left: border, right: border };
function Tbl(header, rows, widths, caption) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: head, size: 20, font: FONT })] })],
  });
  tabNo++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabNo}: ${caption}`, bold: true, size: 20, font: FONT })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const Center = (t, size, bold = false, after = 80, color) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after },
  children: [new TextRun({ text: t, size, bold, font: FONT, color })] });

// ------------------------------------------------------------------ numbers used in text
const bestImg = Math.max(e5.test_accuracy, a10.fine_tuned.test_accuracy, a035.fine_tuned.test_accuracy);
const lstmV = e8.test.battery_voltage, rnnV = e7.test.battery_voltage, perV = e8.persistence_baseline.battery_voltage;
const lstmT = e8.test.battery_temp, rnnT = e7.test.battery_temp, perT = e8.persistence_baseline.battery_temp;
const imp = (a, b) => ((1 - a / b) * 100).toFixed(1) + "%";
const better = (a, b) => (a < b ? "lower" : "higher");
const depth = e3.depth_study;
const cnnReport = e5.report;
const worstCls = Object.entries(cnnReport).filter(([k]) => !["accuracy", "macro avg", "weighted avg"].includes(k))
  .sort((a, b) => a[1]["f1-score"] - b[1]["f1-score"]).slice(0, 3).map(([k, v]) => `${k} (F1 ${v["f1-score"].toFixed(3)})`);

// ------------------------------------------------------------------ content
const children = [];

// ---- title page
children.push(
  Center("K J Somaiya College of Engineering, Mumbai", 32, true, 60),
  Center("(Somaiya Vidyavihar University)", 22, false, 60),
  Center("Department of Information Technology", 26, true, 400),
  Center("Deep Learning - Mini Project Report", 28, true, 120),
  Center("Academic Year 2026-27 (ODD Term) · LY B.Tech (IT) · Semester VII", 22, false, 500),
  Center("SomaiyaSat Smart Monitor", 44, true, 80, "1F3864"),
  Center("Deep Learning for Satellite Health Monitoring and Earth Image Classification", 26, false, 200),
  Center("AI Use Case: KJS-SRS-01 - SomaiyaSat & SomaiyaPod PocketQube Mission", 22, false, 500),
);
children.push(...Tbl(["Name of Student", "Roll Number", "Batch"],
  [["Yashveer Singh", "16010423117", "B2"], ["Arjun Patil", "16010423121", "B2"]], [4200, 2800, 2000], "Group members"));
children.push(
  new Paragraph({ spacing: { before: 200, after: 80 }, alignment: AlignmentType.CENTER, children: runs("**Course:** Deep Learning     **Course In-charge:** Prof. Sarika Dharangaonkar") }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "GitHub: ", bold: true, font: FONT, size: 22 }),
    new ExternalHyperlink({ link: LINKS.github, children: [new TextRun({ text: LINKS.github, style: "Hyperlink", font: FONT, size: 22 })] })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Live demo: ", bold: true, font: FONT, size: 22 }),
    new ExternalHyperlink({ link: LINKS.vercel, children: [new TextRun({ text: LINKS.vercel, style: "Hyperlink", font: FONT, size: 22 })] })] }),
);

// ---- aim + abstract
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(T("Aim"));
children.push(P("Design, implementation, report submission, and demonstration of an AI use case related mini project (group of 2 students)."));
children.push(T("Abstract"));
children.push(P("SomaiyaSat is a PocketQube satellite developed under the KJS-SRS-01 AI use case. It continuously sends housekeeping telemetry (battery voltage, currents and temperatures) and captures Earth images for SSTV transmission. Analysing this data by hand is slow and error-prone. This project applies deep learning to automate two tasks: monitoring the satellite's health and classifying the images it captures. The work is built directly on the eight Deep Learning lab experiments."));
children.push(P(`For health monitoring, a physics-based simulator of the PocketQube power and thermal system generated 138,240 minutes of labelled telemetry, with five injected fault types. A back-propagation network written from scratch in NumPy, a feed-forward network and a deep network classify each reading as Normal, Warning or Critical, reaching up to **${pct(Math.max(e1.test_accuracy, e2.test_accuracy, e3.test_accuracy))}** accuracy on orbits held out from training. Recurrent (SimpleRNN) and LSTM models forecast battery voltage and temperature 15 minutes ahead from the previous hour. The LSTM reaches a mean absolute error of **${mv(lstmV.mae)}** and **${lstmT.mae.toFixed(2)} °C**, against ${mv(perV.mae)} and ${perT.mae.toFixed(2)} °C for a persistence baseline.`));
children.push(P(`For image classification, a compact CNN (${fmt(e5.params)} parameters, ${wkb(e5.params)} KB of weights) trained from scratch on the EuroSAT Sentinel-2 dataset (27,000 images, 10 land-cover classes) achieves **${pct(e5.test_accuracy)}** test accuracy. Its feature maps and Grad-CAM heat-maps are visualised to show what the network learns. Transfer learning with ImageNet-pretrained MobileNetV2 raises accuracy to **${pct(a10.fine_tuned.test_accuracy)}** (α = 1.0). The α = 0.35 variant reaches ${pct(a035.fine_tuned.test_accuracy)} with ${fmt(a035.total_params)} parameters, which suits low-power on-board hardware. All models are compared on accuracy, loss, size and latency. They are deployed in a web dashboard where they run entirely in the browser, which shows that lightweight deep learning can support autonomous monitoring and smarter data handling on student-built satellites.`));
children.push(P("**Keywords:** PocketQube, telemetry, anomaly detection, back-propagation, deep neural network, CNN, feature maps, transfer learning, MobileNetV2, RNN, LSTM, EuroSAT.", { italics: false }));

children.push(T("Table of Contents", true));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));

// ---- 1 introduction
children.push(H1("1. Introduction"));
children.push(H2("1.1 Background"));
children.push(P("PocketQubes are 5 cm-cube picosatellites built by universities and student teams. SomaiyaSat, together with the SomaiyaPod deployer, forms the KJS-SRS-01 mission. Like every spacecraft, it reports housekeeping telemetry (battery state, power-bus currents and board temperatures) and carries a small camera whose Earth images are sent to the ground as SSTV (slow-scan television) frames. A ground-station operator must decide on every pass whether the satellite is healthy and which images are worth keeping."));
children.push(P("The satellite spends about one third of every ~95-minute orbit in Earth's shadow, so its battery, temperatures and power budget swing continuously. Spotting a problem early (a failing cell, a stuck heater, a latch-up on the power bus) gives operators time to act before the mission is lost. Similarly, tagging images automatically by what they show (forest, water, urban, farmland...) lets the team prioritise downlink time."));
children.push(H2("1.2 Problem statement"));
children.push(P("Build lightweight deep learning models that (a) classify SomaiyaSat's health from telemetry, (b) forecast critical telemetry channels ahead of time, and (c) classify the land type in Earth images. The models must be small enough to run on low-power hardware, and must be demonstrated in a working application."));
children.push(H2("1.3 Objectives"));
[
  "Generate realistic PocketQube telemetry, with labelled fault episodes, for training and evaluation.",
  "Implement back-propagation from first principles and verify it numerically (Exp 1).",
  "Train feed-forward and deep neural networks for three-class health classification (Exp 2, 3).",
  "Train a CNN from scratch on EuroSAT and visualise its feature maps (Exp 4, 5).",
  "Apply transfer learning with MobileNetV2 and study the accuracy versus size trade-off (Exp 6).",
  "Forecast battery voltage and temperature with RNN and LSTM models (Exp 7, 8).",
  "Compare all models on accuracy, loss and model size, and deploy them in a web dashboard (GitHub + Vercel).",
].forEach((t) => children.push(B(t)));
children.push(H2("1.4 Mapping of lab experiments to the project"));
children.push(...Tbl(["Exp", "Lab experiment", "CO", "Role in this project"], [
  ["1", "Back propagation algorithm", "CO1", "NumPy MLP health classifier with hand-coded gradients and a gradient check"],
  ["2", "Feed forward neural network", "CO2", "Single-hidden-layer FFNN health classifier"],
  ["3", "Deep neural network", "CO2", "Depth study and 128-64-32 BatchNorm/Dropout DNN (used on the dashboard)"],
  ["4", "Visualization of feature maps in CNN", "CO3", "First-layer filters, activation maps, Grad-CAM on Earth images"],
  ["5", "Convolutional neural network", "CO3", "CNN trained from scratch on EuroSAT (used on the dashboard)"],
  ["6", "Transfer learning with CNN", "CO3", "MobileNetV2 (α = 1.0 and 0.35) feature extraction and fine-tuning"],
  ["7", "Recurrent neural network", "CO4", "SimpleRNN 15-minute forecaster for voltage and temperature"],
  ["8", "Long short-term memory network", "CO4", "LSTM forecaster, compared with the RNN and a baseline (used on the dashboard)"],
], [700, 2900, 700, 4700], "Lab experiments and where each one is used"));

// ---- 2 literature
children.push(H1("2. Background and Related Work"));
children.push(P("**Telemetry monitoring.** Spacecraft operators traditionally use fixed limit checks on each channel. Data-driven methods instead learn the normal joint behaviour of many channels. Feed-forward and recurrent networks have been used for spacecraft anomaly detection, and LSTM-based forecasting of telemetry, with alerts raised on large prediction errors, was demonstrated on NASA's SMAP and MSL missions [5]. Our health classifier and forecasters follow the same two ideas at PocketQube scale."));
children.push(P("**Remote-sensing image classification.** EuroSAT [3] provides 27,000 labelled 64×64 Sentinel-2 patches in 10 land-use and land-cover classes. Its authors report about 98.6% accuracy with a fine-tuned ResNet-50. CNNs learn hierarchical features, from edges and colour blobs in early layers to textures and object parts deeper in the network, which feature-map visualisation and Grad-CAM [6] make visible."));
children.push(P("**Efficient CNNs and transfer learning.** MobileNetV2 [4] uses depthwise-separable convolutions in inverted-residual blocks with linear bottlenecks, which cut computation by roughly an order of magnitude compared with standard convolutions. A width multiplier α shrinks every layer, giving a family of models for constrained devices. Re-using ImageNet features (transfer learning) lets small datasets reach high accuracy quickly."));
children.push(P("**Recurrent networks.** A SimpleRNN keeps a hidden state that is updated at every time step, but its gradients vanish over long sequences. The LSTM [7] adds input, forget and output gates and a cell state, so information can flow across many time steps. This matters for telemetry, where the next 15 minutes depend on where the satellite is in its 95-minute orbit."));

// ---- 3 datasets
children.push(H1("3. Datasets"));
children.push(H2("3.1 SomaiyaSat telemetry (simulated)"));
children.push(P("Real SomaiyaSat downlink logs are not publicly available, so we wrote a physics-based simulator (src/telemetry_sim.py) of a 1S Li-ion PocketQube in a ~500 km low Earth orbit. It produces one sample per minute and models:"));
[
  "a 94.6-minute orbit with 37% of each orbit in eclipse, and body-mounted solar panels on a tumbling satellite;",
  "battery state of charge integrated from (solar - load) current, and terminal voltage = OCV(SOC) + I·R_internal;",
  "first-order thermal models of the battery and the on-board computer, with a thermostatic survival heater;",
  "payload activity: camera captures (+85 mA) and 6-10 minute SSTV transmissions (+210 mA);",
  "sensor noise on every channel (ADC and thermistor noise).",
].forEach((t) => children.push(B(t)));
children.push(P("Each of the 48 simulated two-day runs (2,880 minutes each, 138,240 samples in total) is either nominal or contains one injected fault: battery ageing (reduced capacity, higher resistance, leakage), heater failure, solar-string loss, power-bus latch-up, or a transmitter stuck at high power. The health label is computed from the true, noise-free state using operator limits (Table 3). The models only see the noisy readings, so the classes overlap near the limits, just as they do on a real satellite."));
children.push(...Tbl(["Class", "Rule (any condition)"], [
  ["Critical", "V < 3.40 V, or T_batt > 52 °C, or T_batt < -6 °C, or T_board > 70 °C, or load > 430 mA"],
  ["Warning", "V < 3.62 V, or T_batt > 40 °C, or T_batt < 2 °C, or T_board > 58 °C, or load > 300 mA, or sunlit with solar < 35 mA"],
  ["Normal", "none of the above"],
], [1400, 7600], "Health labelling rules"));
children.push(P("**Inputs (7):** battery voltage, battery current, battery temperature, board temperature, solar current, load current and the sun-sensor flag. **Split by run**, not by row, so that test orbits are never seen during training: runs with run mod 6 < 4 are used for training (92,160 samples), run mod 6 = 4 for validation and run mod 6 = 5 for testing (23,040 samples each). Features are standardised with the training-set mean and standard deviation."));
children.push(...Figure(PLOT("data_telemetry_sample.png"), "Ten hours of simulated telemetry (grey bands are eclipses). The heater can be seen cycling during eclipses.", 6.2));
children.push(...Figure(PLOT("data_telemetry_labels.png"), "Class distribution per split (log scale) and the label mix for each fault type", 6.2));
children.push(H2("3.2 EuroSAT RGB"));
children.push(P("EuroSAT RGB [3] (Zenodo record 7711810, EuroSAT_RGB.zip, 94.7 MB) contains 27,000 JPEG patches of 64×64 pixels at 10 m/pixel, from Sentinel-2 bands B4, B3 and B2. There are 10 classes with 2,000-3,000 images each: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River and SeaLake. It stands in for SomaiyaSat's Earth images. We used a stratified 70/15/15 split (seed 42): 18,900 training, 4,050 validation and 4,050 test images. Inputs are standardised per channel. Training uses random horizontal and vertical flips and 90° rotations, because satellite images have no fixed 'up'."));
children.push(...Figure(PLOT("data_eurosat_samples.png"), "EuroSAT RGB samples, three per class", 6.4));

// ---- 4 architecture
children.push(H1("4. System Architecture and Tools"));
children.push(...Figure(FIG("architecture.png"), "SomaiyaSat Smart Monitor: data flow from the satellite to the ground-station dashboard", 6.5));
children.push(P("The telemetry branch runs two models in parallel: a classifier that labels the current state, and a forecaster that predicts the next 15 minutes of the two channels most tied to mission loss (battery voltage and battery temperature). The image branch tags every downlinked image with a land-cover class. The trained weights are exported to JSON and binary files and executed in the browser by about 160 lines of plain JavaScript (dense, BatchNorm, LSTM, convolution and pooling layers), so the dashboard needs no server and could equally run on a ground-station laptop with no internet connection."));
children.push(...Tbl(["Component", "Technology"], [
  ["Language / DL framework", "Python 3.13, TensorFlow 2.21 / Keras 3.15, NumPy (Exp 1)"],
  ["Data and metrics", "pandas, scikit-learn (splits, F1, confusion matrices), matplotlib"],
  ["Pretrained model", "MobileNetV2 ImageNet weights (keras-applications release files)"],
  ["Dashboard", "Static HTML/CSS/JavaScript, custom in-browser inference (web/js/nn.js)"],
  ["Version control / hosting", "Git + GitHub; Vercel (static deployment)"],
  ["Hardware used for training", "2-core CPU, 8 GB RAM (no GPU)"],
], [3000, 6000], "Tools and technologies"));

// ---- 5 experiments
children.push(H1("5. Implementation and Results"));
children.push(P("Every experiment is a standalone script in src/ (e.g. exp5_cnn.py). Each script writes its metrics to results/metrics/*.json and its figures to results/plots/. All numbers in this chapter are read from those files. Seeds are fixed (42) for reproducibility."));

children.push(H2("5.1 Experiment 1 - Back-propagation from scratch"));
children.push(P(`A ${e1.architecture} was implemented in NumPy only. The forward pass computes z1 = XW1 + b1, a1 = tanh(z1), z2 = a1W2 + b2, p = softmax(z2). The backward pass applies the chain rule by hand: dL/dz2 = p - y (softmax with cross-entropy), dW2 = a1ᵀ·dz2, da1 = dz2·W2ᵀ, dz1 = da1 ⊙ (1 - a1²), dW1 = Xᵀ·dz1. A class-weighted loss compensates for the dominance of Normal samples.`));
children.push(P(`**Gradient check.** Before training, the analytic gradients were compared with central finite differences, (L(θ+ε) - L(θ-ε)) / 2ε with ε = 1e-5. The maximum relative error was **${e1.gradient_check_max_rel_error.toExponential(2)}** (mean ${e1.gradient_check_mean_rel_error.toExponential(2)}), which confirms the derivation. Training used ${e1.optimizer}, for ${e1.epochs} epochs.`));
children.push(...Tbl(["Metric", "Value"], [["Parameters", e1.params], ["Test accuracy", pct(e1.test_accuracy)], ["Test macro-F1", e1.test_macro_f1], ["Training time", e1.train_time_s + " s"]], [4500, 4500], "Exp 1 results (held-out test runs)"));
children.push(...Figure(PLOT("exp1_training.png"), "Exp 1 - loss and accuracy of the NumPy network", 6.0));
children.push(P(`**Observation:** a network with only ${e1.params} parameters, trained with plain SGD + momentum, already separates the three health states well. Most of its errors are Normal readings flagged as Warning (false alarms) rather than missed critical events.`));

children.push(H2("5.2 Experiment 2 - Feed-forward neural network"));
children.push(P(`The same task was solved in Keras with a feed-forward network: ${e2.architecture}, trained with ${e2.optimizer}. Keras handles automatic differentiation, which replaces the hand-derived gradients of Exp 1.`));
children.push(...Tbl(["Metric", "Value"], [["Parameters", e2.params], ["Test accuracy", pct(e2.test_accuracy)], ["Test macro-F1", e2.test_macro_f1], ["Weights (float32)", wkb(e2.params) + " KB"]], [4500, 4500], "Exp 2 results"));
children.push(...Figure(PLOT("exp2_training.png"), "Exp 2 - FFNN training curves", 6.0));

children.push(H2("5.3 Experiment 3 - Deep neural network"));
children.push(P("**Depth study.** Networks with 1 to 4 hidden layers of 64 ReLU units were trained with identical settings (Adam, 15 epochs)."));
children.push(...Tbl(["Hidden layers", "Parameters", "Test accuracy", "Macro-F1"], depth.map((d) => [d.hidden_layers, fmt(d.params), pct(d.test_accuracy), d.test_macro_f1]), [2250, 2250, 2250, 2250], "Exp 3 - effect of depth"));
children.push(P(`Accuracy rises from ${pct(depth[0].test_accuracy)} with one hidden layer to ${pct(depth[depth.length - 1].test_accuracy)} with four. The extra layers model the interactions between channels: for example, low solar current is only abnormal when the sun sensor says the satellite is sunlit, and a given voltage is worrying or not depending on the load.`));
children.push(P("**Regularisation study.** The 128-64-32 network was then trained with different dropout rates, with and without BatchNorm, and with and without class-weighted loss (Table 8). Heavy dropout (0.2) and class weighting both pushed the model towards raising extra Warnings, which lowers accuracy. BatchNorm with light dropout (0.1) and an unweighted loss gave the best result."));
children.push(...Tbl(["Dropout", "BatchNorm", "Class-weighted loss", "Test accuracy", "Macro-F1"], e3.regularisation_study.map((r) => [r.dropout, r.batchnorm ? "yes" : "no", r.class_weight ? "yes" : "no", pct(r.test_accuracy), r.test_macro_f1]), [1500, 1600, 2200, 1900, 1800], "Exp 3 - regularisation study (128-64-32 network)"));
children.push(P(`**Final DNN.** ${e3.architecture}. Training used ${e3.optimizer}, and early stopping ended it after ${e3.epochs_run} epochs. BatchNorm stabilises training, and dropout reduces co-adaptation of neurons. This model runs in the dashboard's live health monitor.`));
children.push(...Tbl(["Metric", "Value"], [["Parameters", fmt(e3.params)], ["Test accuracy", pct(e3.test_accuracy)], ["Test macro-F1", e3.test_macro_f1],
  ["Critical recall", e3.report.Critical.recall.toFixed(4)], ["Warning recall", e3.report.Warning.recall.toFixed(4)], ["Weights (float32)", wkb(e3.params) + " KB"]], [4500, 4500], "Exp 3 - final DNN results"));
children.push(...Figure(PLOT("exp3_depth_study.png"), "Exp 3 - test accuracy and macro-F1 against the number of hidden layers", 4.6));
children.push(...Figure(PLOT("exp3_confusion.png"), "Exp 3 - normalised confusion matrix of the DNN on held-out runs", 3.6));
children.push(P(`**Observation:** the DNN never confuses Critical with Normal in either direction. Every error is between neighbouring classes, which is expected because the noisy sensors make readings near a limit ambiguous. ${pct(e3.report.Critical.recall)} of Critical readings are caught, and the rest are reported as Warning, so an operator is still alerted. Compared with the single-hidden-layer FFNN (Exp 2), the DNN raises far fewer false Warnings on Normal data (${e3.confusion_matrix[0][1]} vs ${e2.confusion_matrix[0][1]} test minutes).`));

children.push(H2("5.4 Experiment 5 - Convolutional neural network (EuroSAT)"));
children.push(P("Experiment 5 is presented before Experiment 4 because Exp 4 visualises the network trained here."));
children.push(P(`**Architecture:** ${e5.architecture}. All convolutions are 3×3 with 'same' padding and no bias; each is followed by BatchNorm. Global average pooling replaces large fully-connected layers, which keeps the model at only **${fmt(e5.params)} parameters (${wkb(e5.params)} KB of float32 weights)**. **Training:** ${e5.optimizer}; ${e5.epochs_run} epochs in ${(e5.train_time_s / 60).toFixed(0)} minutes on a 2-core CPU.`));
children.push(P("**An issue we found and fixed:** in our first attempt the inputs were only rescaled to [0, 1]. Because EuroSAT images are dark, first-layer activations had tiny variance, and BatchNorm's running averages lagged far behind the batch statistics. Validation accuracy collapsed to 13-47% while training accuracy rose to 84%. Standardising each input channel with the training-set mean and standard deviation, and using a BatchNorm momentum of 0.9, fixed this."));
children.push(...Tbl(["Metric", "Value"], [["Test accuracy", pct(e5.test_accuracy)], ["Test macro-F1", e5.test_macro_f1], ["Parameters", fmt(e5.params)], ["Weights (float32)", wkb(e5.params) + " KB"], ["CPU latency (batch = 1)", e5.cpu_latency_ms_per_image + " ms / image"]], [4500, 4500], "Exp 5 - CNN results on 4,050 test images"));
children.push(...Figure(PLOT("exp5_training.png"), "Exp 5 - CNN training and validation curves", 6.0));
children.push(...Figure(PLOT("exp5_confusion.png"), "Exp 5 - normalised confusion matrix (test set)", 4.8));
children.push(P(`**Observation:** the hardest classes are ${worstCls.join(", ")}. PermanentCrop is confused mostly with HerbaceousVegetation and AnnualCrop, which look alike at 10 m per pixel, and some narrow rivers are taken for AnnualCrop field strips. Highway and River are occasionally swapped because both are thin linear features. Forest, Residential and SeaLake reach an F1 of about 0.99.`));

children.push(H2("5.5 Experiment 4 - Visualisation of feature maps"));
children.push(P("Using the Exp 5 CNN we visualised (a) the 32 learned first-layer 3×3 RGB filters, (b) the activation maps of four convolutional layers for sample images, and (c) Grad-CAM heat-maps, computed from the gradient of the predicted class score with respect to the last convolutional layer."));
children.push(...Figure(PLOT("exp4_filters.png"), "Exp 4 - the 32 first-layer 3×3 filters (RGB weights, min-max scaled per filter)", 5.0));
children.push(...Figure(PLOT("exp4_feature_maps_river.png"), "Exp 4 - feature maps for a River image at increasing depth (8 most active channels per layer)", 6.4));
children.push(...Figure(PLOT("exp4_gradcam.png"), "Exp 4 - Grad-CAM: the image regions that drive each prediction", 6.5));
children.push(...Tbl(["Layer", "Output shape", "Mean activation", "Fraction of zeros"], e4.layer_stats.map((s) => [s.layer, s.shape.join("×"), s.mean_activation, pct(s.fraction_zero, 1)]), [2250, 2250, 2250, 2250], "Exp 4 - activation statistics on 256 test images"));
children.push(P(`**Observation:** a 3×3 filter is too small to show a clear pattern on its own, but its effect is visible in the feature maps. In block 1, channels respond to edges and colour boundaries and keep the full 64×64 detail, so the river bank and field borders are traced sharply. By block 2 the maps are coarser (32×32) and more semantic: in the River example one channel fires on all the land and stays silent over the water, which effectively segments the river. The block 3 maps (16×16) respond to larger regions and textures. After ReLU the share of zero activations rises from ${pct(e4.layer_stats[0].fraction_zero, 0)} in the first layer to ${pct(e4.layer_stats[2].fraction_zero, 0)} in block 2, so deeper representations are sparser. Grad-CAM shows the network looking at the right evidence: the road for Highway, the bright roofs for Industrial, the channel for River and the whole water body for SeaLake. This supports trusting the classifier's decisions.`));

children.push(H2("5.6 Experiment 6 - Transfer learning with MobileNetV2"));
children.push(P(`ImageNet-pretrained MobileNetV2 backbones were adapted to EuroSAT. Images are resized from 64×64 to 128×128 inside the model (an earlier run at 96×96 gave 3.5 points lower frozen-feature accuracy) and scaled to [-1, 1]. The new head is GlobalAveragePooling → Dropout(0.3) → Dense(10, softmax). **Phase A (feature extraction):** the backbone is frozen and only the head is trained (backbone features are computed once). **Phase B (fine-tuning):** inverted-residual blocks 10-16 and the final 1×1 convolution are unfrozen (from backbone layer ${a10.fine_tune_from_layer} of ${a10.backbone_layers}), BatchNorm statistics are kept frozen, and the model is trained end-to-end for up to 12 epochs with Adam at 1e-4 (halved on plateau) and the same augmentation as Exp 5.`));
children.push(...Tbl(["Model", "Total params", "Frozen acc.", "Fine-tuned acc.", "Fine-tuned F1", "Weights (KB)", "Latency (ms)"],
  [[`MobileNetV2 α = 1.0`, fmt(a10.total_params), pct(a10.feature_extraction.test_accuracy), pct(a10.fine_tuned.test_accuracy), a10.fine_tuned.test_macro_f1, wkb(a10.total_params), a10.cpu_latency_ms_per_image],
   [`MobileNetV2 α = 0.35`, fmt(a035.total_params), pct(a035.feature_extraction.test_accuracy), pct(a035.fine_tuned.test_accuracy), a035.fine_tuned.test_macro_f1, wkb(a035.total_params), a035.cpu_latency_ms_per_image],
   ["CNN from scratch (Exp 5)", fmt(e5.params), "-", pct(e5.test_accuracy), e5.test_macro_f1, wkb(e5.params), e5.cpu_latency_ms_per_image]],
  [2100, 1250, 1150, 1250, 1150, 1050, 1050], "Exp 6 - transfer learning compared with the scratch CNN"));
children.push(...Figure(PLOT("exp6_transfer.png"), "Exp 6 - validation accuracy across both phases, and test accuracy", 6.2));
children.push(...Figure(PLOT("exp6_confusion.png"), "Exp 6 - confusion matrix of the best fine-tuned MobileNetV2", 4.8));
children.push(P(TL_OBS()));

children.push(H2("5.7 Experiment 7 - Recurrent neural network"));
children.push(P(`**Task:** given the last ${e7.lookback_min} minutes of all 7 telemetry channels, predict battery voltage and battery temperature for each of the next ${e7.horizon_min} minutes (a 15 × 2 output). Windows are taken every 3 minutes and never cross run boundaries (${fmt(e7.windows.train)} training, ${fmt(e7.windows.val)} validation and ${fmt(e7.windows.test)} test windows). **Model:** SimpleRNN(64, tanh) → Dense(30) → reshape to 15 × 2, with ${fmt(e7.params)} parameters. Trained with MSE loss, Adam (2e-3), early stopping and LR reduction (${e7.epochs_run} epochs).`));
children.push(P(`**Baseline:** a persistence forecast (the future equals the last value), which is hard to beat over short horizons. The SimpleRNN reaches a voltage MAE of **${mv(rnnV.mae)}** against ${mv(perV.mae)} for persistence, and a temperature MAE of **${rnnT.mae.toFixed(2)} °C** against ${perT.mae.toFixed(2)} °C.`));
children.push(...Figure(PLOT("exp7_training.png"), "Exp 7 - SimpleRNN training curves (MSE on standardised targets)", 6.0));

children.push(H2("5.8 Experiment 8 - Long Short-Term Memory network"));
children.push(P(`The SimpleRNN layer was replaced by an LSTM(64) with otherwise identical settings (${fmt(e8.params)} parameters, ${e8.epochs_run} epochs). The LSTM's gates let it carry information about the orbit phase (time since entering eclipse) and the recent load history across the 60-step window.`));
children.push(...Tbl(["Model", "Params", "V MAE", "V MAE @ t+15", "T MAE (°C)", "T MAE @ t+15 (°C)"], [
  ["Persistence", "-", mv(perV.mae), mv(perV["mae_t+15"]), perT.mae.toFixed(3), perT["mae_t+15"].toFixed(3)],
  ["SimpleRNN (Exp 7)", fmt(e7.params), mv(rnnV.mae), mv(rnnV["mae_t+15"]), rnnT.mae.toFixed(3), rnnT["mae_t+15"].toFixed(3)],
  ["LSTM (Exp 8)", fmt(e8.params), mv(lstmV.mae), mv(lstmV["mae_t+15"]), lstmT.mae.toFixed(3), lstmT["mae_t+15"].toFixed(3)],
], [2100, 1100, 1450, 1500, 1350, 1500], "Exp 7 vs Exp 8 - forecasting error on held-out runs"));
children.push(...Figure(PLOT("exp8_horizon_comparison.png"), "MAE at each horizon from 1 to 15 minutes: persistence, SimpleRNN and LSTM", 6.2));
children.push(...Figure(PLOT("exp8_forecast_example.png"), "15-minute-ahead forecasts on an unseen run with a battery-ageing fault (grey bands are eclipses)", 6.4));
children.push(P(RNN_OBS()));

// ---- 6 comparison
children.push(H1("6. Comparative Analysis"));
children.push(...Tbl(["Task", "Model", "Params", "Weights (KB)", "Main metric"], [
  ["Health classification", "NumPy MLP (Exp 1)", e1.params, wkb(e1.params), `acc ${pct(e1.test_accuracy)}, F1 ${e1.test_macro_f1}`],
  ["Health classification", "FFNN (Exp 2)", e2.params, wkb(e2.params), `acc ${pct(e2.test_accuracy)}, F1 ${e2.test_macro_f1}`],
  ["Health classification", "DNN (Exp 3)", fmt(e3.params), wkb(e3.params), `acc ${pct(e3.test_accuracy)}, F1 ${e3.test_macro_f1}`],
  ["Image classification", "CNN scratch (Exp 5)", fmt(e5.params), wkb(e5.params), `acc ${pct(e5.test_accuracy)}`],
  ["Image classification", "MobileNetV2 α1.0 (Exp 6)", fmt(a10.total_params), wkb(a10.total_params), `acc ${pct(a10.fine_tuned.test_accuracy)}`],
  ["Image classification", "MobileNetV2 α0.35 (Exp 6)", fmt(a035.total_params), wkb(a035.total_params), `acc ${pct(a035.fine_tuned.test_accuracy)}`],
  ["Forecasting (15 min)", "SimpleRNN (Exp 7)", fmt(e7.params), wkb(e7.params), `V ${mv(rnnV.mae)}, T ${rnnT.mae.toFixed(2)} °C`],
  ["Forecasting (15 min)", "LSTM (Exp 8)", fmt(e8.params), wkb(e8.params), `V ${mv(lstmV.mae)}, T ${lstmT.mae.toFixed(2)} °C`],
], [2000, 2400, 1200, 1100, 2300], "All models side by side (held-out test data)"));
COMPARE_OBS().forEach((t) => children.push(B(t)));

// ---- 7 dashboard
children.push(H1("7. Web Dashboard and Deployment"));
children.push(P("The demonstration is a static web application (web/) deployed on Vercel from the GitHub repository. All inference runs in the visitor's browser using the exported weights, so there is no server and no Python is needed at run time:"));
[
  "**Live health monitor (Exp 3 DNN):** sliders for all 7 telemetry inputs, plus scenario presets (nominal, eclipse, SSTV downlink, latch-up, solar loss, heater failure, battery ageing). Shows the predicted state and class probabilities.",
  "**Orbit replay with forecast (Exp 8 LSTM):** plays back a held-out run that contains a battery-ageing fault. Every minute the LSTM forecasts the next 15 minutes of voltage and temperature, and the DNN labels the current state, which is shown next to the ground truth.",
  "**Earth image classifier (Exp 5 CNN, Exp 4 feature maps):** click any of 30 held-out test images, or upload an image. The CNN's top-3 classes and its first-layer feature maps are computed in the browser.",
  "**Experiments and results:** cards for all eight lab experiments, the comparison tables and every plot.",
].forEach((t) => children.push(B(t)));
if (fs.existsSync(FIG("dashboard.png"))) children.push(...Figure(FIG("dashboard.png"), "The deployed dashboard", 6.3));
if (fs.existsSync(FIG("dashboard_vision.png"))) children.push(...Figure(FIG("dashboard_vision.png"), "Dashboard - orbit replay with LSTM forecast and the Earth image classifier", 6.3));
children.push(P("**Correctness of the browser models:** the JavaScript CNN (with BatchNorm folded into the convolution weights), the DNN and the LSTM were checked against Keras on the same inputs and agree to within floating-point rounding (see Appendix A)."));
children.push(P("**Deployment steps:** (1) push the repository to GitHub; (2) in Vercel choose 'Add New → Project', import the repository and deploy. vercel.json already sets the output directory to web/ with no build step."));

// ---- 8 conclusion
children.push(H1("8. Conclusion and Future Scope"));
children.push(P(CONCLUSION()));
children.push(H2("Limitations"));
[
  "The telemetry is simulated. It follows real PocketQube physics, but the models must be re-trained or fine-tuned on actual SomaiyaSat downlink data once it is available.",
  "EuroSAT images (10 m/pixel, Sentinel-2) differ from SomaiyaSat camera and SSTV frames in resolution, colour response and compression, so domain adaptation will be needed.",
  "The health labels come from fixed operator limits. Novel faults outside these rules would need an unsupervised method (e.g. alerts on forecast error).",
].forEach((t) => children.push(B(t)));
children.push(H2("Future scope"));
[
  "Raise alerts when the LSTM forecast deviates from the observed telemetry, to detect faults the rules do not cover.",
  "Quantise the CNN or MobileNetV2-α0.35 to int8 (TensorFlow Lite) for an on-board microcontroller, so images can be filtered before downlink.",
  "Simulate SSTV transmission noise during training, to make the image classifier robust to it.",
  "Connect the dashboard to the live ground-station decoder.",
].forEach((t) => children.push(B(t)));

// ---- references
children.push(H1("9. References"));
[
  "J. Patterson and A. Gibson, Deep Learning: A Practitioner's Approach, O'Reilly Media, 2017.",
  "I. Goodfellow, Y. Bengio and A. Courville, Deep Learning, MIT Press, 2016.",
  "P. Helber, B. Bischke, A. Dengel and D. Borth, \"EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,\" IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 12, no. 7, pp. 2217-2226, 2019.",
  "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, \"MobileNetV2: Inverted Residuals and Linear Bottlenecks,\" Proc. IEEE CVPR, 2018.",
  "K. Hundman, V. Constantinou, C. Laporte, I. Colwell and T. Soderstrom, \"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding,\" Proc. ACM KDD, 2018.",
  "R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,\" Proc. IEEE ICCV, 2017.",
  "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.",
  "N. Buduma, Fundamentals of Deep Learning, O'Reilly Media, 2017.",
].forEach((t) => children.push(N(t)));

// ---- appendix
children.push(H1("Appendix A - Project Structure and How to Run"));
children.push(...Tbl(["Path", "Contents"], [
  ["src/telemetry_sim.py", "PocketQube telemetry simulator (data/telemetry.csv)"],
  ["src/exp1_backprop.py ... exp8_lstm.py", "One script per lab experiment"],
  ["src/export_web.py", "Exports weights, gallery and results to web/"],
  ["results/metrics, results/plots", "All metrics (JSON) and figures used in this report"],
  ["models/", "Trained Keras models (.keras)"],
  ["web/", "Dashboard: index.html, js/nn.js (inference), js/app.js, data/, img/"],
  ["report/", "This report (DOCX/PDF) and its generator"],
  ["run_all.sh, requirements.txt, vercel.json", "Re-run everything; dependencies; Vercel config"],
], [3800, 5200], "Repository layout"));
children.push(P("**Run:** pip install -r requirements.txt; put EuroSAT_RGB.zip in data/; run ./run_all.sh. **View locally:** python -m http.server -d web 8000 and open http://localhost:8000."));
if (fs.existsSync(path.join(ROOT, "results/metrics/web_parity.json"))) {
  const wp = JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics/web_parity.json")));
  children.push(P(`**Browser-model check:** maximum absolute difference between JavaScript and Keras outputs: CNN ${wp.cnn_max_abs_diff.toExponential(1)} (${wp.cnn_n} images, top-1 agreement ${pct(wp.cnn_top1_agree, 0)}), DNN ${wp.dnn_max_abs_diff.toExponential(1)}, LSTM ${wp.lstm_max_abs_diff.toExponential(1)}.`));
}

// ------------------------------------------------------------------ observation text (data-driven)
function TL_OBS() {
  const gain = a10.fine_tuned.test_accuracy - e5.test_accuracy;
  const ftGain = a10.fine_tuned.test_accuracy - a10.feature_extraction.test_accuracy;
  return `**Observation:** with the backbone frozen, ImageNet features alone give ${pct(a10.feature_extraction.test_accuracy)} (α = 1.0) and ${pct(a035.feature_extraction.test_accuracy)} (α = 0.35) after only a few minutes of head training. Fine-tuning the top blocks adds ${(ftGain * 100).toFixed(1)} percentage points for α = 1.0, reaching ${pct(a10.fine_tuned.test_accuracy)}, which is ${gain >= 0 ? (gain * 100).toFixed(1) + " points above" : (-gain * 100).toFixed(1) + " points below"} our scratch CNN. The α = 0.35 model has ${(a10.total_params / a035.total_params).toFixed(1)}× fewer parameters than α = 1.0 and reaches ${pct(a035.fine_tuned.test_accuracy)}. This is the accuracy-versus-size trade-off a low-power PocketQube has to make. Our scratch CNN remains the smallest model (${fmt(e5.params)} parameters), which is why it is the one deployed in the browser.`;
}
function RNN_OBS() {
  const vs = (a, b) => `${imp(a, b)} ${a < b ? "lower" : "higher"}`;
  return `**Observation:** both recurrent models clearly beat persistence. The persistence error grows quickly with the horizon (temperature: ${perT["mae_t+1"].toFixed(2)} °C at t+1 but ${perT["mae_t+15"].toFixed(2)} °C at t+15), because the satellite keeps crossing between sunlight and eclipse. The learned models know the orbit phase from the input window and anticipate these crossings. Compared with the SimpleRNN, the LSTM's voltage MAE is ${vs(lstmV.mae, rnnV.mae)} and its temperature MAE is ${vs(lstmT.mae, rnnT.mae)}, and the gap widens towards t+15 (voltage ${mv(rnnV["mae_t+15"])} vs ${mv(lstmV["mae_t+15"])}). This is consistent with the LSTM's gates keeping longer-range context that the SimpleRNN's hidden state loses. The sensors themselves have noise with a standard deviation of 18 mV and 0.9 °C, which puts the lowest achievable MAE at about 14 mV and 0.72 °C, so the LSTM is close to this floor. In the example run the forecasts follow the eclipse cycles and the steady decline caused by the ageing cell. The remaining error is concentrated at the sharp voltage drops at eclipse entry late in the fault, which the models anticipate only partially and follow a few minutes late.`;
}
function COMPARE_OBS() {
  return [
    `For telemetry health, all three networks exceed ${pct(Math.min(e1.test_accuracy, e2.test_accuracy, e3.test_accuracy), 1)} accuracy on unseen orbits. The remaining errors are borderline readings next to a limit, where the sensor noise makes the true class ambiguous.`,
    `Depth helps the health task (Exp 3 depth study) and a lot for images: a ${fmt(e5.params)}-parameter CNN reaches ${pct(e5.test_accuracy)}, and ImageNet transfer learning reaches ${pct(bestImg)}.`,
    `The models used on the dashboard are all small (float32 weights): DNN ${wkb(e3.params)} KB, CNN ${wkb(e5.params)} KB, LSTM ${wkb(e8.params)} KB. A desktop CPU runs the CNN in ${e5.cpu_latency_ms_per_image} ms per image, and the browser version takes about 0.15 s. MobileNetV2 α1.0 needs ${wkb(a10.total_params)} KB, which is ${(a10.total_params / e5.params).toFixed(0)}× more than our CNN.`,
    `The LSTM is the better forecaster (Exp 8 vs Exp 7) for a modest increase in parameters (${fmt(e7.params)} → ${fmt(e8.params)}).`,
  ];
}
function CONCLUSION() {
  return `We built SomaiyaSat Smart Monitor, an end-to-end deep learning system for the KJS-SRS-01 PocketQube mission that uses all eight Deep Learning lab experiments. Health classifiers built from back-propagation (Exp 1), feed-forward (Exp 2) and deep (Exp 3) networks label telemetry with up to ${pct(Math.max(e1.test_accuracy, e2.test_accuracy, e3.test_accuracy))} accuracy and never mistake a Critical state for a Normal one (Exp 3 DNN). RNN and LSTM forecasters (Exp 7, 8) predict battery voltage and temperature 15 minutes ahead, and the LSTM achieves ${mv(lstmV.mae)} and ${lstmT.mae.toFixed(2)} °C MAE. For Earth images, a ${fmt(e5.params)}-parameter CNN (Exp 5) reaches ${pct(e5.test_accuracy)} on EuroSAT. Its feature maps and Grad-CAM (Exp 4) show that it focuses on meaningful structures, and MobileNetV2 transfer learning (Exp 6) raises accuracy to ${pct(bestImg)}. All of this runs in a browser dashboard deployed on Vercel. The results show that lightweight deep learning models can support autonomous monitoring and smarter data handling in small, student-built satellites.`;
}

// ------------------------------------------------------------------ document
const doc = new Document({
  features: { updateFields: true },
  creator: "Yashveer Singh, Arjun Patil", title: "SomaiyaSat Smart Monitor - Deep Learning Mini Project Report",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, font: FONT, color: "2E5597" }, paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 540 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, right: 1300, bottom: 1300, left: 1300 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "KJSCE · IT · Deep Learning Mini Project · SomaiyaSat Smart Monitor", size: 16, color: "808080", font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, font: FONT })] })] }) },
    children,
  }],
});

const out = path.join(__dirname, "SomaiyaSat_Smart_Monitor_Report.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
