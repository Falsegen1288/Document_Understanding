const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Layout Detection Benchmarking";
pres.title = "Layout Detection Benchmarking — Results Review";

const W = 13.333, H = 7.5;

// ---------- Palette ----------
const NAVY = "1C2B3A";       // primary dark text
const CHARCOAL = "33414E";   // secondary text
const MUTED = "6B7785";      // captions / footers
const RULE = "DCE2E7";       // hairlines
const PANEL = "F4F6F8";      // very light panel bg
const WHITE = "FFFFFF";

const BLUE = "4878A8";       // DocLayoutYOLO
const REDORANGE = "D9636B";  // Nemotron-Parse
const PURPLE = "9B59B6";     // ADE-DPT2
const GREEN = "2ECC71";      // Ground Truth

const GOOD_BG = "E8F8EE";
const WARN_BG = "FBF0DD";
const BAD_BG = "FBE7E8";
const GOOD_TX = "1E8449";
const WARN_TX = "9C6B0A";
const BAD_TX = "B23A42";

const TITLE_FONT = "Cambria";
const BODY_FONT = "Calibri";
const MONO_FONT = "Courier New";

// ---------- Helpers ----------

function footer(slide, num) {
  slide.addText(`${num}`, {
    x: W - 0.7, y: H - 0.42, w: 0.4, h: 0.3,
    fontFace: BODY_FONT, fontSize: 10, color: MUTED, align: "right", margin: 0,
  });
  slide.addText("Layout Detection Benchmarking — Results Review", {
    x: 0.5, y: H - 0.42, w: 6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, color: MUTED, align: "left", margin: 0,
  });
}

function slideTitle(slide, text, opts = {}) {
  slide.addText(text, {
    x: 0.5, y: 0.35, w: opts.w || W - 1.0, h: 0.6,
    fontFace: TITLE_FONT, fontSize: opts.fontSize || 30, bold: true,
    color: NAVY, align: "left", margin: 0,
  });
}

// Draw model color indicator chip
function modelChip(slide, x, y, label, color) {
  const w = 0.16, h = 0.16;
  slide.addShape(pres.shapes.OVAL, { x, y, w, h, fill: { color }, line: { type: "none" } });
  slide.addText(label, {
    x: x + w + 0.08, y: y - 0.07, w: 2.6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, margin: 0, valign: "middle",
  });
}

// Formula callout box: monospace formula + label
function formulaBox(slide, x, y, w, h, formulaLines) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  const runs = [];
  formulaLines.forEach((line, i) => {
    runs.push({ text: line, options: { breakLine: i < formulaLines.length - 1 } });
  });
  slide.addText(runs, {
    x: x + 0.15, y: y, w: w - 0.3, h: h,
    fontFace: MONO_FONT, fontSize: 11.5, italic: true, color: NAVY,
    valign: "middle", align: "left", margin: 0, lineSpacingMultiple: 1.08,
  });
}

// Metric block: Name header + formula box + significance + remark
function metricBlock(slide, x, y, w, name, formulaLines, significance, remark, formulaH) {
  let cy = y;
  slide.addText(name, {
    x, y: cy, w, h: 0.28,
    fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: NAVY, margin: 0,
  });
  cy += 0.32;
  const fh = formulaH || 0.42;
  formulaBox(slide, x, cy, w, fh, formulaLines);
  cy += fh + 0.08;
  slide.addText([
    { text: "Significance: ", options: { bold: true, color: CHARCOAL } },
    { text: significance, options: { color: CHARCOAL } },
  ], {
    x, y: cy, w, h: 0.5,
    fontFace: BODY_FONT, fontSize: 10.5, margin: 0, valign: "top", lineSpacingMultiple: 1.05,
  });
  cy += 0.5;
  slide.addText([
    { text: "Threshold/Remark: ", options: { bold: true, color: CHARCOAL } },
    { text: remark, options: { color: CHARCOAL } },
  ], {
    x, y: cy, w, h: 0.5,
    fontFace: BODY_FONT, fontSize: 10.5, margin: 0, valign: "top", lineSpacingMultiple: 1.05,
  });
}

function sectionDivider(slide, x, y, w) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w, h: 0,
    line: { color: RULE, width: 1 },
  });
}

const headerFill = { color: NAVY };
const headerOpts = { color: WHITE, bold: true, fontFace: BODY_FONT, fontSize: 10.5, align: "center", valign: "middle", fill: headerFill };
function cell(text, opts = {}) {
  return { text: String(text), options: { fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, align: "center", valign: "middle", ...opts } };
}
function headCell(text) {
  return { text, options: { ...headerOpts } };
}

// ============================================================
// SLIDE 1 — Title
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };

  // subtle structural accent: thin rule + label, no stripe
  slide.addText("BENCHMARK RESULTS REVIEW", {
    x: 0.9, y: 2.15, w: 8, h: 0.35,
    fontFace: BODY_FONT, fontSize: 13, color: MUTED, charSpacing: 3, bold: true, margin: 0,
  });

  slide.addText("Layout Detection Benchmarking", {
    x: 0.9, y: 2.55, w: 11.3, h: 1.0,
    fontFace: TITLE_FONT, fontSize: 44, bold: true, color: NAVY, margin: 0,
  });
  slide.addText("Results Review", {
    x: 0.9, y: 3.35, w: 11.3, h: 0.8,
    fontFace: TITLE_FONT, fontSize: 44, bold: true, color: NAVY, margin: 0,
  });

  slide.addText("DocLayoutYOLO  vs  NVIDIA Nemotron-Parse-v1.1  vs  LandingAI ADE-DPT2", {
    x: 0.9, y: 4.35, w: 11.3, h: 0.5,
    fontFace: BODY_FONT, fontSize: 16, italic: true, color: CHARCOAL, margin: 0,
  });

  // model legend chips
  modelChip(slide, 0.9, 5.15, "DocLayoutYOLO", BLUE);
  modelChip(slide, 3.4, 5.15, "Nemotron-Parse-v1.1", REDORANGE);
  modelChip(slide, 6.4, 5.15, "ADE-DPT2", PURPLE);
  modelChip(slide, 8.4, 5.15, "Ground Truth", GREEN);

  sectionDivider(slide, 0.9, 5.7, 11.5);
  slide.addText("Traditional public-dataset benchmark  +  Custom real-document benchmark vs human ground truth", {
    x: 0.9, y: 5.85, w: 11.3, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, color: MUTED, margin: 0,
  });
  footer(slide, 1);
}

// ============================================================
// SLIDE 2 — Setup (models + datasets tables)
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Evaluation Setup");
  footer(slide, 2);

  slide.addText("Models Evaluated", {
    x: 0.5, y: 1.05, w: 6, h: 0.32,
    fontFace: BODY_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });

  const modelsTable = [
    [headCell("Model"), headCell("Architecture"), headCell("Params"), headCell("Used in")],
    [
      { text: "DocLayoutYOLO", options: { fontFace: BODY_FONT, fontSize: 10, bold: true, color: BLUE, align: "left", valign: "middle" } },
      cell("YOLO detector", { align: "left" }),
      cell("~20M"),
      cell("Both benchmarks", { align: "left" }),
    ],
    [
      { text: "Nemotron-Parse-v1.1", options: { fontFace: BODY_FONT, fontSize: 10, bold: true, color: REDORANGE, align: "left", valign: "middle" } },
      cell("ViT-H encoder + mBart decoder", { align: "left" }),
      cell("~885M"),
      cell("Both benchmarks", { align: "left" }),
    ],
    [
      { text: "ADE-DPT2", options: { fontFace: BODY_FONT, fontSize: 10, bold: true, color: PURPLE, align: "left", valign: "middle" } },
      cell("Cloud API parser", { align: "left" }),
      cell("N/A"),
      cell("Traditional only", { align: "left" }),
    ],
  ];
  slide.addTable(modelsTable, {
    x: 0.5, y: 1.42, w: 6.0, colW: [2.0, 2.3, 0.85, 0.85],
    border: { pt: 0.75, color: RULE }, autoPage: false,
    rowH: 0.42,
  });

  slide.addText("Datasets / Documents Used", {
    x: 6.85, y: 1.05, w: 6, h: 0.32,
    fontFace: BODY_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });

  const dataTable = [
    [headCell("Dataset / Doc"), headCell("Classes"), headCell("Pages eval."), headCell("Notes")],
    [cell("DocLayNet", { align: "left", bold: true }), cell("11"), cell("3"), cell("COCO format", { align: "left", fontSize: 9 })],
    [cell("PubLayNet", { align: "left", bold: true }), cell("5"), cell("3"), cell("COCO format", { align: "left", fontSize: 9 })],
    [cell("DocBank", { align: "left", bold: true }), cell("9"), cell("3"), cell("Token-level → block aggregation", { align: "left", fontSize: 9 })],
    [cell("Custom medical catalog\n(30pg PDF)", { align: "left", bold: true, fontSize: 9.5 }), cell("7 present\nin GT", { fontSize: 9.5 }), cell("5 (human-\nannotated GT,\n63 elements)", { fontSize: 9 }), cell("Real product document", { align: "left", fontSize: 9 })],
  ];
  slide.addTable(dataTable, {
    x: 6.85, y: 1.42, w: 6.0, colW: [1.9, 1.0, 1.3, 1.8],
    border: { pt: 0.75, color: RULE }, autoPage: false,
    rowH: [0.42, 0.4, 0.4, 0.4, 0.62],
  });

  // brief framing note
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 4.55, w: 12.33, h: 1.85, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  slide.addText("Two Evaluation Tracks", {
    x: 0.8, y: 4.72, w: 11.7, h: 0.32,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  slide.addText([
    { text: "Traditional benchmark — ", options: { bold: true, color: BLUE } },
    { text: "3 models × 3 public datasets (DocLayNet, PubLayNet, DocBank), scored against published COCO-style annotations.", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 5.1, w: 11.7, h: 0.55,
    fontFace: BODY_FONT, fontSize: 11.5, margin: 0, lineSpacingMultiple: 1.1,
  });
  slide.addText([
    { text: "Custom benchmark — ", options: { bold: true, color: REDORANGE } },
    { text: "2 models (DocLayoutYOLO, Nemotron-Parse) evaluated against human-labeled ground truth on a real 30-page medical product catalog.", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 5.65, w: 11.7, h: 0.6,
    fontFace: BODY_FONT, fontSize: 11.5, margin: 0, lineSpacingMultiple: 1.1,
  });
}

// ============================================================
// SLIDE 3 — Traditional benchmark metrics
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Traditional Benchmark — Metrics");
  footer(slide, 3);

  const colW = 3.9, gap = 0.27, x0 = 0.5;
  const y0 = 1.12;

  metricBlock(slide, x0, y0, colW,
    "IoU (Intersection over Union)",
    ["IoU = Area(A ∩ B) / Area(A ∪ B)"],
    "Base unit of \"how well two boxes overlap.\"",
    "IoU ≥ 0.50 is the standard match threshold.",
    0.42
  );

  metricBlock(slide, x0 + colW + gap, y0, colW,
    "Precision / Recall / F1",
    ["P = TP/(TP+FP)   R = TP/(TP+FN)", "F1 = 2PR / (P + R)"],
    "P = share of predictions that were real; R = share of real items found; F1 = balance of both.",
    "F1 > 0.60 is generally considered usable.",
    0.55
  );

  metricBlock(slide, x0 + 2 * (colW + gap), y0, colW,
    "11-pt Interpolated AP",
    ["AP = (1/11) · Σ max{P(r): r ≥ t}", "  for t = 0, 0.1, ..., 1.0"],
    "Rewards a model that stays precise across all recall levels, not just one operating point.",
    "Closer to 1.0 = consistently strong across confidence thresholds.",
    0.55
  );

  const y1 = y0 + 2.35;
  metricBlock(slide, x0, y1, colW,
    "mAP@50 / mAP@50:95",
    ["mAP@50:95 = avg AP over IoU", "  0.50 → 0.95 in steps of 0.05,", "  then averaged across classes"],
    "mAP@50:95 is the stricter score (rewards tight boxes); mAP@50 is more forgiving.",
    "A big gap (e.g. 0.44 vs 0.30) means boxes are roughly right but not tightly localized.",
    0.68
  );

  metricBlock(slide, x0 + colW + gap, y1, colW,
    "Mean IoU",
    ["Mean IoU = avg IoU of matched", "  (TP) pairs only"],
    "Measures localization tightness, ignoring misses entirely.",
    "Mean IoU > 0.85 indicates tight boxes.",
    0.55
  );

  // note panel filling the third slot at y1
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x0 + 2 * (colW + gap), y: y1, w: colW, h: 2.05, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  slide.addText("Computed via pycocotools (COCO standard)", {
    x: x0 + 2 * (colW + gap) + 0.18, y: y1 + 0.15, w: colW - 0.36, h: 0.4,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: NAVY, margin: 0,
  });
  slide.addText("All five metrics above are reported per dataset, per model, on the following slide.", {
    x: x0 + 2 * (colW + gap) + 0.18, y: y1 + 0.6, w: colW - 0.36, h: 1.3,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: CHARCOAL, margin: 0, lineSpacingMultiple: 1.15,
  });
}

// ============================================================
// SLIDE 4 — Traditional benchmark results table
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Traditional Benchmark — Results");
  footer(slide, 4);

  const headers = ["Dataset", "Model", "mAP@50", "mAP@50:95", "Precision", "Recall", "F1", "Mean IoU"];

  const rows = [
    ["DocLayNet", "DocLayoutYOLO", 0.436, 0.352, 0.588, 0.556, 0.571, 0.894],
    ["DocLayNet", "Nemotron-Parse", 0.439, 0.304, 0.780, 0.593, 0.674, 0.868],
    ["DocLayNet", "ADE-DPT2", 0.242, 0.144, 0.545, 0.333, 0.414, 0.859],
    ["PubLayNet", "DocLayoutYOLO", 0.545, 0.412, 0.686, 0.800, 0.738, 0.905],
    ["PubLayNet", "Nemotron-Parse", 0.383, 0.224, 0.818, 0.600, 0.692, 0.884],
    ["PubLayNet", "ADE-DPT2", 0.490, 0.275, 0.739, 0.567, 0.642, 0.848],
    ["DocBank", "DocLayoutYOLO", 0.016, 0.007, 0.233, 0.184, 0.206, 0.761],
    ["DocBank", "Nemotron-Parse", 0.015, 0.005, 0.231, 0.158, 0.187, 0.694],
    ["DocBank", "ADE-DPT2", 0.023, 0.008, 0.333, 0.184, 0.237, 0.691],
  ];

  const modelColor = (m) => m === "DocLayoutYOLO" ? BLUE : (m === "Nemotron-Parse" ? REDORANGE : PURPLE);

  // compute best per metric within each dataset group (cols index 2..7 in rows)
  const groups = { "DocLayNet": [0, 1, 2], "PubLayNet": [3, 4, 5], "DocBank": [6, 7, 8] };
  const bestMask = rows.map(() => [false, false, false, false, false, false]);
  Object.values(groups).forEach((idxs) => {
    for (let c = 0; c < 6; c++) {
      let bestVal = -Infinity, bestIdx = -1;
      idxs.forEach((i) => {
        const v = rows[i][2 + c];
        if (v > bestVal) { bestVal = v; bestIdx = i; }
      });
      bestMask[bestIdx][c] = true;
    }
  });

  const tableData = [headers.map((h) => headCell(h))];
  rows.forEach((r, i) => {
    const [ds, model, ...vals] = r;
    const rowArr = [];
    rowArr.push(cell(ds, { align: "left", fontSize: 9.5 }));
    rowArr.push({ text: model, options: { fontFace: BODY_FONT, fontSize: 9.5, bold: true, color: modelColor(model), align: "left", valign: "middle" } });
    vals.forEach((v, c) => {
      rowArr.push(cell(v.toFixed(3), { bold: bestMask[i][c], fontSize: 9.5, color: bestMask[i][c] ? NAVY : CHARCOAL }));
    });
    tableData.push(rowArr);
  });

  slide.addTable(tableData, {
    x: 0.5, y: 1.05, w: 12.33,
    colW: [1.55, 1.95, 1.4, 1.55, 1.4, 1.25, 1.0, 1.23],
    border: { pt: 0.75, color: RULE },
    rowH: 0.34,
    autoPage: false,
  });

  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 4.5, w: 12.33, h: 0.62, rectRadius: 0.05,
    fill: { color: WARN_BG }, line: { type: "none" },
  });
  slide.addText([
    { text: "Note: ", options: { bold: true, color: WARN_TX } },
    { text: "DocBank collapses for all 3 models (mAP@50 ≤ 0.023) — likely an annotation-granularity mismatch, not true model failure; flagged for investigation, not a real performance gap.", options: { color: WARN_TX } },
  ], {
    x: 0.75, y: 4.5, w: 11.85, h: 0.62,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.1,
  });

  slide.addText("Bold = best value per metric within each dataset group.", {
    x: 0.5, y: 5.25, w: 8, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9.5, italic: true, color: MUTED, margin: 0,
  });
}

// ============================================================
// SLIDE 5 — Custom benchmark metrics, Layers 1-2
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Custom Benchmark — Metrics (Layers 1–2)");
  footer(slide, 5);

  const colW = 3.9, gap = 0.27, x0 = 0.5, y0 = 1.12;

  metricBlock(slide, x0, y0, colW,
    "Layer 1 — Geometric Match",
    ["Hungarian-matched P / R / F1 / mean IoU", "  at IoU ≥ 0.50", "Class Acc = correct-label matches /", "  total matched pairs"],
    "Tests both \"did we find the region\" and \"did we label it right\" — separately.",
    "F1 > 0.60 good; Class Acc should track close to F1 — near 0 with healthy F1 signals a label-mapping bug, not real classification failure.",
    0.8
  );

  metricBlock(slide, x0 + colW + gap, y0, colW,
    "Layer 2 — COTe Coverage",
    ["Coverage = Area(GT ∩ anyPred) / Area(GT)"],
    "Measures completeness — how much of the true region was found at all.",
    "Coverage > 0.70 is good.",
    0.42
  );

  metricBlock(slide, x0 + 2 * (colW + gap), y0, colW,
    "Layer 2 — COTe Overlap",
    ["Overlap = Area(GT ∩ doubly-covered)", "  / Area(Pred)"],
    "Captures redundant double-coverage of ground truth by predictions.",
    "Lower is better; should stay small relative to Coverage.",
    0.55
  );

  const y1 = y0 + 2.45;
  metricBlock(slide, x0, y1, colW,
    "Layer 2 — COTe Trespass",
    ["Trespass = Area(Pred outside GT)", "  / Area(Pred)"],
    "How much of the predicted area is \"wasted\" outside any true region.",
    "Trespass < 0.15 is good.",
    0.55
  );

  metricBlock(slide, x0 + colW + gap, y1, colW,
    "Layer 2 — COTe Excess",
    ["Excess = Area(Pred) / Area(GT)"],
    "Over/under-detection ratio — total predicted area vs. total true area.",
    "Excess ≈ 1.0 is ideal (>1 over-detecting, <1 under-detecting).",
    0.42
  );

  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x0 + 2 * (colW + gap), y: y1, w: colW, h: 1.95, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  slide.addText("Pixel-mask based", {
    x: x0 + 2 * (colW + gap) + 0.18, y: y1 + 0.15, w: colW - 0.36, h: 0.32,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: NAVY, margin: 0,
  });
  slide.addText("Layer 2 metrics operate on rendered pixel masks rather than box coordinates, giving a finer-grained read on spatial quality than IoU alone.", {
    x: x0 + 2 * (colW + gap) + 0.18, y: y1 + 0.55, w: colW - 0.36, h: 1.3,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: CHARCOAL, margin: 0, lineSpacingMultiple: 1.15,
  });
}

// ============================================================
// SLIDE 6 — Custom benchmark metrics, Layers 3-4
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Custom Benchmark — Metrics (Layers 3–4)");
  footer(slide, 6);

  // Layer 3 block (wide, spans full width)
  const x0 = 0.5, y0 = 1.05, fullW = 12.33;
  slide.addText("Layer 3 — LED Error Taxonomy", {
    x: x0, y: y0, w: fullW, h: 0.3,
    fontFace: BODY_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
  });
  slide.addText("Thresholds: IoU ≥ 0.10 \"related\" · ≥ 0.50 \"matched\" · ≥ 0.85 \"duplicate\"", {
    x: x0, y: y0 + 0.32, w: fullW, h: 0.26,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED, margin: 0,
  });

  const cats = [
    ["Missing", "GT with no related pred"],
    ["Hallucination", "Pred with no related GT"],
    ["Size-Error", "Matched, 0.10 ≤ IoU < 0.50"],
    ["Misclassification", "Matched, IoU ≥ 0.50, wrong label"],
    ["Split", "1 GT → multiple preds"],
    ["Merge", "Multiple GT → 1 pred"],
    ["Duplicate", "2 preds, mutual IoU ≥ 0.85"],
    ["Overlap-Pred", "2 preds, 0.10 ≤ mutual IoU < 0.85"],
  ];
  const catTable = [
    [headCell("Category"), headCell("Definition"), headCell("Category"), headCell("Definition")],
  ];
  for (let i = 0; i < 4; i++) {
    catTable.push([
      cell(cats[i][0], { align: "left", bold: true, fontSize: 9.5 }),
      cell(cats[i][1], { align: "left", fontSize: 9.5 }),
      cell(cats[i + 4][0], { align: "left", bold: true, fontSize: 9.5 }),
      cell(cats[i + 4][1], { align: "left", fontSize: 9.5 }),
    ]);
  }
  slide.addTable(catTable, {
    x: x0, y: y0 + 0.62, w: fullW, colW: [1.7, 4.46, 1.7, 4.46],
    border: { pt: 0.75, color: RULE }, rowH: 0.3, autoPage: false,
  });

  slide.addText([
    { text: "Significance: ", options: { bold: true, color: CHARCOAL } },
    { text: "Turns one aggregate error rate into \"what kind of mistake is happening\" — directly actionable.   ", options: { color: CHARCOAL } },
    { text: "Threshold/Remark: ", options: { bold: true, color: CHARCOAL } },
    { text: "Missing + Hallucination < 10% = healthy; Misclassification < 10% = good; anything dominating the breakdown points straight at the fix needed.", options: { color: CHARCOAL } },
  ], {
    x: x0, y: y0 + 1.95, w: fullW, h: 0.55,
    fontFace: BODY_FONT, fontSize: 10, margin: 0, lineSpacingMultiple: 1.1,
  });

  // Layer 4 block
  const y1 = y0 + 2.65;
  const colW = 5.95, gap = 0.43;
  metricBlock(slide, x0, y1, colW,
    "Layer 4 — Reading Order: ROKT",
    ["ROKT (Kendall's τ) = (C − D) / (n(n−1)/2)", "  on Hungarian-matched (IoU ≥ 0.30)", "  GT vs predicted order"],
    "Tests whether the model reconstructs the correct reading sequence, not just the right boxes.",
    "τ > 0.70 = strong agreement; 1.0 is perfect.",
    0.68
  );
  metricBlock(slide, x0 + colW + gap, y1, colW,
    "Layer 4 — Reading Order: ROA",
    ["ROA = 1 − (adjacent inversions) / (n − 1)"],
    "Measures local reading-order correctness via adjacent-pair inversions.",
    "1.0 is perfect; lower values indicate more local order swaps.",
    0.42
  );
}

// ============================================================
// SLIDE 7 — Custom benchmark results: Layer 1
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Custom Benchmark — Results: Layer 1");
  footer(slide, 7);

  const headers = ["Page", "Model", "Precision", "Recall", "F1", "Mean IoU", "Class Acc"];
  const rows = [
    ["7", "DocLayoutYOLO", 0.900, 0.900, 0.900, 1.000, 0.778],
    ["7", "Nemotron-Parse", 0.900, 0.900, 0.900, 0.910, 0.000],
    ["15", "DocLayoutYOLO", 1.000, 0.952, 0.976, 0.997, 0.350],
    ["15", "Nemotron-Parse", 0.944, 0.810, 0.872, 0.771, 0.000],
    ["17", "DocLayoutYOLO", 0.800, 0.800, 0.800, 0.920, 0.750],
    ["17", "Nemotron-Parse", 0.500, 0.400, 0.444, 0.818, 0.000],
    ["19", "DocLayoutYOLO", 0.250, 0.077, 0.118, 0.833, 0.000],
    ["19", "Nemotron-Parse", 0.846, 0.846, 0.846, 0.723, 0.000],
    ["25", "DocLayoutYOLO", 1.000, 1.000, 1.000, 1.000, 0.643],
    ["25", "Nemotron-Parse", 0.867, 0.929, 0.897, 0.835, 0.000],
  ];
  const meanRows = [
    ["Mean", "DocLayoutYOLO", 0.790, 0.746, 0.759, 0.950, 0.504],
    ["Mean", "Nemotron-Parse", 0.811, 0.777, 0.792, 0.811, 0.000],
  ];

  const modelColor = (m) => m === "DocLayoutYOLO" ? BLUE : REDORANGE;

  // best per page-pair (between the two models) for each metric col
  const bestMask = rows.map(() => [false, false, false, false, false]);
  for (let i = 0; i < rows.length; i += 2) {
    for (let c = 0; c < 5; c++) {
      const v0 = rows[i][2 + c], v1 = rows[i + 1][2 + c];
      if (v0 >= v1) bestMask[i][c] = true; else bestMask[i + 1][c] = true;
    }
  }
  const meanBest = [[false, false, false, false, false], [false, false, false, false, false]];
  for (let c = 0; c < 5; c++) {
    if (meanRows[0][2 + c] >= meanRows[1][2 + c]) meanBest[0][c] = true; else meanBest[1][c] = true;
  }

  const tableData = [headers.map((h) => headCell(h))];
  rows.forEach((r, i) => {
    const [pg, model, ...vals] = r;
    const rowArr = [cell(pg, { fontSize: 9.5 })];
    rowArr.push({ text: model, options: { fontFace: BODY_FONT, fontSize: 9.5, bold: true, color: modelColor(model), align: "left", valign: "middle" } });
    vals.forEach((v, c) => {
      rowArr.push(cell(v.toFixed(3), { bold: bestMask[i][c], fontSize: 9.5 }));
    });
    tableData.push(rowArr);
  });
  meanRows.forEach((r, i) => {
    const [pg, model, ...vals] = r;
    const rowArr = [cell(pg, { fontSize: 9.5, bold: true })];
    rowArr.push({ text: model, options: { fontFace: BODY_FONT, fontSize: 9.5, bold: true, color: modelColor(model), align: "left", valign: "middle", fill: { color: PANEL } } });
    vals.forEach((v, c) => {
      rowArr.push(cell(v.toFixed(3), { bold: true, fontSize: 9.5, fill: { color: PANEL } }));
    });
    tableData.push(rowArr);
  });

  slide.addTable(tableData, {
    x: 0.5, y: 1.0, w: 12.33,
    colW: [1.0, 2.4, 1.85, 1.7, 1.5, 1.7, 1.45, 0.73].slice(0, 7),
    border: { pt: 0.75, color: RULE },
    rowH: 0.29,
    autoPage: false,
  });

  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 4.45, w: 12.33, h: 0.65, rectRadius: 0.05,
    fill: { color: WARN_BG }, line: { type: "none" },
  });
  slide.addText([
    { text: "Note: ", options: { bold: true, color: WARN_TX } },
    { text: "Nemotron's Class Accuracy is 0.000 on every page despite healthy F1 — almost certainly a label-taxonomy normalization bug, flagged for engineering follow-up, not read as a real result.", options: { color: WARN_TX } },
  ], {
    x: 0.75, y: 4.45, w: 11.85, h: 0.65,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.1,
  });
}

// ============================================================
// SLIDE 8 — Custom benchmark results: Layers 2-3
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Custom Benchmark — Results: Layers 2–3");
  footer(slide, 8);

  // Layer 2 table (left)
  const l2headers = ["Page", "Model", "Coverage", "Overlap", "Trespass", "Excess"];
  const l2rows = [
    ["7", "DocLayoutYOLO", 0.952, 0.000, 0.000, 0.952],
    ["7", "Nemotron-Parse", 0.988, 0.000, 0.049, 1.039],
    ["15", "DocLayoutYOLO", 0.940, 0.000, 0.000, 0.940],
    ["15", "Nemotron-Parse", 0.165, 0.000, 0.182, 0.202],
    ["17", "DocLayoutYOLO", 0.948, 0.000, 0.011, 0.959],
    ["17", "Nemotron-Parse", 0.503, 0.000, 0.037, 0.522],
    ["19", "DocLayoutYOLO", 0.928, 0.000, 0.396, 1.535],
    ["19", "Nemotron-Parse", 0.874, 0.000, 0.046, 0.916],
    ["25", "DocLayoutYOLO", 1.000, 0.000, 0.000, 1.000],
    ["25", "Nemotron-Parse", 0.855, 0.000, 0.291, 1.206],
  ];
  const l2mean = [
    ["Mean", "DocLayoutYOLO", 0.954, 0.000, 0.081, 1.077],
    ["Mean", "Nemotron-Parse", 0.677, 0.000, 0.121, 0.777],
  ];
  const modelColor = (m) => m === "DocLayoutYOLO" ? BLUE : REDORANGE;

  const l2Data = [l2headers.map((h) => headCell(h))];
  l2rows.forEach((r) => {
    const [pg, model, ...vals] = r;
    const rowArr = [cell(pg, { fontSize: 8.5 })];
    rowArr.push({ text: model.replace("DocLayoutYOLO", "DLY").replace("Nemotron-Parse", "Nemotron"), options: { fontFace: BODY_FONT, fontSize: 8.5, bold: true, color: modelColor(model), align: "left", valign: "middle" } });
    vals.forEach((v) => rowArr.push(cell(v.toFixed(3), { fontSize: 8.5 })));
    l2Data.push(rowArr);
  });
  l2mean.forEach((r) => {
    const [pg, model, ...vals] = r;
    const rowArr = [cell(pg, { fontSize: 8.5, bold: true, fill: { color: PANEL } })];
    rowArr.push({ text: model.replace("DocLayoutYOLO", "DLY").replace("Nemotron-Parse", "Nemotron"), options: { fontFace: BODY_FONT, fontSize: 8.5, bold: true, color: modelColor(model), align: "left", valign: "middle", fill: { color: PANEL } } });
    vals.forEach((v) => rowArr.push(cell(v.toFixed(3), { fontSize: 8.5, bold: true, fill: { color: PANEL } })));
    l2Data.push(rowArr);
  });

  slide.addText("Layer 2 — Spatial Quality (COTe)", {
    x: 0.5, y: 0.98, w: 6.1, h: 0.28,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  slide.addTable(l2Data, {
    x: 0.5, y: 1.3, w: 6.1, colW: [0.65, 1.45, 1.0, 1.0, 1.0, 1.0],
    border: { pt: 0.6, color: RULE }, rowH: 0.275, autoPage: false,
  });

  // Layer 3 table (right)
  const l3rows = [
    ["Missing", "11 (20.8%)", "7 (11.7%)"],
    ["Hallucination", "3 (5.7%)", "5 (8.3%)"],
    ["Size-Error", "2 (3.8%)", "3 (5.0%)"],
    ["Split", "0 (0.0%)", "0 (0.0%)"],
    ["Merge", "1 (1.9%)", "1 (1.7%)"],
    ["Overlap-Pred", "1 (1.9%)", "0 (0.0%)"],
    ["Duplicate", "0 (0.0%)", "0 (0.0%)"],
    ["Misclassification", "22 (41.5%)", "52 (86.7%)"],
  ];
  slide.addText("Layer 3 — LED Error Taxonomy", {
    x: 6.9, y: 0.98, w: 5.93, h: 0.28,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  slide.addText("Counts out of 53 DocLayoutYOLO / 60 Nemotron-Parse total detections", {
    x: 6.9, y: 1.26, w: 5.93, h: 0.22,
    fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, margin: 0,
  });

  const l3Data = [
    [headCell("Error type"), { text: "DocLayoutYOLO", options: { ...headerOpts, color: WHITE } }, { text: "Nemotron-Parse", options: { ...headerOpts, color: WHITE } }],
  ];
  l3rows.forEach((r) => {
    const isMiscls = r[0] === "Misclassification";
    l3Data.push([
      cell(r[0], { align: "left", fontSize: 9.5, bold: isMiscls }),
      cell(r[1], { fontSize: 9.5, color: BLUE, bold: isMiscls, fill: isMiscls ? { color: BAD_BG } : undefined }),
      cell(r[2], { fontSize: 9.5, color: REDORANGE, bold: isMiscls, fill: isMiscls ? { color: BAD_BG } : undefined }),
    ]);
  });
  slide.addTable(l3Data, {
    x: 6.9, y: 1.54, w: 5.93, colW: [2.43, 1.75, 1.75],
    border: { pt: 0.6, color: RULE }, rowH: 0.32, autoPage: false,
  });

  // bottom takeaway strip
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 4.55, w: 12.33, h: 0.85, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  slide.addText([
    { text: "Reading the two layers together: ", options: { bold: true, color: NAVY } },
    { text: "DocLayoutYOLO holds higher Coverage with lower Trespass on most pages, while Nemotron's Layer 3 breakdown is dominated by Misclassification (86.7%) — consistent with the Class Accuracy bug seen in Layer 1.", options: { color: CHARCOAL } },
  ], {
    x: 0.75, y: 4.55, w: 11.85, h: 0.85,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.12,
  });
}

// ============================================================
// SLIDE 9 — Layer 4 + Consolidated scorecard
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Layer 4 Results & Consolidated Scorecard");
  footer(slide, 9);

  // Layer 4 mini panel
  slide.addText("Layer 4 — Reading Order", {
    x: 0.5, y: 1.0, w: 5.6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  const l4Data = [
    [headCell("Model"), headCell("Kendall's τ"), headCell("ROA")],
    [{ text: "DocLayoutYOLO", options: { fontFace: BODY_FONT, fontSize: 10.5, bold: true, color: BLUE, align: "left", valign: "middle" } }, cell("1.000", { bold: true }), cell("1.000", { bold: true })],
    [{ text: "Nemotron-Parse", options: { fontFace: BODY_FONT, fontSize: 10.5, bold: true, color: REDORANGE, align: "left", valign: "middle" } }, cell("1.000", { bold: true }), cell("1.000", { bold: true })],
  ];
  slide.addTable(l4Data, {
    x: 0.5, y: 1.32, w: 5.6, colW: [2.4, 1.6, 1.6],
    border: { pt: 0.75, color: RULE }, rowH: 0.4, autoPage: false,
  });
  slide.addText("Perfect reading-order agreement across all pages for both models.", {
    x: 0.5, y: 2.55, w: 5.6, h: 0.45,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: CHARCOAL, margin: 0, lineSpacingMultiple: 1.1,
  });

  // Scorecard table (right, taller)
  slide.addText("Consolidated Scorecard", {
    x: 6.45, y: 1.0, w: 6.38, h: 0.3,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });

  function statusCell(val, status) {
    const map = {
      good: { bg: GOOD_BG, tx: GOOD_TX, mark: "✅" },
      warn: { bg: WARN_BG, tx: WARN_TX, mark: "⚠️" },
      bad: { bg: BAD_BG, tx: BAD_TX, mark: "❌" },
    };
    const m = map[status];
    return cell(`${val}  ${m.mark}`, { fill: { color: m.bg }, color: m.tx, bold: true, fontSize: 10 });
  }

  const scoreHeaders = ["Layer", "Metric", "Threshold", "DocLayoutYOLO", "Nemotron-Parse"];
  const scoreData = [scoreHeaders.map((h) => headCell(h))];
  scoreData.push([cell("1"), cell("F1 @ IoU 0.50", { align: "left", fontSize: 9.5 }), cell(">0.60", { fontSize: 9.5 }), statusCell("0.759", "good"), statusCell("0.792", "good")]);
  scoreData.push([cell("2"), cell("Coverage", { align: "left", fontSize: 9.5 }), cell(">0.70", { fontSize: 9.5 }), statusCell("0.954", "good"), statusCell("0.677", "warn")]);
  scoreData.push([cell("2"), cell("Trespass", { align: "left", fontSize: 9.5 }), cell("<0.15", { fontSize: 9.5 }), statusCell("0.081", "good"), statusCell("0.121", "good")]);
  scoreData.push([cell("3"), cell("Missing + Hallucination", { align: "left", fontSize: 9.5 }), cell("<10%", { fontSize: 9.5 }), statusCell("26.5%", "warn"), statusCell("20.0%", "warn")]);
  scoreData.push([cell("3"), cell("Misclassification", { align: "left", fontSize: 9.5 }), cell("<10%", { fontSize: 9.5 }), statusCell("41.5%", "bad"), statusCell("86.7%", "bad")]);
  scoreData.push([cell("4"), cell("Kendall's τ", { align: "left", fontSize: 9.5 }), cell(">0.70", { fontSize: 9.5 }), statusCell("1.000", "good"), statusCell("1.000", "good")]);

  slide.addTable(scoreData, {
    x: 6.45, y: 1.32, w: 6.38, colW: [0.55, 1.9, 0.93, 1.5, 1.5],
    border: { pt: 0.75, color: RULE }, rowH: 0.42, autoPage: false,
  });

  // bottom strip
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 5.85, w: 12.33, h: 0.85, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  slide.addText([
    { text: "Read the scorecard as: ", options: { bold: true, color: NAVY } },
    { text: "geometric localization and reading order are solid for both models; the breakdown is concentrated entirely in classification accuracy, most severely for Nemotron-Parse.", options: { color: CHARCOAL } },
  ], {
    x: 0.75, y: 5.85, w: 11.85, h: 0.85,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.12,
  });
}

// ============================================================
// SLIDE 10 — Key findings & next steps
// ============================================================
{
  const slide = pres.addSlide();
  slide.background = { color: WHITE };
  slideTitle(slide, "Key Findings & Next Steps");
  footer(slide, 10);

  slide.addText("Key Findings", {
    x: 0.5, y: 1.05, w: 5.9, h: 0.32,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: NAVY, margin: 0,
  });

  const findings = [
    "Both models localize regions well and nail reading order — Layer 4 reading-order metrics hit a perfect 1.000 for both.",
    "Classification is the weak point for both models, and dramatically worse for Nemotron-Parse — likely a taxonomy-mapping bug that needs fixing before re-judging it.",
    "DocLayoutYOLO is far cheaper to run (CPU-capable, ~20M params) and processed all 30 document pages, vs. Nemotron's 5.",
    "Sample sizes in both benchmarks are small (3 pages/dataset, 5 GT pages) — treat results as directional, not definitive.",
  ];

  slide.addText(
    findings.map((f, i) => ({ text: f, options: { bullet: { code: "2022" }, breakLine: i < findings.length - 1, color: CHARCOAL, paraSpaceAfter: 12 } })),
    {
      x: 0.5, y: 1.45, w: 5.9, h: 4.6,
      fontFace: BODY_FONT, fontSize: 13, margin: 0, valign: "top", lineSpacingMultiple: 1.18,
    }
  );

  slide.addText("Next Steps", {
    x: 6.85, y: 1.05, w: 6.0, h: 0.32,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: NAVY, margin: 0,
  });

  const nextSteps = [
    "Fix the Nemotron-Parse label-taxonomy mapping bug.",
    "Re-run both benchmarks with larger samples for statistical confidence.",
    "Add ADE-DPT2 to the custom benchmark for full 3-way parity.",
  ];

  slide.addText(
    nextSteps.map((f, i) => ({ text: f, options: { bullet: { code: "2022" }, breakLine: i < nextSteps.length - 1, color: CHARCOAL, paraSpaceAfter: 14 } })),
    {
      x: 6.85, y: 1.45, w: 6.0, h: 2.6,
      fontFace: BODY_FONT, fontSize: 13, margin: 0, valign: "top", lineSpacingMultiple: 1.18,
    }
  );

  // legend recap panel
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.85, y: 4.3, w: 6.0, h: 1.75, rectRadius: 0.06,
    fill: { color: PANEL }, line: { color: RULE, width: 0.75 },
  });
  slide.addText("Models Referenced", {
    x: 7.1, y: 4.46, w: 5.5, h: 0.3,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: NAVY, margin: 0,
  });
  modelChip(slide, 7.1, 4.85, "DocLayoutYOLO — fast, cheap, CPU-capable", BLUE);
  modelChip(slide, 7.1, 5.2, "Nemotron-Parse-v1.1 — strong but mislabeled classes", REDORANGE);
  modelChip(slide, 7.1, 5.55, "ADE-DPT2 — traditional benchmark only, to date", PURPLE);
}

pres.writeFile({ fileName: "results_review.pptx" }).then(() => {
  console.log("Deck written.");
});
