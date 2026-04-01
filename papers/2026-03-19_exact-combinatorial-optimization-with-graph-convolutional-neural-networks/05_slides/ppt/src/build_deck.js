const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { latexToSvgDataUri } = require("./pptxgenjs_helpers/latex");
const { warnIfSlideHasOverlaps, warnIfSlideElementsOutOfBounds } = require("./pptxgenjs_helpers/layout");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "PaperReadand Report";
pptx.subject = "Research presentation deck";
pptx.title = "Exact Combinatorial Optimization with GCNNs";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const COLORS = {
  bg: "F8FAFC",
  surface: "FFFFFF",
  text: "0F172A",
  accent: "0F766E",
  accentSoft: "CCFBF1",
  accentDeep: "115E59",
  muted: "475569",
  line: "CBD5E1",
  lineDark: "94A3B8",
  warm: "FFF7ED",
  warmLine: "FDBA74",
  green: "DCFCE7",
  greenLine: "22C55E",
  red: "FEE2E2",
  redLine: "EF4444",
  grayFill: "E2E8F0",
};

const SLIDE_W = 13.333;

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function addBase(slide, index, title, subtitle = "") {
  slide.background = { color: COLORS.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: SLIDE_W,
    h: 0.68,
    line: { color: COLORS.bg, transparency: 100 },
    fill: { color: COLORS.bg },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.55,
    y: 0.18,
    w: 0.78,
    h: 0.3,
    rectRadius: 0.05,
    line: { color: COLORS.accent, pt: 0.8 },
    fill: { color: COLORS.accentSoft },
  });
  slide.addText(String(index).padStart(2, "0"), {
    x: 0.73,
    y: 0.235,
    w: 0.4,
    h: 0.12,
    fontFace: "Aptos",
    fontSize: 12,
    bold: true,
    color: COLORS.accentDeep,
    margin: 0,
    align: "center",
  });
  slide.addText(title, {
    x: 1.55,
    y: 0.16,
    w: 8.8,
    h: 0.26,
    fontFace: "Aptos Display",
    fontSize: 24,
    bold: true,
    color: COLORS.text,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 1.57,
      y: 0.43,
      w: 9.8,
      h: 0.14,
      fontFace: "Aptos",
      fontSize: 9.5,
      color: COLORS.muted,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.58,
    y: 0.72,
    w: 12.12,
    h: 0,
    line: { color: COLORS.line, pt: 1 },
  });
}

function addFooter(slide, evidence) {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.58,
    y: 7.02,
    w: 12.12,
    h: 0,
    line: { color: COLORS.line, pt: 1 },
  });
  slide.addText(`Evidence: ${evidence}`, {
    x: 0.62,
    y: 7.08,
    w: 8.2,
    h: 0.14,
    fontFace: "Aptos",
    fontSize: 9,
    color: COLORS.muted,
    margin: 0,
  });
  slide.addText("PaperReadand Report | exact-combinatorial-optimization-with-gcnn", {
    x: 10.25,
    y: 7.08,
    w: 2.2,
    h: 0.14,
    fontFace: "Aptos",
    fontSize: 9,
    color: COLORS.muted,
    align: "right",
    margin: 0,
  });
}

function addPanel(slide, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    line: { color: opts.line || COLORS.line, pt: opts.pt || 1 },
    fill: { color: opts.fill || COLORS.surface },
  });
}

function addLabel(slide, text, x, y, w, fill, color = COLORS.accentDeep) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.28,
    rectRadius: 0.05,
    line: { color: fill, pt: 0.8 },
    fill: { color: fill },
  });
  slide.addText(text, {
    x: x + 0.08,
    y: y + 0.07,
    w: w - 0.16,
    h: 0.1,
    fontFace: "Aptos",
    fontSize: 10.5,
    bold: true,
    color,
    margin: 0,
    align: "center",
  });
}

function addBullets(slide, bullets, x, y, w, h, fontSize = 17) {
  const runs = [];
  bullets.forEach((text) => {
    runs.push({
      text,
      options: { bullet: { indent: 12 } },
    });
  });
  slide.addText(runs, {
    x,
    y,
    w,
    h,
    fontFace: "Aptos",
    fontSize,
    color: COLORS.text,
    breakLine: true,
    paraSpaceAfterPt: 9,
    valign: "top",
    margin: 0,
  });
}

function addBodyText(slide, text, x, y, w, h, fontSize = 15, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: opts.fontFace || "Aptos",
    fontSize,
    bold: opts.bold || false,
    color: opts.color || COLORS.text,
    italic: opts.italic || false,
    align: opts.align || "left",
    valign: opts.valign || "top",
    margin: opts.margin !== undefined ? opts.margin : 0,
  });
}

function addMetricCard(slide, x, y, w, h, title, value, detail, fill = COLORS.surface) {
  addPanel(slide, x, y, w, h, { fill, line: COLORS.line });
  addBodyText(slide, title, x + 0.16, y + 0.12, w - 0.32, 0.18, 10.5, {
    bold: true,
    color: COLORS.muted,
  });
  addBodyText(slide, value, x + 0.16, y + 0.33, w - 0.32, 0.28, 20, {
    bold: true,
    color: COLORS.accentDeep,
  });
  addBodyText(slide, detail, x + 0.16, y + 0.68, w - 0.32, h - 0.82, 10.5, {
    color: COLORS.text,
  });
}

function addProcessStep(slide, x, y, w, h, title, body, fill) {
  addPanel(slide, x, y, w, h, { fill, line: COLORS.lineDark });
  addBodyText(slide, title, x + 0.14, y + 0.12, w - 0.28, 0.18, 11, {
    bold: true,
  });
  addBodyText(slide, body, x + 0.14, y + 0.38, w - 0.28, h - 0.48, 10.5, {
    color: COLORS.muted,
  });
}

function addArrow(slide, x, y, w) {
  slide.addShape(pptx.ShapeType.chevron, {
    x,
    y,
    w,
    h: 0.26,
    line: { color: COLORS.lineDark, pt: 0.8 },
    fill: { color: COLORS.grayFill },
  });
}

function addEquation(slide, latex, x, y, w, h) {
  const data = latexToSvgDataUri(latex, true);
  slide.addImage({
    data,
    x,
    y,
    w,
    h,
  });
}

function finalizeSlide(slide, notes) {
  slide.addNotes(notes);
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function addComparisonBar(slide, x, y, label, gcnn, baseline) {
  addBodyText(slide, label, x, y - 0.02, 1.55, 0.16, 10.5, {
    bold: true,
    color: COLORS.text,
  });
  slide.addText(`GCNN ${gcnn.toFixed(1)}`, {
    x: x + 1.6,
    y: y - 0.02,
    w: 0.72,
    h: 0.15,
    fontFace: "Aptos",
    fontSize: 10,
    bold: true,
    color: COLORS.accentDeep,
    margin: 0,
  });
  slide.addText(`Prior ${baseline.toFixed(1)}`, {
    x: x + 1.6,
    y: y + 0.18,
    w: 0.72,
    h: 0.15,
    fontFace: "Aptos",
    fontSize: 10,
    color: COLORS.muted,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 2.35,
    y,
    w: 1.55,
    h: 0.1,
    line: { color: COLORS.grayFill, transparency: 100 },
    fill: { color: COLORS.grayFill },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 2.35,
    y,
    w: (1.55 * gcnn) / 100,
    h: 0.1,
    line: { color: COLORS.accent, transparency: 100 },
    fill: { color: COLORS.accent },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 2.35,
    y: y + 0.2,
    w: 1.55,
    h: 0.1,
    line: { color: COLORS.grayFill, transparency: 100 },
    fill: { color: COLORS.grayFill },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 2.35,
    y: y + 0.2,
    w: (1.55 * baseline) / 100,
    h: 0.1,
    line: { color: COLORS.lineDark, transparency: 100 },
    fill: { color: COLORS.lineDark },
  });
}

function slide1() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    1,
    "Exact Combinatorial Optimization with Graph Convolutional Neural Networks",
    "NeurIPS 2019 | Maxime Gasse, Didier Chetelat, Nicola Ferroni, Laurent Charlin, Andrea Lodi"
  );
  addPanel(slide, 0.68, 1.08, 7.15, 4.95, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Core Takeaway", 0.92, 1.26, 1.42, COLORS.accentSoft);
  addBodyText(slide, "Learned branching inside an exact MILP solver", 0.95, 1.78, 5.9, 0.46, 23, {
    bold: true,
  });
  addBodyText(
    slide,
    "A bipartite GCNN imitates strong branching decisions, improves over prior learning-to-branch baselines, and still transfers to instances larger than those seen during training.",
    0.95,
    2.32,
    6.15,
    0.95,
    15,
    { color: COLORS.muted }
  );
  addBullets(
    slide,
    [
      "Task: variable selection in branch-and-bound for mixed-integer linear programs.",
      "Representation: current LP state as a variable-constraint bipartite graph.",
      "Learning setup: behavioral cloning from strong branching expert actions.",
      "Scope: brancher only; SCIP still handles LP solves, cuts, heuristics, and exact search.",
    ],
    0.95,
    3.35,
    6.2,
    2.0,
    14.5
  );

  addMetricCard(
    slide,
    8.22,
    1.16,
    2.18,
    1.2,
    "Problem",
    "MILP branching",
    "Choose the next branching variable in branch-and-bound.",
    COLORS.warm
  );
  addMetricCard(
    slide,
    10.55,
    1.16,
    2.08,
    1.2,
    "Expert",
    "Strong branching",
    "High-quality but computationally expensive per-node decisions.",
    COLORS.green
  );
  addMetricCard(
    slide,
    8.22,
    2.56,
    4.41,
    1.28,
    "Claim",
    "Better ML brancher",
    "Outperforms earlier ML baselines and can beat SCIP's default branching rule on large problems.",
    COLORS.surface
  );
  addPanel(slide, 8.22, 4.08, 4.41, 1.72, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Position in the solver", 8.46, 4.24, 1.9, COLORS.accentSoft);
  addProcessStep(slide, 8.44, 4.68, 1.05, 0.74, "MILP", "instance", COLORS.warm);
  addArrow(slide, 9.6, 4.94, 0.44);
  addProcessStep(slide, 10.08, 4.68, 1.18, 0.74, "B&B", "search tree", COLORS.surface);
  addArrow(slide, 11.38, 4.94, 0.44);
  addProcessStep(slide, 11.86, 4.68, 0.56, 0.74, "GCNN", "brancher", COLORS.green);

  addFooter(slide, "Abstract; Section 1; Section 5.2");
  finalizeSlide(
    slide,
    [
      "This paper learns a branching policy for MILP branch-and-bound, not an end-to-end neural solver.",
      "The learned component replaces only variable selection inside SCIP.",
      "Evidence: Abstract; Section 1 Introduction; Section 5.2 Comparative experiment.",
    ].join("\n")
  );
}

function slide2() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    2,
    "Research Question and Why It Matters",
    "Decision quality versus inference cost inside exact branch-and-bound"
  );
  addPanel(slide, 0.68, 1.02, 6.2, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Questions", 0.9, 1.18, 1.02, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Can a learned policy approximate strong branching quality without paying its per-node cost?",
      "Can MILP search states be encoded without large hand-crafted feature sets?",
      "Can the learned policy transfer to larger instances than seen during training?",
      "Will better imitation accuracy inside the brancher translate into fewer B&B nodes and lower solve time?",
    ],
    0.96,
    1.72,
    5.5,
    3.0,
    15.5
  );
  addPanel(slide, 0.98, 4.95, 5.56, 1.2, { fill: COLORS.warm, line: COLORS.warmLine });
  addBodyText(
    slide,
    "Practical goal: improve an exact solver on a family of related MILPs without rewriting the full search algorithm.",
    1.18,
    5.23,
    5.1,
    0.6,
    13.5,
    { bold: true }
  );

  addPanel(slide, 7.15, 1.02, 5.48, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Quality / Cost Tradeoff", 7.38, 1.18, 1.92, COLORS.accentSoft);
  slide.addShape(pptx.ShapeType.line, {
    x: 7.82,
    y: 5.92,
    w: 3.7,
    h: 0,
    line: { color: COLORS.lineDark, pt: 1.3, beginArrowType: "none", endArrowType: "triangle" },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 7.82,
    y: 5.92,
    w: 0,
    h: -3.82,
    line: { color: COLORS.lineDark, pt: 1.3, beginArrowType: "none", endArrowType: "triangle" },
  });
  addBodyText(slide, "Inference cost", 10.52, 6.02, 1.0, 0.16, 10.5, { color: COLORS.muted });
  addBodyText(slide, "Decision quality", 7.28, 2.02, 1.0, 0.16, 10.5, {
    color: COLORS.muted,
    bold: true,
  });
  addProcessStep(slide, 8.16, 5.06, 1.66, 0.82, "Default heuristics", "fast, hand-designed,\ngeneral-purpose", COLORS.warm);
  addProcessStep(slide, 10.04, 2.58, 1.66, 0.88, "Strong branching", "excellent choices,\nslow per node", COLORS.red);
  addProcessStep(slide, 8.98, 3.62, 1.74, 0.86, "Target GCNN policy", "expert-like quality\nat lower cost", COLORS.green);
  addBodyText(
    slide,
    "The paper measures success inside a full SCIP solver, not on an isolated classifier benchmark.",
    7.52,
    6.26,
    4.66,
    0.42,
    11.5,
    { color: COLORS.muted }
  );

  addFooter(slide, "Section 1; Section 3.3; Section 5.1");
  finalizeSlide(
    slide,
    [
      "This slide frames the paper as a tradeoff between expensive expert decisions and usable solver-time decisions.",
      "The evaluation target is not just imitation accuracy but downstream solve time and node count.",
      "Evidence: Section 1; Section 3.3 MDP formulation; Section 5.1 evaluation setup.",
    ].join("\n")
  );
}

function slide3() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    3,
    "Method at a Glance",
    "Figure 2 distilled into one state -> policy -> branch-action pipeline"
  );
  addPanel(slide, 0.68, 1.02, 12.0, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Pipeline", 0.92, 1.2, 0.82, COLORS.accentSoft);

  const steps = [
    ["State", "Current B&B node\nLP relaxation"],
    ["Graph", "Bipartite state\ns_t = (G, C, E, V)"],
    ["Policy", "GCNN\npi_theta(a | s_t)"],
    ["Output", "Masked scores over\ncandidate variables"],
    ["Action", "Branch on the\nhighest-ranked variable"],
  ];

  let x = 0.98;
  steps.forEach((step, idx) => {
    addProcessStep(
      slide,
      x,
      2.02,
      idx === 2 ? 2.15 : 1.85,
      1.18,
      step[0],
      step[1],
      idx === 2 ? COLORS.green : COLORS.surface
    );
    if (idx < steps.length - 1) {
      addArrow(slide, x + (idx === 2 ? 2.27 : 1.97), 2.48, 0.44);
    }
    x += idx === 2 ? 2.72 : 2.32;
  });

  addBullets(
    slide,
    [
      "Use the LP state at a B&B node, not the entire solver history.",
      "Exploit the natural variable-constraint graph instead of relying on heavy manual feature engineering.",
      "Train on expert actions directly, rather than predicting strong branching scores or rankings.",
      "Keep the rest of SCIP untouched so the policy plugs into a realistic exact-solver workflow.",
    ],
    1.0,
    3.72,
    6.2,
    2.2,
    14.5
  );

  addPanel(slide, 7.68, 3.56, 4.55, 2.18, { fill: COLORS.warm, line: COLORS.warmLine });
  addLabel(slide, "Why the formulation matters", 7.96, 3.78, 1.84, COLORS.warm);
  addBodyText(slide, "Prior work often imitated strong branching via score regression or ranking.", 7.96, 4.2, 3.95, 0.48, 12.5);
  addBodyText(slide, "This paper instead uses action classification, which makes the supervision target cleaner and closer to the deployed policy.", 7.96, 4.72, 3.95, 0.64, 12.5, {
    bold: true,
  });

  addFooter(slide, "Figure 2; Section 4.1; Section 4.2; Section 4.3");
  finalizeSlide(
    slide,
    [
      "The important change here is not just using a GNN, but using the bipartite MILP graph and direct action imitation.",
      "The deployed policy only decides which variable to branch on next.",
      "Evidence: Figure 2; Section 4.1; Section 4.2; Section 4.3.",
    ].join("\n")
  );
}

function slide4() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    4,
    "Dataset and Supervision",
    "State-action pairs extracted from branch-and-bound using a side-effect-free strong branching oracle"
  );
  addPanel(slide, 0.68, 1.02, 6.1, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Supervision design", 0.9, 1.18, 1.42, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Four benchmark families: set covering, combinatorial auction, capacitated facility location, and maximum independent set.",
      "Per benchmark: 100,000 training samples and 20,000 validation samples, with matching test collection.",
      "Each label is the expert's chosen branching variable at a B&B node.",
      "The label is a discrete action, not a tree-size value, node bound, or scalar score.",
    ],
    0.98,
    1.72,
    5.4,
    2.5,
    14.5
  );
  addPanel(slide, 0.98, 4.58, 5.45, 1.12, { fill: COLORS.green, line: COLORS.greenLine });
  addBodyText(slide, "vanillafullstrong", 1.18, 4.86, 1.62, 0.18, 13, { bold: true, color: COLORS.accentDeep });
  addBodyText(slide, "The authors re-implement strong branching without solver side effects so dataset collection is clean and reproducible.", 2.72, 4.82, 3.3, 0.45, 11.5);

  addPanel(slide, 7.0, 1.02, 5.65, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Collection flow", 7.24, 1.18, 1.12, COLORS.accentSoft);
  addProcessStep(slide, 7.28, 1.74, 1.18, 0.88, "Random instances", "one benchmark family\nat a time", COLORS.warm);
  addArrow(slide, 8.62, 2.03, 0.42);
  addProcessStep(slide, 9.08, 1.74, 1.38, 0.88, "Run SCIP", "collect B&B node\nstates", COLORS.surface);
  addArrow(slide, 10.62, 2.03, 0.42);
  addProcessStep(slide, 11.08, 1.74, 1.18, 0.88, "Oracle label", "expert action\nper node", COLORS.green);

  const familyCards = [
    ["Set covering", "easy / medium / hard"],
    ["Combinatorial auction", "easy / medium / hard"],
    ["Facility location", "easy / medium / hard"],
    ["Max independent set", "easy / medium / hard"],
  ];
  let startX = 7.28;
  let startY = 3.05;
  familyCards.forEach((item, idx) => {
    addPanel(slide, startX + (idx % 2) * 2.54, startY + Math.floor(idx / 2) * 1.18, 2.28, 0.92, {
      fill: idx % 2 === 0 ? COLORS.warm : COLORS.surface,
      line: COLORS.line,
    });
    addBodyText(slide, item[0], startX + (idx % 2) * 2.54 + 0.14, startY + Math.floor(idx / 2) * 1.18 + 0.14, 1.96, 0.2, 11, {
      bold: true,
    });
    addBodyText(slide, item[1], startX + (idx % 2) * 2.54 + 0.14, startY + Math.floor(idx / 2) * 1.18 + 0.46, 1.96, 0.18, 10, {
      color: COLORS.muted,
    });
  });
  addMetricCard(
    slide,
    7.28,
    5.58,
    5.08,
    0.9,
    "Data scale",
    "100k / 20k / 20k",
    "train / validation / test state-action samples per benchmark",
    COLORS.surface
  );

  addFooter(slide, "Section 4.1; Section 5.1; Supplementary Section 1");
  finalizeSlide(
    slide,
    [
      "The critical point is that supervision is the expert action chosen at a node.",
      "The dataset is benchmark-specific and collected with a side-effect-free strong branching implementation.",
      "Evidence: Section 4.1; Section 5.1 Training; Supplementary Section 1 dataset collection details.",
    ].join("\n")
  );
}

function slide5() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    5,
    "GCNN Architecture",
    "Bipartite message passing over constraints, variables, and sparse edge features"
  );
  addPanel(slide, 0.68, 1.02, 5.35, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Architecture facts", 0.92, 1.18, 1.48, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Inputs are constraint nodes, variable nodes, and sparse edge features from the current LP relaxation.",
      "One graph convolution is split into two half-convolutions: variable -> constraint and constraint -> variable.",
      "Each message/update function is a 2-layer MLP.",
      "A final 2-layer perceptron scores variable nodes only, followed by masked softmax.",
      "The proposed variant uses sum aggregation and prenorm rather than mean aggregation.",
    ],
    0.96,
    1.72,
    4.55,
    3.42,
    14.2
  );
  addPanel(slide, 0.96, 5.4, 4.6, 0.92, { fill: COLORS.warm, line: COLORS.warmLine });
  addBodyText(slide, "Ablation takeaway", 1.16, 5.68, 1.36, 0.16, 12, { bold: true });
  addBodyText(slide, "On larger set covering instances, sum aggregation + prenorm generalizes better than mean or sum-without-prenorm.", 2.54, 5.62, 2.72, 0.36, 11.2);

  addPanel(slide, 6.28, 1.02, 6.37, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Figure 2 style diagram", 6.54, 1.18, 1.56, COLORS.accentSoft);
  addProcessStep(slide, 6.62, 1.88, 1.56, 1.02, "Bipartite graph", "constraint nodes\nvariable nodes\nedge features", COLORS.warm);
  addArrow(slide, 8.34, 2.26, 0.42);
  addProcessStep(slide, 8.82, 1.88, 1.64, 1.02, "Half-conv 1", "variable ->\nconstraint", COLORS.surface);
  addArrow(slide, 10.62, 2.26, 0.42);
  addProcessStep(slide, 11.08, 1.88, 1.12, 1.02, "Half-conv 2", "constraint ->\nvariable", COLORS.surface);
  addProcessStep(slide, 7.06, 3.56, 1.7, 0.92, "Sum aggregation", "keeps counting-style\nsignals", COLORS.green);
  addProcessStep(slide, 8.98, 3.56, 1.5, 0.92, "Prenorm", "stabilizes training\nbefore optimization", COLORS.green);
  addProcessStep(slide, 10.72, 3.56, 1.48, 0.92, "Policy head", "2-layer MLP on\nvariable nodes", COLORS.surface);
  slide.addShape(pptx.ShapeType.line, {
    x: 7.38,
    y: 4.88,
    w: 4.7,
    h: 0,
    line: { color: COLORS.lineDark, pt: 1.1, beginArrowType: "none", endArrowType: "triangle" },
  });
  addBodyText(slide, "masked softmax over candidate branching variables", 7.42, 5.06, 4.34, 0.18, 11.5, {
    bold: true,
  });
  addBodyText(slide, "Constraint nodes are discarded at the end; the action is decided on variable nodes.", 6.72, 5.52, 5.18, 0.42, 11.5, {
    color: COLORS.muted,
  });

  addFooter(slide, "Figure 2; Section 4.2; Section 4.3; Section 5.3");
  finalizeSlide(
    slide,
    [
      "The proposed architecture is intentionally shallow enough to keep inference cost low inside the solver.",
      "The paper claims sum aggregation and prenorm are the key architectural choices for larger-instance generalization.",
      "Evidence: Figure 2; Section 4.2; Section 4.3; Table 3; Section 5.3.",
    ].join("\n")
  );
}

function slide6() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    6,
    "Output, Loss, and Inference Are Aligned",
    "One of the cleanest action-level alignment stories in the paper"
  );
  addPanel(slide, 0.68, 1.02, 6.0, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Key equations", 0.92, 1.18, 1.1, COLORS.accentSoft);
  addPanel(slide, 0.96, 1.72, 5.45, 1.16, { fill: COLORS.warm, line: COLORS.warmLine });
  addBodyText(slide, "Policy output", 1.16, 1.92, 1.0, 0.16, 12, { bold: true });
  addEquation(slide, String.raw`\pi_\theta(a \mid s_t)`, 2.18, 1.9, 1.8, 0.44);
  addBodyText(slide, "masked softmax over candidate branching variables", 4.25, 1.90, 1.72, 0.30, 10.8, {
    color: COLORS.muted,
  });

  addPanel(slide, 0.96, 3.08, 5.45, 1.26, { fill: COLORS.surface, line: COLORS.line });
  addBodyText(slide, "Training loss", 1.16, 3.28, 1.12, 0.16, 12, { bold: true });
  addEquation(
    slide,
    String.raw`L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log \pi_{\theta}(a_i^\star \mid s_i)`,
    1.16,
    3.52,
    4.8,
    0.5
  );

  addPanel(slide, 0.96, 4.62, 5.45, 1.1, { fill: COLORS.green, line: COLORS.greenLine });
  addBodyText(slide, "Inference rule", 1.16, 4.84, 1.08, 0.16, 12, { bold: true, color: COLORS.accentDeep });
  addEquation(
    slide,
    String.raw`a_t = \arg\max_{a \in \mathcal{A}(s_t)} \pi_\theta(a \mid s_t)`,
    1.16,
    5.04,
    4.8,
    0.42
  );

  addPanel(slide, 7.0, 1.02, 5.65, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Interpretation", 7.24, 1.18, 1.08, COLORS.accentSoft);
  addProcessStep(slide, 7.3, 1.82, 1.56, 0.98, "Output", "probability distribution\nover legal branch actions", COLORS.warm);
  addArrow(slide, 8.98, 2.18, 0.42);
  addProcessStep(slide, 9.46, 1.82, 1.56, 0.98, "Loss", "imitate expert's\nchosen action", COLORS.surface);
  addArrow(slide, 11.14, 2.18, 0.42);
  addProcessStep(slide, 11.62, 1.82, 0.72, 0.98, "Use", "branch", COLORS.green);
  addBullets(
    slide,
    [
      "The local action space is the same at training and inference time.",
      "The paper does not train on solve time or node count directly; those remain downstream metrics.",
      "So action-level alignment is strong, but objective-level alignment is still surrogate learning.",
    ],
    7.32,
    3.36,
    4.7,
    1.78,
    13.5
  );
  addPanel(slide, 7.28, 5.36, 5.02, 0.98, { fill: COLORS.warm, line: COLORS.warmLine });
  addBodyText(slide, "Important nuance", 7.52, 5.62, 1.26, 0.16, 12, { bold: true });
  addBodyText(slide, "The bigger mismatch is not output/loss/inference. It is imitation of a high-quality expert versus optimization of global solver efficiency.", 8.84, 5.54, 3.14, 0.34, 11.2);

  addFooter(slide, "Equation (3); Section 4.1; Section 4.3; Section 5.1");
  finalizeSlide(
    slide,
    [
      "This paper is unusually clean on output / loss / inference alignment.",
      "The remaining mismatch is one level higher: optimize imitation, evaluate solve efficiency.",
      "Evidence: Equation (3); Section 4.1; Section 4.3; Section 5.1.",
    ].join("\n")
  );
}

function slide7() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    7,
    "Novelty and Experimental Evidence",
    "Sell the representation, formulation, and realistic solver evaluation, not an exaggerated priority claim"
  );
  addPanel(slide, 0.68, 1.02, 4.2, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Real novelty", 0.92, 1.18, 1.12, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Bipartite MILP graph state encoding instead of heavier manual feature engineering.",
      "GCNN policy over branch actions rather than rank or score prediction.",
      "Evaluation inside a realistic SCIP setup, including comparison to default branching.",
    ],
    0.96,
    1.72,
    3.4,
    2.1,
    14
  );
  addPanel(slide, 0.96, 4.22, 3.6, 1.52, { fill: COLORS.warm, line: COLORS.warmLine });
  addBodyText(slide, "Selected runtime highlight", 1.16, 4.48, 1.52, 0.16, 12, { bold: true });
  addBodyText(slide, "Set covering, medium: GCNN 10.29 s versus LMART 14.42 s and SCIP RPB 17.41 s.", 1.16, 4.76, 3.0, 0.44, 11.2);
  addBodyText(slide, "The paper also reports 99/100 wins for GCNN on that split.", 1.16, 5.22, 3.0, 0.22, 11.2, {
    color: COLORS.muted,
  });

  addPanel(slide, 5.08, 1.02, 4.1, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Table 1: acc@1", 5.32, 1.18, 1.12, COLORS.accentSoft);
  addBodyText(slide, "GCNN versus best prior ML baseline", 5.36, 1.54, 2.2, 0.16, 10.5, {
    color: COLORS.muted,
  });
  addComparisonBar(slide, 5.36, 2.06, "Set covering", 65.5, 57.6);
  addComparisonBar(slide, 5.36, 2.86, "Comb. auction", 61.6, 57.3);
  addComparisonBar(slide, 5.36, 3.66, "Facility location", 71.2, 68.0);
  addComparisonBar(slide, 5.36, 4.46, "Max indep. set", 56.5, 48.9);
  addPanel(slide, 5.32, 5.48, 3.38, 0.82, { fill: COLORS.green, line: COLORS.greenLine });
  addBodyText(slide, "Headline", 5.56, 5.74, 0.7, 0.14, 11.5, { bold: true, color: COLORS.accentDeep });
  addBodyText(slide, "Highest acc@1 on all four benchmarks.", 6.38, 5.72, 2.06, 0.18, 11.5);

  addPanel(slide, 9.38, 1.02, 3.27, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Table 2 / Table 3", 9.62, 1.18, 1.22, COLORS.accentSoft);
  addProcessStep(slide, 9.64, 1.82, 2.7, 1.08, "Generalization", "Train on easy only;\nstill improves solver behavior on medium / hard instances.", COLORS.warm);
  addProcessStep(slide, 9.64, 3.1, 2.7, 1.08, "Ablation", "On larger set covering instances,\nsum + prenorm beats mean and sum-without-prenorm.", COLORS.surface);
  addProcessStep(slide, 9.64, 4.38, 2.7, 1.08, "Nuance", "Benefits shrink as problems grow much larger; speed-quality tradeoff remains.", COLORS.green);

  addFooter(slide, "Section 2; Table 1; Table 2; Table 3; Section 5.2; Section 5.3");
  finalizeSlide(
    slide,
    [
      "Do not describe this as the first imitation-learning branching paper.",
      "The stronger novelty claim is the combination of graph representation, GCNN policy, action classification, and full-solver evaluation.",
      "Evidence: Section 2; Table 1; Table 2; Table 3; Section 5.2; Section 5.3.",
    ].join("\n")
  );
}

function slide8() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    8,
    "Strengths, Limits, and Takeaways",
    "Where the paper is strongest, and where to stay disciplined in claiming novelty or capability"
  );
  addPanel(slide, 0.68, 1.02, 5.7, 4.86, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Strengths", 0.92, 1.18, 0.88, COLORS.accentSoft);
  addProcessStep(slide, 0.98, 1.82, 5.1, 1.0, "Clean action design", "Output, loss, and deployed inference operate on the same action space.", COLORS.green);
  addProcessStep(slide, 0.98, 3.0, 5.1, 1.0, "Representation win", "The bipartite graph reduces manual feature engineering and respects variable-sized MILPs.", COLORS.surface);
  addProcessStep(slide, 0.98, 4.18, 5.1, 1.0, "Realistic evaluation", "The policy is tested inside SCIP rather than in a stripped-down research solver.", COLORS.warm);

  addPanel(slide, 6.68, 1.02, 5.97, 4.86, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Limits", 6.92, 1.18, 0.64, COLORS.accentSoft);
  addProcessStep(slide, 6.98, 1.82, 5.37, 1.0, "Partial observability", "The graph state is only a subset of the full solver state, so the process is effectively a POMDP.", COLORS.red);
  addProcessStep(slide, 6.98, 3.0, 5.37, 1.0, "Surrogate objective", "The model imitates strong branching instead of directly optimizing solve time or node count.", COLORS.surface);
  addProcessStep(slide, 6.98, 4.18, 5.37, 1.0, "Scaling caveat", "Performance gains shrink on much larger instances; larger models can hurt wall-clock time despite better decisions.", COLORS.warm);

  addPanel(slide, 1.04, 6.02, 11.32, 0.48, { fill: COLORS.accentDeep, line: COLORS.accentDeep });
  addBodyText(
    slide,
    "Takeaway: this paper is best understood as learned branching inside a classical exact solver, not as a standalone neural combinatorial optimizer.",
    1.32,
    6.16,
    10.76,
    0.14,
    13.5,
    { color: "FFFFFF", bold: true, align: "center" }
  );

  addFooter(slide, "Section 4.2; Section 5.2; Section 6; Section 7");
  finalizeSlide(
    slide,
    [
      "A strong closing line is that the solver remains classical and exact; the learned model modernizes one important decision rule.",
      "Future directions naturally include stronger downstream objectives and reinforcement-learning variants.",
      "Evidence: Section 4.2; Section 5.2; Section 6 Discussion; Section 7 Conclusion.",
    ].join("\n")
  );
}

async function main() {
  const outputDir = path.resolve(__dirname, "../output");
  ensureDir(outputDir);
  slide1();
  slide2();
  slide3();
  slide4();
  slide5();
  slide6();
  slide7();
  slide8();
  const outputFile = path.join(
    outputDir,
    "exact-combinatorial-optimization-with-gcnn-report.pptx"
  );
  await pptx.writeFile({ fileName: outputFile });
  console.log(outputFile);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});




