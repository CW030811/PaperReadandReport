const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { latexToSvgDataUri } = require("./pptxgenjs_helpers/latex");
const { imageSizingContain } = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "PaperReadand Report";
pptx.subject = "Research presentation deck";
pptx.title =
  "Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation";
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
  accentDeep: "115E59",
  accentSoft: "CCFBF1",
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
const ASSET_DIR = path.resolve(__dirname, "../assets");

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function fig(name) {
  return path.join(ASSET_DIR, name);
}

function addBase(slide, index, title, subtitle = "") {
  slide.background = { color: COLORS.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: SLIDE_W,
    h: 0.74,
    line: { color: COLORS.bg, transparency: 100 },
    fill: { color: COLORS.bg },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.56,
    y: 0.2,
    w: 0.78,
    h: 0.31,
    rectRadius: 0.05,
    line: { color: COLORS.accent, pt: 0.8 },
    fill: { color: COLORS.accentSoft },
  });
  slide.addText(String(index).padStart(2, "0"), {
    x: 0.73,
    y: 0.252,
    w: 0.42,
    h: 0.13,
    fontFace: "Aptos",
    fontSize: 12,
    bold: true,
    color: COLORS.accentDeep,
    align: "center",
    margin: 0,
  });
  slide.addText(title, {
    x: 1.56,
    y: 0.16,
    w: 9.4,
    h: 0.24,
    fontFace: "Aptos Display",
    fontSize: 24,
    bold: true,
    color: COLORS.text,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 1.58,
      y: 0.44,
      w: 10.2,
      h: 0.12,
      fontFace: "Aptos",
      fontSize: 9.5,
      color: COLORS.muted,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.58,
    y: 0.77,
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
    w: 8.1,
    h: 0.12,
    fontFace: "Aptos",
    fontSize: 9,
    color: COLORS.muted,
    margin: 0,
  });
  slide.addText("PaperReadand Report | S^2-MHCN", {
    x: 10.35,
    y: 7.08,
    w: 2.1,
    h: 0.12,
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
    rectRadius: 0.05,
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
    rectRadius: 0.04,
    line: { color: fill, pt: 0.8 },
    fill: { color: fill },
  });
  slide.addText(text, {
    x: x + 0.08,
    y: y + 0.075,
    w: w - 0.16,
    h: 0.1,
    fontFace: "Aptos",
    fontSize: 10.5,
    bold: true,
    color,
    align: "center",
    margin: 0,
  });
}

function addBodyText(slide, text, x, y, w, h, fontSize = 14, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: opts.fontFace || "Aptos",
    fontSize,
    bold: opts.bold || false,
    italic: opts.italic || false,
    color: opts.color || COLORS.text,
    align: opts.align || "left",
    valign: opts.valign || "top",
    margin: opts.margin !== undefined ? opts.margin : 0,
  });
}

function addBullets(slide, bullets, x, y, w, h, fontSize = 16) {
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
    paraSpaceAfterPt: 8,
    valign: "top",
    margin: 0,
  });
}

function addProcessCard(slide, x, y, w, h, title, body, fill = COLORS.surface) {
  addPanel(slide, x, y, w, h, { fill, line: COLORS.lineDark });
  addBodyText(slide, title, x + 0.14, y + 0.12, w - 0.28, 0.16, 11, {
    bold: true,
  });
  addBodyText(slide, body, x + 0.14, y + 0.38, w - 0.28, h - 0.5, 10.5, {
    color: COLORS.muted,
  });
}

function addArrow(slide, x, y, w) {
  slide.addShape(pptx.ShapeType.chevron, {
    x,
    y,
    w,
    h: 0.24,
    line: { color: COLORS.lineDark, pt: 0.8 },
    fill: { color: COLORS.grayFill },
  });
}

function addFigurePanel(slide, imagePath, x, y, w, h, caption = "") {
  addPanel(slide, x, y, w, h, { fill: COLORS.surface, line: COLORS.line });
  slide.addImage({
    path: imagePath,
    ...imageSizingContain(imagePath, x + 0.18, y + 0.18, w - 0.36, h - 0.52),
  });
  if (caption) {
    addBodyText(slide, caption, x + 0.24, y + h - 0.25, w - 0.48, 0.12, 9.5, {
      color: COLORS.muted,
      align: "center",
    });
  }
}

function addEquationCard(slide, title, latex, x, y, w, h, fill = COLORS.surface) {
  addPanel(slide, x, y, w, h, { fill, line: COLORS.line });
  addBodyText(slide, title, x + 0.16, y + 0.12, w - 0.32, 0.14, 11, {
    bold: true,
    color: COLORS.muted,
  });
  const data = latexToSvgDataUri(latex, true);
  slide.addImage({
    data,
    x: x + 0.16,
    y: y + 0.4,
    w: w - 0.32,
    h: h - 0.55,
  });
}

function addCallout(slide, x, y, w, h, title, body, fill, line, textColor = COLORS.text) {
  addPanel(slide, x, y, w, h, { fill, line });
  addBodyText(slide, title, x + 0.16, y + 0.14, w - 0.32, 0.14, 11.5, {
    bold: true,
    color: textColor,
  });
  addBodyText(slide, body, x + 0.16, y + 0.36, w - 0.32, h - 0.48, 11.2, {
    color: textColor,
  });
}

function addResultRow(
  slide,
  x,
  y,
  w,
  dataset,
  metric,
  refLabel,
  refValue,
  mhcnValue,
  s2Value
) {
  addPanel(slide, x, y, w, 0.64, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, dataset, x + 0.12, y + 0.16, 0.82, COLORS.accentSoft);
  addBodyText(slide, metric, x + 1.06, y + 0.16, 1.02, 0.12, 10.5, {
    bold: true,
  });
  addBodyText(slide, `${refLabel}: ${refValue}`, x + 2.15, y + 0.16, 1.35, 0.12, 10.2, {
    color: COLORS.muted,
  });
  addBodyText(slide, `MHCN: ${mhcnValue}`, x + 3.72, y + 0.16, 1.32, 0.12, 10.2, {
    color: COLORS.text,
  });
  addBodyText(slide, `S^2: ${s2Value}`, x + 5.18, y + 0.16, 1.05, 0.12, 10.2, {
    bold: true,
    color: COLORS.accentDeep,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 6.32,
    y: y + 0.2,
    w: 0.56,
    h: 0.22,
    line: { color: COLORS.accent, pt: 0.8 },
    fill: { color: COLORS.accentSoft },
  });
  addBodyText(slide, "best", x + 6.39, y + 0.245, 0.32, 0.08, 8.6, {
    bold: true,
    color: COLORS.accentDeep,
    align: "center",
  });
}

function finalizeSlide(slide, notes) {
  slide.addNotes(notes);
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function slide1() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    1,
    "Problem, Claim, and Research Question",
    "WWW 2021 | Junliang Yu et al. | Social recommendation under sparsity and cold-start"
  );

  addPanel(slide, 0.68, 1.02, 4.6, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Why this paper", 0.92, 1.2, 1.18, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Target task: improve social recommendation when user-item interactions are sparse.",
      "Core claim: pairwise social edges miss high-order patterns such as friends plus shared purchase behavior.",
      "Main question: how to encode these high-order relations and turn them into better top-K ranking performance?",
    ],
    0.98,
    1.72,
    3.9,
    2.1,
    14.2
  );
  addCallout(
    slide,
    0.96,
    4.28,
    4.02,
    1.12,
    "One-sentence takeaway",
    "Move from pairwise social graph modeling to high-order relation construction, then regularize the learned structure with self-supervision.",
    COLORS.warm,
    COLORS.warmLine
  );
  addCallout(
    slide,
    0.96,
    5.62,
    4.02,
    0.9,
    "What to remember",
    "This is not just another recommender with a better backbone. The key move is to explicitly model high-order social structure.",
    COLORS.green,
    COLORS.greenLine
  );

  addFigurePanel(
    slide,
    fig("Figure1.png"),
    5.56,
    1.02,
    7.09,
    5.88,
    "Figure 1: intuitive examples of high-order user relations."
  );

  addFooter(slide, "Abstract; Section 1; Figure 1");
  finalizeSlide(
    slide,
    [
      "Open by explaining the paper's limitation claim before mentioning equations.",
      "The practical pain point is data sparsity, especially cold-start users.",
      "Evidence: Abstract; Section 1 Introduction; Figure 1.",
    ].join("\n")
  );
}

function slide2() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    2,
    "Method Roadmap",
    "From motif extraction to ranking and structural regularization"
  );

  addPanel(slide, 0.68, 1.02, 4.1, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Four steps", 0.92, 1.2, 0.9, COLORS.accentSoft);
  addProcessCard(
    slide,
    0.96,
    1.72,
    3.54,
    0.92,
    "1. Detect motifs",
    "Use interaction matrix R and social matrix S to find 10 local triangular motif instances.",
    COLORS.warm
  );
  addProcessCard(
    slide,
    0.96,
    2.86,
    3.54,
    0.92,
    "2. Build channels",
    "Group motif instances into social, joint, and purchase channels, each inducing one user relation structure.",
    COLORS.surface
  );
  addProcessCard(
    slide,
    0.96,
    4.0,
    3.54,
    0.92,
    "3. Learn embeddings",
    "Run channel-specific hypergraph propagation plus one user-item graph branch for collaborative item signal.",
    COLORS.surface
  );
  addProcessCard(
    slide,
    0.96,
    5.14,
    3.54,
    0.92,
    "4. Train jointly",
    "Fuse channels with attention and optimize BPR together with hierarchical mutual information regularization.",
    COLORS.green
  );

  addFigurePanel(
    slide,
    fig("Figure3.png"),
    5.0,
    1.02,
    7.65,
    5.88,
    "Figure 3: the method combines three hypergraph channels, one graph branch, and an auxiliary self-supervised task."
  );

  addFooter(slide, "Section 3; Figure 3");
  finalizeSlide(
    slide,
    [
      "Use this slide as the map for the rest of the talk.",
      "The clean mental model is motif extraction -> three channels -> fusion -> ranking -> self-supervised regularization.",
      "Evidence: Section 3; Figure 3.",
    ].join("\n")
  );
}

function slide3() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    3,
    "Motif Construction and Channel Semantics",
    "Figure 2 explains what the three channels actually encode"
  );

  addFigurePanel(
    slide,
    fig("Figure2.png"),
    0.68,
    1.02,
    11.98,
    2.62,
    "Figure 2: 10 motif types grouped into social, joint, and purchase channels."
  );

  addProcessCard(
    slide,
    0.88,
    4.05,
    3.72,
    1.32,
    "Social channel",
    "Motifs M1-M7 capture follow / social connectivity patterns among users.",
    COLORS.green
  );
  addProcessCard(
    slide,
    4.82,
    4.05,
    3.72,
    1.32,
    "Joint channel",
    "Motifs M8-M9 mix social and purchase evidence to encode shared but hybrid high-order relations.",
    COLORS.warm
  );
  addProcessCard(
    slide,
    8.76,
    4.05,
    3.72,
    1.32,
    "Purchase channel",
    "Motif M10 focuses on purchase-driven high-order proximity and later becomes the strongest branch in ablation.",
    COLORS.surface
  );

  addCallout(
    slide,
    0.88,
    5.7,
    11.6,
    0.8,
    "Important reading note",
    "A motif is a local triangle instance, not a partition of the whole graph. Each channel then induces one user hypergraph or one equivalent motif-induced adjacency A_c.",
    COLORS.accentSoft,
    COLORS.accent
  );

  addFooter(slide, "Section 3.2.1; Figure 2");
  finalizeSlide(
    slide,
    [
      "Be precise here: the paper identifies local motif instances and groups them by semantics.",
      "This step is what turns vague high-order intuition into three usable channels.",
      "Evidence: Section 3.2.1; Figure 2.",
    ].join("\n")
  );
}

function slide4() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    4,
    "Representation Learning in Vanilla MHCN",
    "SGU -> channel propagation -> attention fusion -> graph branch -> ranking"
  );

  addPanel(slide, 0.68, 1.02, 7.15, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Pipeline inside MHCN", 0.92, 1.2, 1.42, COLORS.accentSoft);
  addProcessCard(slide, 0.98, 1.78, 1.82, 0.92, "SGU", "Gate the shared user embedding into channel-specific inputs.", COLORS.warm);
  addArrow(slide, 2.95, 2.12, 0.34);
  addProcessCard(slide, 3.34, 1.78, 1.96, 0.92, "Hypergraph conv", "Run Eq. (2) / Eq. (4) inside each semantic channel.", COLORS.surface);
  addArrow(slide, 5.47, 2.12, 0.34);
  addProcessCard(slide, 5.86, 1.78, 1.45, 0.92, "Attention", "Fuse P_s*, P_j*, and P_p* with user-specific weights.", COLORS.green);

  addProcessCard(slide, 1.38, 3.34, 2.16, 0.98, "Graph branch", "A LightGCN-style user-item graph branch injects item-side collaborative signal.", COLORS.surface);
  addArrow(slide, 3.75, 3.72, 0.34);
  addProcessCard(slide, 4.24, 3.34, 2.24, 0.98, "Final embeddings", "Combine channel fusion with the graph branch to get user and item embeddings P and Q.", COLORS.warm);
  addArrow(slide, 6.69, 3.72, 0.34);
  addProcessCard(slide, 5.0, 4.92, 1.88, 0.86, "Prediction", "Score with p_u^T q_i and train with BPR ranking loss.", COLORS.green);
  slide.addShape(pptx.ShapeType.line, {
    x: 5.95,
    y: 4.32,
    w: 0,
    h: 0.48,
    line: { color: COLORS.lineDark, pt: 1.0, endArrowType: "triangle" },
  });

  addCallout(
    slide,
    1.0,
    5.34,
    3.65,
    0.9,
    "Key nuance",
    "The propagation operator itself is mostly weight-light; the main learnable pieces are the SGU parameters, embeddings, and attention.",
    COLORS.accentSoft,
    COLORS.accent
  );

  addPanel(slide, 8.02, 1.02, 4.63, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Core equations", 8.26, 1.2, 1.12, COLORS.accentSoft);
  addEquationCard(
    slide,
    "Channel input gating",
    String.raw`\mathbf P_c^{(0)} = \mathbf P^{(0)} \odot \sigma(\mathbf P^{(0)}\mathbf W_g^c + \mathbf b_g^c)`,
    8.24,
    1.72,
    4.18,
    1.26,
    COLORS.warm
  );
  addEquationCard(
    slide,
    "Efficient hypergraph propagation",
    String.raw`\mathbf P_c^{(l+1)} = \hat{\mathbf D}_c^{-1}\mathbf A_c\mathbf P_c^{(l)}`,
    8.24,
    3.18,
    4.18,
    1.22
  );
  addEquationCard(
    slide,
    "Recommendation score",
    String.raw`\hat r_{u,i} = \mathbf p_u^\top \mathbf q_i`,
    8.24,
    4.62,
    4.18,
    1.06,
    COLORS.green
  );

  addFooter(slide, "Section 3.2.2; Eq. (1); Eq. (4); Eq. (6); Eq. (7); Eq. (8)");
  finalizeSlide(
    slide,
    [
      "This slide is the main technical backbone of the talk.",
      "Walk through it as SGU, per-channel propagation, layer average plus attention, graph branch, then BPR-based ranking.",
      "Evidence: Section 3.2.2; Eq. (1), Eq. (4), Eq. (6), Eq. (7), Eq. (8).",
    ].join("\n")
  );
}

function slide5() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    5,
    "Why Self-Supervised Learning Is Added",
    "Hierarchical mutual information acts as structural regularization"
  );

  addPanel(slide, 0.68, 1.02, 4.28, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Design logic", 0.92, 1.2, 1.02, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "After multi-channel aggregation, some high-order structure can be diluted. The paper calls this aggregating loss.",
      "The auxiliary task builds the hierarchy user <- sub-hypergraph <- hypergraph and maximizes mutual information across that hierarchy.",
      "Positive pairs come from the real structure; negative pairs come from shuffled sub-hypergraph representations.",
    ],
    0.98,
    1.72,
    3.86,
    2.18,
    13.8
  );
  addEquationCard(
    slide,
    "Discriminator view",
    String.raw`f_D(\mathbf p_u,\mathbf z_u) > f_D(\mathbf p_u,\tilde{\mathbf z}_u)`,
    0.96,
    4.18,
    4.0,
    0.98,
    COLORS.warm
  );
  addEquationCard(
    slide,
    "Joint training",
    String.raw`L = L_r + \beta L_s`,
    0.96,
    5.34,
    4.0,
    0.86,
    COLORS.green
  );

  addFigurePanel(
    slide,
    fig("Figure4.png"),
    5.18,
    1.02,
    7.47,
    5.88,
    "Figure 4: hierarchical mutual information over user, sub-hypergraph, and hypergraph representations."
  );

  addFooter(slide, "Section 3.3; Figure 4; Eq. (9); Eq. (11); Eq. (12)");
  finalizeSlide(
    slide,
    [
      "Explain this as a regularizer, not as a second recommendation label.",
      "The big idea is to preserve structural information that may be washed out by aggregation.",
      "Evidence: Section 3.3; Figure 4; Eq. (9), Eq. (11), Eq. (12).",
    ].join("\n")
  );
}

function slide6() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    6,
    "Experimental Setup and Main Results",
    "MHCN is already strong, and S^2-MHCN improves it further"
  );

  addPanel(slide, 0.68, 1.02, 3.5, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Setup", 0.92, 1.2, 0.66, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Datasets: LastFM, Douban, and Yelp. Douban is converted from explicit ratings to implicit positive feedback for top-K recommendation.",
      "Evaluation uses 5-fold cross-validation and full-candidate ranking, not sampled candidate sets.",
      "The paper reports both complete test results and cold-start results.",
      "Baselines include BPR, SBPR, GraphRec, DiffNet++, LightGCN, and DHCF.",
    ],
    0.96,
    1.72,
    2.9,
    3.2,
    13.1
  );
  addCallout(
    slide,
    0.96,
    5.3,
    2.92,
    0.92,
    "Reading rule",
    "Separate two claims: MHCN > baseline means the structure works; S^2-MHCN > MHCN means the SSL module is not decorative.",
    COLORS.warm,
    COLORS.warmLine
  );

  addPanel(slide, 4.42, 1.02, 8.23, 2.94, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Table 3: complete test set", 4.66, 1.2, 1.72, COLORS.accentSoft);
  addResultRow(slide, 4.66, 1.74, 7.7, "LastFM", "NDCG@10", "LightGCN", "0.23392", "0.23834", "0.24395");
  addResultRow(slide, 4.66, 2.5, 7.7, "Douban", "Recall@10", "LightGCN", "6.247%", "6.556%", "6.681%");
  addResultRow(slide, 4.66, 3.26, 7.7, "Yelp", "NDCG@10", "LightGCN", "0.04998", "0.05356", "0.06061");

  addPanel(slide, 4.42, 4.08, 8.23, 2.86, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Table 4: cold-start users", 4.66, 4.26, 1.68, COLORS.accentSoft);
  addResultRow(slide, 4.66, 4.8, 7.7, "LastFM", "NDCG@10", "DiffNet++", "0.16031", "0.17218", "0.19138");
  addResultRow(slide, 4.66, 5.54, 7.7, "Douban", "Recall@10", "MHCN", "9.646%", "9.646%", "10.632%");
  addResultRow(slide, 4.66, 6.28, 7.7, "Yelp", "NDCG@10", "MHCN", "0.04354", "0.04354", "0.05143");

  addFooter(slide, "Section 4.1; Section 4.2; Table 3; Table 4");
  finalizeSlide(
    slide,
    [
      "Do not oversell this as only a first-place table.",
      "The stronger reading is that the architecture already beats strong baselines, then SSL brings another boost, especially for cold-start users.",
      "Evidence: Section 4.1; Section 4.2; Table 3; Table 4.",
    ].join("\n")
  );
}

function slide7() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    7,
    "Ablation, Sensitivity, and Mechanism Validation",
    "Figure 5 to Figure 9 explain why the method works and where its boundary lies"
  );

  addPanel(slide, 0.68, 1.02, 3.78, 5.88, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "What the experiments say", 0.92, 1.2, 1.62, COLORS.accentSoft);
  addBullets(
    slide,
    [
      "Figure 5 and Figure 6: all channels help, but purchase channel contributes the most.",
      "Figure 7: hierarchical MIM beats local-only, global-only, and DGI-style alternatives.",
      "Figure 8: the SSL weight beta should stay small; performance peaks around 0.01.",
      "Figure 9: shallow depth works best; deeper models suffer from over-smoothing.",
    ],
    0.96,
    1.72,
    3.0,
    3.0,
    13.4
  );
  addCallout(
    slide,
    0.96,
    5.24,
    3.02,
    0.98,
    "Best one-line summary",
    "The paper proves not only that the method wins, but also which component matters most and what tuning range keeps it stable.",
    COLORS.green,
    COLORS.greenLine
  );

  addFigurePanel(
    slide,
    fig("Figure7.png"),
    4.68,
    1.02,
    7.97,
    2.96,
    "Figure 7: hierarchical MIM is the strongest self-supervised design."
  );
  addFigurePanel(
    slide,
    fig("Figure5&6.png"),
    4.68,
    4.28,
    3.86,
    2.12,
    "Figure 5-6: purchase channel dominates both ablation and attention."
  );
  addFigurePanel(
    slide,
    fig("Figure8&9.png"),
    8.78,
    4.28,
    3.88,
    2.12,
    "Figure 8-9: small beta and depth 2 work best."
  );

  addFooter(slide, "Section 4.3; Section 4.4; Figure 5; Figure 6; Figure 7; Figure 8; Figure 9");
  finalizeSlide(
    slide,
    [
      "This is the slide that validates the design story from earlier sections.",
      "Figure 7 is especially important because it checks whether the hierarchical self-supervised design is actually justified.",
      "Evidence: Section 4.3; Section 4.4; Figure 5 to Figure 9.",
    ].join("\n")
  );
}

function slide8() {
  const slide = pptx.addSlide();
  addBase(
    slide,
    8,
    "Contribution, Limitation, and Takeaway",
    "How to explain the paper clearly to the group"
  );

  addPanel(slide, 0.68, 1.02, 5.72, 4.02, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Main contributions", 0.92, 1.2, 1.34, COLORS.accentSoft);
  addProcessCard(
    slide,
    0.96,
    1.74,
    5.14,
    0.84,
    "1. High-order relation construction",
    "Reframe social recommendation from pairwise social graph modeling to multi-channel motif-induced hypergraph modeling.",
    COLORS.warm
  );
  addProcessCard(
    slide,
    0.96,
    2.78,
    5.14,
    0.84,
    "2. Practical representation pipeline",
    "Combine SGU, efficient hypergraph propagation, channel attention, and one user-item graph branch in a clean ranking architecture.",
    COLORS.surface
  );
  addProcessCard(
    slide,
    0.96,
    3.82,
    5.14,
    0.84,
    "3. Structural self-supervision",
    "Use hierarchical mutual information that matches the user <- sub-hypergraph <- hypergraph hierarchy instead of generic DGI-style SSL.",
    COLORS.green
  );

  addPanel(slide, 6.66, 1.02, 5.99, 4.02, { fill: COLORS.surface, line: COLORS.line });
  addLabel(slide, "Main limitations", 6.9, 1.2, 1.16, COLORS.accentSoft);
  addProcessCard(
    slide,
    6.94,
    1.74,
    5.4,
    1.04,
    "Purchase signal dominates",
    "The strongest evidence comes from purchase-related structure, so the method is not winning through explicit social edges alone.",
    COLORS.red
  );
  addProcessCard(
    slide,
    6.94,
    3.02,
    5.4,
    1.04,
    "Tuning and depth matter",
    "The method prefers small beta and shallow depth. Larger auxiliary weight or deeper propagation can hurt performance through interference or over-smoothing.",
    COLORS.warm
  );

  addCallout(
    slide,
    0.88,
    5.34,
    11.76,
    1.08,
    "Best final mental model",
    "Tell the story as high-order relation construction plus controlled structural regularization. That is the cleanest way to explain why this paper is different from a plain social GNN recommender.",
    COLORS.accent,
    COLORS.accent,
    "FFFFFF"
  );

  addFooter(slide, "Section 5 Conclusion; Table 4; Figure 5; Figure 7; Figure 8; Figure 9");
  finalizeSlide(
    slide,
    [
      "End with a balanced judgment: the paper has a coherent design story and evidence, but it also shows clear boundaries.",
      "That balance makes it a good group-meeting paper.",
      "Evidence: Section 5 Conclusion; Table 4; Figure 5; Figure 7; Figure 8; Figure 9.",
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
    "s2-mhcn-social-recommendation-report.pptx"
  );
  await pptx.writeFile({ fileName: outputFile });
  console.log(outputFile);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
