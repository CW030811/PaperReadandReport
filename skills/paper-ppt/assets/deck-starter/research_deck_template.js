/*
  Starter deck for a concise research presentation.
  Copy this file to 05_slides/ppt/src/build_deck.js inside a paper workspace.
  Then replace the placeholder content with material from 05_slides/slides_outline.md.

  For final rendering and layout validation, also follow:
  C:/Users/WIN/.codex/skills/slides/SKILL.md
*/

const pptxgen = require("pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "PaperReadand Report";
pptx.subject = "Research presentation deck";
pptx.title = "Research Paper Report";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const COLORS = {
  bg: "F8FAFC",
  text: "0F172A",
  accent: "0F766E",
  muted: "475569",
  line: "CBD5E1",
};

const SLIDES = [
  {
    title: "Paper Overview and Core Takeaway",
    bullets: [
      "Paper, venue, year, and task setting.",
      "State the one-sentence takeaway in plain technical language.",
      "Include one line on why the problem matters.",
    ],
    figureLabel: "Suggested figure: paper teaser or task illustration",
  },
  {
    title: "Research Question",
    bullets: [
      "State the exact question the paper tries to answer.",
      "Explain the practical or scientific gap.",
      "Mention the baseline assumption or bottleneck.",
    ],
    figureLabel: "Suggested figure: problem setting or failure case",
  },
  {
    title: "High-Level Idea",
    bullets: [
      "Summarize the method in 2 to 3 mechanism-level bullets.",
      "Highlight the key intuition and what changes from prior work.",
      "Keep implementation details off this slide.",
    ],
    figureLabel: "Suggested figure: pipeline or method sketch",
  },
  {
    title: "Dataset and Supervision",
    bullets: [
      "Name datasets and evaluation splits.",
      "Describe label source or target construction.",
      "Mention preprocessing only if it affects the claim.",
    ],
    figureLabel: "Suggested figure: data flow or sample annotations",
  },
  {
    title: "Network Structure",
    bullets: [
      "Show the architecture at module level.",
      "Call out 3 to 4 components only.",
      "Mention dimensions or heads only if they matter.",
    ],
    figureLabel: "Suggested figure: architecture diagram",
  },
  {
    title: "Output, Loss, and Inference",
    bullets: [
      "State what the network predicts.",
      "Show the main training objective.",
      "Explain inference steps and any post-processing.",
    ],
    figureLabel: "Suggested figure: main equation or inference flow",
  },
  {
    title: "Novel Contribution and Evidence",
    bullets: [
      "List the real novelty, not the paper's marketing language.",
      "Attach the strongest result table or ablation.",
      "Mention one fair comparison to prior work.",
    ],
    figureLabel: "Suggested figure: key table or ablation",
  },
  {
    title: "Strengths, Weaknesses, and Takeaways",
    bullets: [
      "State 2 strengths grounded in results or design.",
      "State 1 to 2 limitations or unresolved questions.",
      "Close with what this paper changes for your own work.",
    ],
    figureLabel: "Suggested figure: none or small summary graphic",
  },
];

function addHeader(slide, title, index) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.6,
    line: { color: COLORS.bg, transparency: 100 },
    fill: { color: COLORS.bg },
  });
  slide.addText(`0${index}. ${title}`, {
    x: 0.6,
    y: 0.18,
    w: 8.8,
    h: 0.28,
    fontFace: "Aptos Display",
    fontSize: 24,
    bold: true,
    color: COLORS.text,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.6,
    y: 0.62,
    w: 12.1,
    h: 0,
    line: { color: COLORS.line, pt: 1.2 },
  });
}

function addBullets(slide, bullets) {
  const runs = [];
  bullets.forEach((text) => {
    runs.push({
      text,
      options: { bullet: { indent: 12 } },
    });
  });

  slide.addText(runs, {
    x: 0.8,
    y: 1.05,
    w: 6.2,
    h: 4.7,
    fontFace: "Aptos",
    fontSize: 17,
    color: COLORS.text,
    breakLine: true,
    paraSpaceAfterPt: 12,
    valign: "top",
    margin: 0,
  });
}

function addFigurePlaceholder(slide, label) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 7.35,
    y: 1.15,
    w: 5.2,
    h: 4.1,
    rectRadius: 0.08,
    line: { color: COLORS.line, pt: 1.2, dash: "dash" },
    fill: { color: "FFFFFF", transparency: 0 },
  });
  slide.addText(label, {
    x: 7.65,
    y: 2.85,
    w: 4.6,
    h: 0.6,
    fontFace: "Aptos",
    fontSize: 14,
    align: "center",
    color: COLORS.muted,
    margin: 0,
  });
}

function addFooter(slide) {
  slide.addText("Replace placeholders with evidence-backed content from the paper workspace.", {
    x: 0.8,
    y: 6.82,
    w: 10.2,
    h: 0.2,
    fontFace: "Aptos",
    fontSize: 9,
    color: COLORS.muted,
    margin: 0,
  });
}

SLIDES.forEach((spec, idx) => {
  const slide = pptx.addSlide();
  slide.background = { color: COLORS.bg };
  addHeader(slide, spec.title, idx + 1);
  addBullets(slide, spec.bullets);
  addFigurePlaceholder(slide, spec.figureLabel);
  addFooter(slide);
});

pptx.writeFile({ fileName: "./output/research-report-template.pptx" });
