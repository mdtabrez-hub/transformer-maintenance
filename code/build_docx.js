/*
 * build_docx.js -- Render manuscript.json into an IEEE-style two-column .docx
 *
 *   node build_docx.js manuscript.json Manuscript.docx
 *
 * The document is a sequence of continuous sections so that wide figures and
 * tables can span both columns while body text stays in two columns.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, SectionType,
  TabStopType, TabStopPosition,
} = require("docx");

const specPath = process.argv[2] || "manuscript.json";
const outPath = process.argv[3] || "Manuscript.docx";
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));

const FONT = "Times New Roman";
const PT = (n) => Math.round(n * 2);          // half-points
const DXA = (inches) => Math.round(inches * 1440);

const PAGE = {
  size: { width: DXA(8.5), height: DXA(11) },
  margin: { top: DXA(0.75), bottom: DXA(1.0), left: DXA(0.62), right: DXA(0.62) },
};
const COL_W = 3.45;      // single column width, inches
const FULL_W = 7.26;     // both columns, inches

function runs(text, opts = {}) {
  // very light inline markup: *italic* and **bold**
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out.length ? out : [new TextRun({ text: "", ...opts })];
}

function imgRun(file, widthIn, maxWidthIn) {
  const dim = spec.assets[file];
  if (!dim) throw new Error("missing asset dimensions for " + file);
  let w = widthIn !== undefined ? widthIn : dim.pt_w / 72.0;
  if (maxWidthIn && w > maxWidthIn) w = maxWidthIn;
  const h = w * (dim.px_h / dim.px_w);
  return new ImageRun({
    type: "png",
    data: fs.readFileSync(file),
    transformation: { width: Math.round(w * 96), height: Math.round(h * 96) },
  });
}

// ------------------------------------------------------------------ blocks --
function block(b) {
  switch (b.t) {
    case "title":
      return [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
        children: runs(b.text, { font: FONT, size: PT(20), bold: true }),
      })];

    case "authors":
      return [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: runs(b.text, { font: FONT, size: PT(11) }),
      })];

    case "affil":
      return [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: runs(b.text, { font: FONT, size: PT(9), italics: true }),
      })];

    case "h1":
      return [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 90 },
        children: runs(b.text.toUpperCase(), { font: FONT, size: PT(10), bold: true }),
      })];

    case "h2":
      return [new Paragraph({
        spacing: { before: 130, after: 70 },
        children: runs(b.text, { font: FONT, size: PT(9.5), bold: true, italics: true }),
      })];

    case "h3":
      return [new Paragraph({
        spacing: { before: 100, after: 55 },
        children: runs(b.text, { font: FONT, size: PT(9), italics: true }),
      })];

    case "abstract":
      return [new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 90 },
        children: [
          new TextRun({ text: "Abstract\u2014", font: FONT, size: PT(9), bold: true, italics: true }),
          ...runs(b.text, { font: FONT, size: PT(9), bold: true }),
        ],
      })];

    case "keywords":
      return [new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 150 },
        children: [
          new TextRun({ text: "Index Terms\u2014", font: FONT, size: PT(9), bold: true, italics: true }),
          ...runs(b.text, { font: FONT, size: PT(9), bold: true }),
        ],
      })];

    case "p":
      return [new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 60, line: 230 },
        indent: b.noindent ? undefined : { firstLine: DXA(0.14) },
        children: runs(b.text, { font: FONT, size: PT(9.2) }),
      })];

    case "note":
      return [new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { before: 70, after: 90 },
        border: {
          left: { style: BorderStyle.SINGLE, size: 12, color: "1F4E79", space: 6 },
        },
        indent: { left: DXA(0.10) },
        children: runs(b.text, { font: FONT, size: PT(8.6), italics: true }),
      })];

    case "eq": {
      const wide = b.wide ? FULL_W : COL_W;
      const dim = spec.assets[b.img];
      let w = dim.pt_w / 72.0;
      const cap = wide - 0.55;
      if (w > cap) w = cap;
      return [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 70, after: 70 },
        tabStops: [{ type: TabStopType.RIGHT, position: Math.round(wide * 1440) }],
        children: [
          imgRun(b.img, w),
          new TextRun({ text: "\t(" + b.n + ")", font: FONT, size: PT(9.2) }),
        ],
      })];
    }

    case "fig": {
      const w = b.wide ? FULL_W : COL_W;
      return [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 110, after: 40 },
          children: [imgRun(b.img, w)],
        }),
        new Paragraph({
          alignment: b.wide ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
          spacing: { after: 130 },
          children: [
            new TextRun({ text: "Fig. " + b.n + ".  ", font: FONT, size: PT(8.2), bold: true }),
            ...runs(b.caption, { font: FONT, size: PT(8.2) }),
          ],
        }),
      ];
    }

    case "table": {
      const w = b.wide ? FULL_W : COL_W;
      const ncol = b.rows[0].length;
      const widths = b.widths
        ? b.widths.map((f) => Math.round(f * w * 1440))
        : Array(ncol).fill(Math.round((w / ncol) * 1440));
      const total = widths.reduce((a, c) => a + c, 0);

      const mkCell = (txt, i, header, lastRow) =>
        new TableCell({
          width: { size: widths[i], type: WidthType.DXA },
          shading: header
            ? { type: ShadingType.CLEAR, fill: "E8EEF4", color: "auto" }
            : undefined,
          margins: { top: 30, bottom: 30, left: 60, right: 60 },
          borders: {
            top: { style: BorderStyle.SINGLE, size: header ? 8 : 2, color: "666666" },
            bottom: { style: BorderStyle.SINGLE, size: header || lastRow ? 8 : 2, color: "666666" },
            left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
          },
          children: [new Paragraph({
            alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
            spacing: { after: 0, line: 200 },
            children: runs(String(txt), { font: FONT, size: PT(7.6), bold: header }),
          })],
        });

      const rows = b.rows.map((r, ri) => new TableRow({
        tableHeader: ri === 0,
        children: r.map((c, ci) => mkCell(c, ci, ri === 0, ri === b.rows.length - 1)),
      }));

      return [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 130, after: 50 },
          children: [
            new TextRun({ text: "TABLE " + b.n, font: FONT, size: PT(8.2), bold: true }),
            new TextRun({ text: "", break: 1 }),
            ...runs(b.caption.toUpperCase(), { font: FONT, size: PT(8.2) }),
          ],
        }),
        new Table({ columnWidths: widths, width: { size: total, type: WidthType.DXA }, rows }),
        new Paragraph({ spacing: { after: 130 }, children: [new TextRun({ text: "", size: PT(4) })] }),
      ];
    }

    case "listitem":
      return [new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 45, line: 230 },
        indent: { left: DXA(0.20), hanging: DXA(0.20) },
        children: runs(b.text, { font: FONT, size: PT(9.2) }),
      })];

    case "ref":
      return [new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 28, line: 205 },
        indent: { left: DXA(0.26), hanging: DXA(0.26) },
        children: runs(b.text, { font: FONT, size: PT(7.9) }),
      })];

    case "code":
      return b.lines.map((ln) => new Paragraph({
        spacing: { after: 0, line: 190 },
        children: [new TextRun({ text: ln === "" ? " " : ln, font: "Consolas", size: PT(6.6) })],
      }));

    default:
      throw new Error("unknown block type: " + b.t);
  }
}

// ---------------------------------------------------------------- sections --
const sections = spec.sections.map((sec, i) => {
  const children = [];
  sec.blocks.forEach((b) => children.push(...block(b)));
  return {
    properties: {
      type: i === 0 ? SectionType.NEXT_PAGE : SectionType.CONTINUOUS,
      page: PAGE,
      column: sec.cols === 2
        ? { count: 2, space: DXA(0.24), equalWidth: true }
        : { count: 1 },
    },
    children,
  };
});

const doc = new Document({
  creator: spec.meta.author,
  title: spec.meta.title,
  description: "Simulation study manuscript",
  styles: { default: { document: { run: { font: FONT, size: PT(9.2) } } } },
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log("wrote " + outPath + "  (" + (buf.length / 1024).toFixed(0) + " kB)");
});
