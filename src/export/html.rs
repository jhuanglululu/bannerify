//! The output: one self-contained HTML page.
//!
//! Everything is inlined — both compare panes, both schematics and the whole
//! per-cell solution — as `data:` URIs and one JSON blob, so the file can be
//! mailed, dropped in a Discord channel or opened from a USB stick with nothing
//! else beside it. The two panes are encoded differently on purpose: the
//! generated wall is PNG (it is flat-coloured, it compresses to nothing, and it
//! is the image the download button hands over), while the original pane is a
//! *photograph* shown at the same size — PNG would make it the single largest
//! thing in the file, several times the wall's own weight, for a pane nobody
//! downloads. There is no template engine: the layout is a single static
//! page, so the "engine" would only ever substitute the values that are already
//! `format!` arguments here.
//!
//! ## What the JavaScript may do
//!
//! Look things up, and nothing else. The preview is a *pre-rendered image*; the
//! page never draws a banner. The crafting guide reads `DATA` — the compact
//! blob below — and prints the layers of one cell, which is a table lookup with
//! a `join`. That boundary is deliberate: the wall the page shows is the wall
//! the exporter rendered, not a re-derivation a viewer could disagree with.
//!
//! ## The blob
//!
//! One object of parallel arrays, indices everywhere:
//!
//! ```text
//! r, c        rows, columns
//! p, n        pattern ids / display names, indexed by layer entries
//! d, l, h     dye ids / display names / #rrggbb, indexed by base and layers
//! b           block ids, indexed by `k`
//! g[row][col] = [base, pattern, dye, pattern, dye, ...]
//! k[row][col] = the block behind that banner
//! ```
//!
//! A cell is a flat little integer array — about a dozen bytes — because the
//! obvious `{"base": "...", "patterns": [{...}]}` shape costs sixty, and the
//! blob is the one part of the page that grows with the wall.

use crate::color::NUM_COLORS;
use crate::export::{
    Materials, Wall, base64, block_counts, color_label, escape, hex, pattern_label, thousands,
};

/// Page-level facts the header prints and the download buttons name.
pub struct Report<'a> {
    /// The input image's file name, as the user gave it.
    pub input: &'a str,
    /// Base name for the downloaded files (no extension).
    pub stem: &'a str,
    /// The preview PNG, encoded.
    pub preview: &'a [u8],
    /// The original image, encoded, at the preview's size.
    pub original: &'a [u8],
    /// MIME type of [`Report::original`] — the original pane is a photograph,
    /// so it is JPEG where the banner wall is PNG (see [`page`]).
    pub original_mime: &'a str,
    /// Preview pixel size, which is also the compare slider's aspect ratio.
    pub size: (usize, usize),
    /// The `.litematic` bytes.
    pub litematic: &'a [u8],
    /// The `.schem` bytes.
    pub schem: &'a [u8],
}

/// Render the whole page.
pub fn page(wall: &Wall<'_>, report: &Report<'_>) -> String {
    let materials = Materials::of(wall);
    let mut out = String::with_capacity(
        (report.preview.len()
            + report.original.len()
            + report.litematic.len()
            + report.schem.len())
            * 4
            / 3
            + 64 * 1024,
    );

    let (stem, ext) = match report.input.rsplit_once('.') {
        Some((stem, ext)) => (stem.to_string(), format!(".{ext}")),
        None => (report.input.to_string(), String::new()),
    };

    out.push_str(HEAD_OPEN);
    out.push_str(&escape(report.input));
    out.push_str(HEAD_CLOSE);
    out.push_str(STYLE);

    // ---- header ------------------------------------------------------------
    out.push_str(&format!(
        "<h1>{}<span class=\"ext\">{}</span></h1>\n\n\
         <p><strong>{} rows × {} columns</strong> · {} banners</p>\n\n",
        escape(&stem),
        escape(&ext),
        thousands(wall.rows),
        thousands(wall.columns),
        thousands(wall.banners()),
    ));

    // ---- downloads ---------------------------------------------------------
    // The preview is already in the page as the compare slider's second pane,
    // so its button hands over *that* image instead of embedding a second copy
    // of it — at wall sizes the duplicate would be the largest thing in the
    // file. The schematics have no such twin and carry their own data URI.
    out.push_str(&format!(
        "<div class=\"btn-row\">\n  <button class=\"btn primary\" \
         onclick=\"downloadPreview('{}.png')\">Download preview PNG</button>\n",
        escape(report.stem)
    ));
    download(
        &mut out,
        "btn",
        "Download .litematic",
        "application/octet-stream",
        report.litematic,
        &format!("{}.litematic", report.stem),
    );
    download(
        &mut out,
        "btn",
        "Download .schem",
        "application/octet-stream",
        report.schem,
        &format!("{}.schem", report.stem),
    );
    out.push_str("</div>\n\n");

    // ---- compare slider ----------------------------------------------------
    let (pw, ph) = report.size;
    out.push_str(&format!(
        "<h2>Preview <span class=\"hint\">drag the slider to compare the \
         original with the generated wall</span></h2>\n\n\
         <div class=\"compare\" id=\"compare\" style=\"--split:50%;aspect-ratio:{pw}/{ph}\">\n"
    ));
    out.push_str(&format!(
        "  <img class=\"pane original\" alt=\"Original image\" src=\"data:{};base64,",
        report.original_mime
    ));
    out.push_str(&base64(report.original));
    out.push_str("\">\n  <img class=\"pane generated\" id=\"preview-img\" alt=\"Generated banner wall\" src=\"data:image/png;base64,");
    out.push_str(&base64(report.preview));
    out.push_str(
        "\">\n  <div class=\"divider\"></div>\n\
         \x20 <span class=\"pane-tag l\">Original</span>\n\
         \x20 <span class=\"pane-tag r\">Generated</span>\n</div>\n\
         <input class=\"compare-range\" type=\"range\" min=\"0\" max=\"100\" value=\"50\"\n\
         \x20      aria-label=\"Comparison split\"\n\
         \x20      oninput=\"document.getElementById('compare').style.setProperty('--split', this.value+'%')\">\n\n",
    );

    // ---- crafting guide ----------------------------------------------------
    // Two columns: the picker, the address line and the give command on the
    // left, the step table on the right — the table is the tall element, and
    // pairing it with the short controls halves the section's height. The
    // shell is static and the lookup only fills three holes in it, which keeps
    // the JavaScript to what it is allowed to be (see the module docs).
    out.push_str(&format!(
        "<h2>Crafting guide <span class=\"hint\">rows 1–{} top to bottom, \
         columns 1–{} left to right</span></h2>\n\n\
         <div class=\"craft\">\n\
         \x20 <div>\n\
         \x20   <div class=\"jump\">\n\
         \x20     <label for=\"row-in\">Row</label><input id=\"row-in\" value=\"1\" inputmode=\"numeric\" oninput=\"showBanner()\">\n\
         \x20     <label for=\"col-in\">Column</label><input id=\"col-in\" value=\"1\" inputmode=\"numeric\" oninput=\"showBanner()\">\n\
         \x20   </div>\n\
         \x20   <p id=\"banner-meta\"></p>\n\
         \x20   <pre id=\"give\"></pre>\n\
         \x20   <div class=\"btn-row\"><button class=\"btn\" onclick=\"copyGive(this)\">Copy command</button></div>\n\
         \x20 </div>\n\
         \x20 <div id=\"banner-steps\"></div>\n\
         </div>\n\n",
        thousands(wall.rows),
        thousands(wall.columns),
    ));

    // ---- materials ---------------------------------------------------------
    out.push_str("<h2>Materials <span class=\"hint\">everything the banners cost</span></h2>\n\n");
    let mut items: Vec<(usize, String, usize)> = Vec::new();
    for c in 0..NUM_COLORS {
        if materials.wool[c] > 0 {
            items.push((c, format!("{} wool", color_label(c)), materials.wool[c]));
        }
    }
    for c in 0..NUM_COLORS {
        if materials.dye[c] > 0 {
            items.push((c, format!("{} dye", color_label(c)), materials.dye[c]));
        }
    }
    items.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
    kv_grid(
        &mut out,
        items.iter().map(|(color, label, count)| {
            (
                format!(
                    "<span class=\"swatch\" style=\"background:{}\"></span>{}",
                    hex(*color),
                    escape(label)
                ),
                thousands(*count),
            )
        }),
    );

    // ---- the blocks behind the banners -------------------------------------
    let blocks = block_counts(wall);
    out.push_str(&format!(
        "<h2>Wall blocks <span class=\"hint\">{} types over {} positions, \
         placed by both schematics</span></h2>\n\n",
        thousands(blocks.len()),
        thousands(wall.block_rows() * wall.columns),
    ));
    kv_grid(
        &mut out,
        blocks.iter().map(|(name, count)| {
            (
                format!("<code class=\"inline\">{}</code>", escape(name)),
                thousands(*count),
            )
        }),
    );

    out.push_str("<hr>\n\n");

    out.push_str(
        "<blockquote><p>Generated by \
         <a href=\"https://github.com/jhuanglululu/bannerify\">bannerify</a>.</p></blockquote>\n\n",
    );
    out.push_str("</div>\n<script>\nconst DATA = ");
    blob(&mut out, wall);
    out.push_str(";\n");
    out.push_str(SCRIPT);
    out.push_str("</script>\n</body>\n</html>\n");
    out
}

/// A list of `(item, count)` pairs, flowed into as many `Item · Count` columns
/// as the viewport fits.
///
/// A single two-column table would be a metre of scrolling for the sixteen dye
/// colours and rather more for the blocks, and a table with a *fixed* number of
/// repeated column pairs cannot narrow without hiding data. So the pairs are
/// grid items instead: `auto-fill` picks the column count from the available
/// width, filling left to right, and collapses to one column on a phone with
/// nothing lost. `item` is trusted HTML (a swatch or a `<code>` id); the
/// callers escape their own text.
fn kv_grid(out: &mut String, entries: impl Iterator<Item = (String, String)>) {
    out.push_str("<div class=\"kv-grid\">\n");
    for (item, count) in entries {
        out.push_str(&format!(
            "  <div class=\"kv\"><span>{item}</span><span class=\"num\">{count}</span></div>\n"
        ));
    }
    out.push_str("</div>\n\n");
}

/// One `data:` download button.
fn download(out: &mut String, class: &str, label: &str, mime: &str, data: &[u8], filename: &str) {
    out.push_str(&format!(
        "  <a class=\"{class}\" download=\"{}\" href=\"data:{mime};base64,",
        escape(filename)
    ));
    out.push_str(&base64(data));
    out.push_str(&format!("\">{}</a>\n", escape(label)));
}

/// Serialise the lookup blob. Hand-written rather than `serde_json`: every
/// string in it is an `[a-z_]` id or a title-cased label, so nothing needs
/// escaping, and the numbers are small non-negative integers.
fn blob(out: &mut String, wall: &Wall<'_>) {
    let strings = |items: &[String]| {
        let quoted: Vec<String> = items.iter().map(|s| format!("\"{s}\"")).collect();
        format!("[{}]", quoted.join(","))
    };

    out.push_str(&format!("{{\"r\":{},\"c\":{},", wall.rows, wall.columns));
    out.push_str(&format!("\"p\":{},", strings(&wall.patterns.names)));
    out.push_str(&format!(
        "\"n\":{},",
        strings(
            &wall
                .patterns
                .names
                .iter()
                .map(|id| pattern_label(id))
                .collect::<Vec<_>>()
        )
    ));
    out.push_str(&format!(
        "\"d\":{},",
        strings(
            &crate::color::COLOR_NAMES
                .iter()
                .map(|s| (*s).to_string())
                .collect::<Vec<_>>()
        )
    ));
    out.push_str(&format!(
        "\"l\":{},",
        strings(&(0..NUM_COLORS).map(color_label).collect::<Vec<_>>())
    ));
    out.push_str(&format!(
        "\"h\":{},",
        strings(&(0..NUM_COLORS).map(hex).collect::<Vec<_>>())
    ));

    out.push_str(&format!("\"b\":{},", strings(&wall.blocks.names)));

    out.push_str("\"g\":[");
    for row in 0..wall.rows {
        if row > 0 {
            out.push(',');
        }
        out.push('[');
        for col in 0..wall.columns {
            if col > 0 {
                out.push(',');
            }
            let cell = wall.cell(row, col);
            out.push('[');
            out.push_str(&cell.base.to_string());
            for &(p, dye) in &cell.layers {
                out.push_str(&format!(",{p},{dye}"));
            }
            out.push(']');
        }
        out.push(']');
    }

    // The block a banner hangs on is the one at its own block row — the same
    // Y the schematics give them (`crate::export::schematic`). The block row
    // below is the one the banner droops over, and is somebody else's banner's.
    out.push_str("],\"k\":[");
    for row in 0..wall.rows {
        if row > 0 {
            out.push(',');
        }
        out.push('[');
        for col in 0..wall.columns {
            if col > 0 {
                out.push(',');
            }
            out.push_str(&wall.block_ids[col][row].to_string());
        }
        out.push(']');
    }
    out.push_str("]}");
}

const HEAD_OPEN: &str = "<!doctype html>\n<html lang=\"en\">\n<head>\n\
    <meta charset=\"utf-8\">\n\
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
    <title>";

const HEAD_CLOSE: &str = " — Banner chart</title>\n";

/// GitHub's rendered-README look, from the layout example in
/// `context/presentations/`.
const STYLE: &str = r#"<style>
:root{
  --fg:#1f2328;
  --fg-muted:#59636e;
  --border:#d1d9e0;
  --border-muted:#d1d9e0b3;
  --canvas:#ffffff;
  --canvas-subtle:#f6f8fa;
  --accent:#0969da;
  --btn-bg:#f6f8fa;
  --btn-border:#d1d9e0;
  --success:#1a7f37;
}
*{box-sizing:border-box;margin:0}
body{
  background:var(--canvas);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif;
  font-size:16px;line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
.markdown-body{max-width:880px;margin:0 auto;padding:56px 40px 104px}

h1{font-size:2em;font-weight:600;padding-bottom:.3em;border-bottom:1px solid var(--border-muted);margin:0 0 16px}
h1 .ext{color:var(--fg-muted);font-weight:400}
h2{font-size:1.5em;font-weight:600;padding-bottom:.3em;border-bottom:1px solid var(--border-muted);margin:40px 0 16px}
h2 .hint{font-size:14px;font-weight:400;color:var(--fg-muted);margin-left:10px;letter-spacing:0}
p{margin:0 0 16px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.muted{color:var(--fg-muted)}
strong{font-weight:600}

code,.mono{
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:85%;
}
code.inline{background:#818b981f;border-radius:6px;padding:.2em .4em}
pre{
  background:var(--canvas-subtle);border-radius:6px;padding:16px;
  overflow-x:auto;margin:0 0 16px;font-size:85%;line-height:1.45;
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}

.btn{
  display:inline-block;font-size:14px;font-weight:500;line-height:20px;
  padding:5px 16px;border:1px solid var(--btn-border);border-radius:6px;
  background:var(--btn-bg);color:var(--fg);cursor:pointer;text-decoration:none;
}
.btn:hover{background:#eff2f5;text-decoration:none}
.btn.primary{background:#1f883d;border-color:#1f883d;color:#fff}
.btn.primary:hover{background:#1a7f37}
.btn.copied{background:#1a7f37;border-color:#1a7f37;color:#fff}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}

table{border-collapse:collapse;margin:0 0 16px;display:block;overflow-x:auto;font-size:15px}
th,td{border:1px solid var(--border);padding:6px 13px;text-align:left}
th{font-weight:600}
tr:nth-child(even) td{background:var(--canvas-subtle)}
td.num{text-align:right;font-variant-numeric:tabular-nums}

/* item/count pairs, flowed into as many columns as fit */
.kv-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:0 28px;margin:0 0 16px;border-top:1px solid var(--border);
}
.kv{
  display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  font-size:15px;padding:5px 2px;border-bottom:1px solid var(--border);
}
.kv .num{text-align:right;font-variant-numeric:tabular-nums;color:var(--fg-muted)}

/* crafting guide: controls + command beside the step table */
.craft{display:grid;grid-template-columns:minmax(260px,1fr) minmax(300px,1.15fr);gap:8px 32px;align-items:start}
.craft pre{white-space:pre-wrap;word-break:break-all}
.craft table{width:100%}
@media (max-width:720px){.craft{grid-template-columns:1fr}}

.swatch{
  display:inline-block;width:13px;height:13px;border-radius:3px;
  vertical-align:-1px;margin-right:6px;border:1px solid #1f23281f;
}

.compare{
  position:relative;overflow:hidden;border:1px solid var(--border);border-radius:6px;
  user-select:none;max-width:700px;
}
.pane{position:absolute;inset:0;width:100%;height:100%;object-fit:fill;image-rendering:pixelated}
.pane.generated{clip-path:inset(0 0 0 var(--split,50%))}
.divider{position:absolute;top:0;bottom:0;left:var(--split,50%);width:2px;background:#fff;pointer-events:none}
.pane-tag{
  position:absolute;top:8px;font-size:12px;background:#ffffffd9;
  border:1px solid var(--border);border-radius:6px;padding:1px 8px;color:var(--fg-muted);pointer-events:none;
}
.pane-tag.l{left:8px}.pane-tag.r{right:8px}
.compare-range{width:100%;max-width:700px;margin:8px 0 16px;accent-color:var(--accent);display:block}
.compare-range:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

.jump{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:0 0 16px}
.jump label{font-size:14px}
.jump input{
  font-size:14px;width:80px;padding:5px 12px;line-height:20px;
  border:1px solid var(--btn-border);border-radius:6px;background:var(--canvas);color:var(--fg);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
}
.jump input:focus-visible{outline:2px solid var(--accent);outline-offset:-1px;border-color:var(--accent)}

blockquote{
  border-left:4px solid var(--border);padding:0 1em;color:var(--fg-muted);margin:0 0 16px;
}
hr{border:0;border-top:1px solid var(--border-muted);margin:24px 0}
</style>
</head>
<body>
<div class="markdown-body">

"#;

/// The page's only behaviour: read a cell out of `DATA` and print it.
const SCRIPT: &str = r#"
function group(n) { return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

function cellAt(r, c) { return DATA.g[r][c]; }

function giveCommand(cell) {
  const item = DATA.d[cell[0]] + '_banner';
  if (cell.length < 3) return '/give @p ' + item;
  const parts = [];
  for (let i = 1; i < cell.length; i += 2) {
    parts.push('{pattern:"' + DATA.p[cell[i]] + '",color:"' + DATA.d[cell[i + 1]] + '"}');
  }
  return '/give @p ' + item + '[banner_patterns=[' + parts.join(',') + ']]';
}

function swatch(dye) {
  return '<span class="swatch" style="background:' + DATA.h[dye] + '"></span>';
}

function showBanner() {
  const meta = document.getElementById('banner-meta');
  const steps = document.getElementById('banner-steps');
  const give = document.getElementById('give');
  const r = parseInt(document.getElementById('row-in').value, 10);
  const c = parseInt(document.getElementById('col-in').value, 10);
  if (!(r >= 1 && r <= DATA.r && c >= 1 && c <= DATA.c)) {
    meta.innerHTML = '<span class="muted">Pick a row between 1 and ' + group(DATA.r) +
      ', and a column between 1 and ' + group(DATA.c) + '.</span>';
    give.textContent = '';
    steps.innerHTML = '';
    return;
  }
  const cell = cellAt(r - 1, c - 1);
  const index = (r - 1) * DATA.c + c;
  meta.innerHTML = '<strong>Row ' + group(r) + ', Column ' + group(c) + '</strong> ' +
    '<span class="muted">— banner ' + group(index) + ' of ' + group(DATA.r * DATA.c) + '</span>';
  give.textContent = giveCommand(cell);

  let html = '<table>\n<tr><th>Step</th><th>Pattern</th><th>Dye</th></tr>\n';
  html += '<tr><td class="num">—</td><td>Wall block</td><td><code class="inline">' +
    DATA.b[DATA.k[r - 1][c - 1]] + '</code></td></tr>\n';
  html += '<tr><td class="num">—</td><td>Base</td><td>' + swatch(cell[0]) +
    DATA.l[cell[0]] + ' wool + stick</td></tr>\n';
  let step = 0;
  for (let i = 1; i < cell.length; i += 2) {
    step += 1;
    html += '<tr><td class="num">' + step + '</td><td>' + DATA.n[cell[i]] + '</td><td>' +
      swatch(cell[i + 1]) + DATA.l[cell[i + 1]] + ' dye</td></tr>\n';
  }
  html += '</table>\n';
  steps.innerHTML = html;
}

function downloadPreview(filename) {
  const a = document.createElement('a');
  a.href = document.getElementById('preview-img').src;
  a.download = filename;
  a.click();
}

function copyGive(btn) {
  navigator.clipboard.writeText(document.getElementById('give').textContent).then(function () {
    btn.textContent = 'Copied';
    btn.classList.add('copied');
    setTimeout(function () { btn.textContent = 'Copy command'; btn.classList.remove('copied'); }, 1500);
  });
}

showBanner();
"#;
