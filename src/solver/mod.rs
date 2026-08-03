//! The banner solver: which dyes and patterns approximate each cell.
//!
//! See `context/plans/2-solver.md`. A cell is solved in four stages, each
//! skipped when its configuration disables it, all sharing one reusable
//! [`Workspace`]:
//!
//! - [`variance`] — the pre-pass that hands each cell a layer budget.
//! - [`greedy`] — the fill: base dye, then one (pattern, dye) layer at a time.
//! - [`refine`] — windowed beam refinement over prefix/suffix caches.
//! - [`perturb`] — random re-rolls, re-refined, kept if better.
//! - [`lab`] — the OKLab final pass (`--lab-refine`).
//!
//! Support:
//!
//! - [`workspace`] — every buffer the stages touch, one allocation per work
//!   item, sized for the top row and viewed as a tail for the others.
//! - [`cell`] — getting a cell's pixels out of a column band and its composite
//!   back into the column's preview strip.

use std::io::{BufWriter, Write};
use std::path::Path;

use colored::Colorize;

use crate::color::COLOR_NAMES;
use crate::logger::error_out;
use crate::pattern::Patterns;

pub mod cell;
pub mod greedy;
pub mod lab;
pub mod perturb;
pub mod refine;
pub mod variance;
pub mod workspace;

pub use perturb::Rng;
pub use workspace::{Plane, Solution, SolveCfg, Stages, Workspace};

/// Write one JSONL line per cell: `{"row","col","base","layers","error"}`.
///
/// `cols[c][r]` is cell `(r, c)` — the work items are block columns, so the
/// results arrive column-major; the file stays **row-major**, which is the
/// reading order of the wall. Written serially after the `par_iter`, from the
/// collected results — the column items never touch a file.
///
/// Hand-formatted rather than pulling in `serde_json` for five fields: dye and
/// pattern ids are `[a-z0-9_]` by construction (asserted below), so no string
/// in the output needs escaping, and the floats are printed with `{:.6}`.
pub fn write_jsonl(path: &Path, patterns: &Patterns, cols: &[Vec<Solution>]) {
    debug_assert!(
        patterns
            .names
            .iter()
            .map(String::as_str)
            .chain(COLOR_NAMES)
            .all(|n| n.chars().all(|c| c.is_ascii_lowercase() || c == '_')),
        "ids must need no JSON escaping"
    );

    let file = std::fs::File::create(path).unwrap_or_else(|e| {
        error_out!(
            "could not write '{}': {}",
            path.display().to_string().yellow(),
            e.to_string().red()
        );
    });
    let mut out = BufWriter::new(file);

    let rows = cols.first().map_or(0, Vec::len);
    debug_assert!(
        cols.iter().all(|col| col.len() == rows),
        "every column has the same number of cells"
    );

    let mut write = || -> std::io::Result<()> {
        let mut line = String::with_capacity(256);
        for r in 0..rows {
            for (c, col) in cols.iter().enumerate() {
                let cell = &col[r];
                line.clear();
                line.push_str(&format!(
                    "{{\"row\":{},\"col\":{},\"base\":\"{}\",\"layers\":[",
                    r, c, COLOR_NAMES[cell.base]
                ));
                for (i, &(p, dye)) in cell.layers.iter().enumerate() {
                    if i > 0 {
                        line.push(',');
                    }
                    line.push_str(&format!(
                        "[\"{}\",\"{}\"]",
                        patterns.names[p], COLOR_NAMES[dye]
                    ));
                }
                line.push_str(&format!("],\"error\":{:.6}}}\n", cell.error));
                out.write_all(line.as_bytes())?;
            }
        }
        out.flush()
    };

    write().unwrap_or_else(|e| {
        error_out!(
            "could not write '{}': {}",
            path.display().to_string().yellow(),
            e.to_string().red()
        );
    });
}
