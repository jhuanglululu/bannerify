//! `--format json`: the solved wall as data, for programs that build, render or
//! post-process it — and the materials object the `--json` run summary shares
//! with it.
//!
//! Ids are unprefixed Minecraft ids (namespace `minecraft:`), the same strings
//! `--exclude-patterns`, `--blocks` and `--list` use. Banners are
//! row-major; rows count from the top, columns from the left, 0-based. Banner
//! `(r, c)` hangs on the face of block `(r, c)`, and the block grid is one row
//! taller than the banner grid (see [`super::schematic`] for the build).

use serde_json::{Map, Value, json};

use crate::color::{COLOR_NAMES, NUM_COLORS};
use crate::export::{Materials, Wall, block_counts, give_command};
use crate::logger::SCHEMA_VERSION;

/// The artifact, pretty-printed so it can be read and diffed as text.
pub fn document(wall: &Wall<'_>, input: &str) -> Vec<u8> {
    let banners: Vec<Value> = (0..wall.rows)
        .flat_map(|row| (0..wall.columns).map(move |column| (row, column)))
        .map(|(row, column)| {
            let cell = wall.cell(row, column);
            let layers: Vec<Value> = cell
                .layers
                .iter()
                .map(|&(p, dye)| json!({"pattern": wall.patterns.names[p], "color": COLOR_NAMES[dye]}))
                .collect();
            json!({
                "row": row,
                "column": column,
                "base": COLOR_NAMES[cell.base],
                "layers": layers,
                "give": give_command(wall, cell),
            })
        })
        .collect();
    let blocks: Vec<Vec<&str>> = (0..wall.block_rows())
        .map(|row| {
            (0..wall.columns)
                .map(|column| wall.blocks.names[wall.block_ids[column][row]].as_str())
                .collect()
        })
        .collect();

    let document = json!({
        "schema_version": SCHEMA_VERSION,
        "generator": concat!("bannerify ", env!("CARGO_PKG_VERSION")),
        "input": input,
        "wall": {
            "columns": wall.columns,
            "banner_rows": wall.rows,
            "block_rows": wall.block_rows(),
        },
        "banners": banners,
        "blocks": blocks,
        "materials": materials(wall),
    });
    let mut out = serde_json::to_vec_pretty(&document).expect("a JSON value always serialises");
    out.push(b'\n');
    out
}

/// What the wall costs: banners per base colour, dye per colour, and blocks,
/// keyed by id. Zero counts are left out.
pub fn materials(wall: &Wall<'_>) -> Value {
    let m = Materials::of(wall);
    let per_color = |counts: &[usize; NUM_COLORS]| -> Map<String, Value> {
        COLOR_NAMES
            .iter()
            .zip(counts)
            .filter(|&(_, &n)| n > 0)
            .map(|(name, &n)| (name.to_string(), json!(n)))
            .collect()
    };
    let blocks: Map<String, Value> = block_counts(wall)
        .into_iter()
        .map(|(name, n)| (name, json!(n)))
        .collect();
    json!({
        "banners": per_color(&m.wool),
        "dye": per_color(&m.dye),
        "total_dye": m.total_dye(),
        "blocks": blocks,
    })
}
