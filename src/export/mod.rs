//! Turning the solved wall into the things a player can use: one self-contained
//! HTML page, with the schematics and the preview images embedded in it.
//!
//! [`Wall`] is the read-only view every writer takes — the solutions, the
//! matched blocks and the tables their indices point into — so no writer owns
//! or copies the grid, and all of them agree on what "row 3, column 5" means:
//! **rows count from the top, columns from the left, both 1-based in anything a
//! user reads and 0-based in anything the code indexes.**
//!
//! - [`nbt`] — the write-only NBT encoder both schematics share.
//! - [`schematic`] — `.schem` (Sponge v2) and `.litematic` (Litematica v6).
//! - [`html`] — the page itself.
//!
//! ## Names
//!
//! Our pattern asset filenames **are** the in-game registry ids: all 42 of
//! `assets/banners/` appear verbatim in the `minecraft:banner_pattern`
//! registry, so `/give` and the block-entity `patterns` tags can use
//! [`crate::pattern::Patterns::names`] unchanged and no id mapping table
//! exists. What does need translating is the *display* name a crafting guide
//! should print — the heraldic ones ("Bend Sinister", "Bordure Indented") —
//! and that is [`pattern_label`].

use std::collections::HashMap;

use crate::block::Blocks;
use crate::color::{COLOR_NAMES, COLORS_RGB, NUM_COLORS};
use crate::pattern::Patterns;
use crate::solver::Solution;

pub mod html;
pub mod nbt;
pub mod schematic;

/// The finished wall, as every exporter reads it.
///
/// Both grids arrive **column-major** — the pipeline's work item is a block
/// column — and are transposed only by the accessors, never by a copy.
pub struct Wall<'a> {
    /// Banner rows.
    pub rows: usize,
    /// Banner (and block) columns.
    pub columns: usize,
    /// The pattern table the solutions index.
    pub patterns: &'a Patterns,
    /// The block table [`Wall::block_ids`] indexes.
    pub blocks: &'a Blocks,
    /// `block_ids[col][block_row]`.
    pub block_ids: &'a [Vec<usize>],
    /// `cells[col][row]`.
    pub cells: &'a [Vec<Solution>],
}

impl Wall<'_> {
    /// Block rows: one more than the banner rows.
    pub fn block_rows(&self) -> usize {
        self.rows + 1
    }

    /// Banners in the wall.
    pub fn banners(&self) -> usize {
        self.rows * self.columns
    }

    /// The solution for banner cell `(row, col)`, 0-based from the top left.
    pub fn cell(&self, row: usize, col: usize) -> &Solution {
        &self.cells[col][row]
    }

    /// The `minecraft:`-prefixed block id behind block cell `(row, col)`.
    pub fn block(&self, row: usize, col: usize) -> &str {
        &self.blocks.qualified[self.block_ids[col][row]]
    }

    /// Every solution, row-major — the reading order of the wall.
    pub fn iter(&self) -> impl Iterator<Item = &Solution> {
        (0..self.rows).flat_map(move |r| (0..self.columns).map(move |c| self.cell(r, c)))
    }
}

/// What building the wall costs, counted from the solutions.
pub struct Materials {
    /// Banners (i.e. wool) needed per base dye, indexed like [`COLOR_NAMES`].
    pub wool: [usize; NUM_COLORS],
    /// Dye needed per colour: one per pattern layer laid in that colour.
    pub dye: [usize; NUM_COLORS],
}

impl Materials {
    /// Count a finished wall.
    pub fn of(wall: &Wall<'_>) -> Self {
        let mut m = Self {
            wool: [0; NUM_COLORS],
            dye: [0; NUM_COLORS],
        };
        for cell in wall.iter() {
            m.wool[cell.base] += 1;
            for &(_, dye) in &cell.layers {
                m.dye[dye] += 1;
            }
        }
        m
    }

    /// Total pattern layers across the wall — the header's "dye" figure.
    pub fn total_dye(&self) -> usize {
        self.dye.iter().sum()
    }
}

/// The `/give` item for one banner, in 1.21+ component syntax.
///
/// `white_banner[banner_patterns=[{pattern:"bend",color:"orange"}]]` — the
/// pattern is the registry id, unprefixed (the `minecraft:` namespace is the
/// default), and a banner with no layers is just the item.
pub fn give_command(wall: &Wall<'_>, cell: &Solution) -> String {
    let item = format!("{}_banner", COLOR_NAMES[cell.base]);
    if cell.layers.is_empty() {
        return format!("/give @p {item}");
    }
    let layers: Vec<String> = cell
        .layers
        .iter()
        .map(|&(p, dye)| {
            format!(
                "{{pattern:\"{}\",color:\"{}\"}}",
                wall.patterns.names[p], COLOR_NAMES[dye]
            )
        })
        .collect();
    format!("/give @p {item}[banner_patterns=[{}]]", layers.join(","))
}

/// A dye or wool colour as `#rrggbb`, for the page's swatches.
pub fn hex(color: usize) -> String {
    let [r, g, b] = COLORS_RGB[color];
    format!("#{r:02x}{g:02x}{b:02x}")
}

/// A colour id as a label: `light_blue` → `Light blue`.
pub fn color_label(color: usize) -> String {
    title_case(COLOR_NAMES[color])
}

/// The in-game display name of a banner pattern.
///
/// The heraldic names from Minecraft's language file — what the crafting table
/// and the pattern's tooltip call the layer. Ids not in the table are shown
/// title-cased, which is what a future pattern would want anyway.
pub fn pattern_label(id: &str) -> String {
    let name = match id {
        "base" => "Base",
        "border" => "Bordure",
        "bricks" => "Field masoned",
        "circle" => "Roundel",
        "creeper" => "Creeper charge",
        "cross" => "Saltire",
        "curly_border" => "Bordure indented",
        "diagonal_left" => "Per bend sinister",
        "diagonal_right" => "Per bend",
        "diagonal_up_left" => "Per bend inverted",
        "diagonal_up_right" => "Per bend sinister inverted",
        "flow" => "Flow",
        "flower" => "Flower charge",
        "globe" => "Globe",
        "gradient" => "Gradient",
        "gradient_up" => "Base gradient",
        "guster" => "Guster",
        "half_horizontal" => "Per fess",
        "half_horizontal_bottom" => "Per fess inverted",
        "half_vertical" => "Per pale",
        "half_vertical_right" => "Per pale inverted",
        "mojang" => "Thing",
        "piglin" => "Snout",
        "rhombus" => "Lozenge",
        "skull" => "Skull charge",
        "small_stripes" => "Paly",
        "square_bottom_left" => "Base dexter canton",
        "square_bottom_right" => "Base sinister canton",
        "square_top_left" => "Chief dexter canton",
        "square_top_right" => "Chief sinister canton",
        "straight_cross" => "Cross",
        "stripe_bottom" => "Base",
        "stripe_center" => "Pale",
        "stripe_downleft" => "Bend sinister",
        "stripe_downright" => "Bend",
        "stripe_left" => "Pale dexter",
        "stripe_middle" => "Fess",
        "stripe_right" => "Pale sinister",
        "stripe_top" => "Chief",
        "triangle_bottom" => "Chevron",
        "triangle_top" => "Inverted chevron",
        "triangles_bottom" => "Base indented",
        "triangles_top" => "Chief indented",
        _ => return title_case(id),
    };
    name.to_string()
}

/// `light_blue` → `Light blue`.
fn title_case(id: &str) -> String {
    let mut out = id.replace('_', " ");
    if let Some(first) = out.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    out
}

/// `1234567` → `1,234,567`, the page's number format.
pub fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Standard base64, for the page's `data:` URIs.
///
/// Hand-rolled: the alphabet is twelve lines and the alternative is a
/// dependency whose entire job is those twelve lines.
pub fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Escape the five characters that must not appear raw in HTML text.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Blocks the wall actually uses, with how many of each — reported on the page
/// so a builder knows what to gather.
pub fn block_counts(wall: &Wall<'_>) -> Vec<(String, usize)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for col in wall.block_ids {
        for &id in col {
            *counts.entry(wall.blocks.names[id].as_str()).or_default() += 1;
        }
    }
    let mut out: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(name, n)| (name.to_string(), n))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_groups_from_the_right() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn base64_matches_the_rfc_test_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }
}
