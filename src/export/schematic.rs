//! The two in-game delivery formats: Sponge `.schem` and Litematica
//! `.litematic`.
//!
//! ## The build both describe
//!
//! ```text
//!            z=0            z=1
//!   y=rows   block row 0    banner row 0     <- top of the wall
//!   y=rows-1 block row 1    banner row 1
//!     ...
//!   y=1      block row rows-1  banner row rows-1
//!   y=0      block row rows    (air)         <- the lip below the last banner
//! ```
//!
//! X is the wall's columns, Y is up (so wall row 0, the top, is the *highest*
//! Y), Z is depth: the blocks at `z = 0` and the wall banners one block in
//! front of them at `z = 1`, `facing=south`, which is the face that hangs on a
//! block to its north. A wall of `rows` banner rows is `rows + 1` blocks tall.
//!
//! ## Sponge v2
//!
//! `Palette` maps block-state strings to ids, `BlockData` is those ids as
//! varints in `x + z·W + y·W·L` order, and `BlockEntities` carries the banners'
//! `patterns`. `DataVersion` stays [`DATA_VERSION`] (1.21.4) — the format has
//! not changed since, and a DataVersion newer than the reader is the one thing
//! WorldEdit refuses.
//!
//! ## Litematica v6
//!
//! There is no published spec; this follows the format's writers (litemapy's
//! `to_nbt`, which mirrors Litematica's own `LitematicaBitArray`):
//!
//! - root `Version` 6, `SubVersion` 1, `MinecraftDataVersion`;
//! - `Metadata` with `EnclosingSize`, counts and timestamps;
//! - one region, whose `BlockStatePalette` is a *list* of
//!   `{Name, Properties}` compounds (not Sponge's string→id map);
//! - `BlockStates` is a packed `TAG_Long_Array`: `max(2, ceil(log2(palette)))`
//!   bits per entry, index `y·W·L + z·W + x`, **entries straddle long
//!   boundaries** — this is the pre-1.16 chunk packing, not the padded 1.16+
//!   one, and getting it wrong is the classic way to produce a file that opens
//!   and renders garbage;
//! - `TileEntities` are the block entities' NBT with `x`/`y`/`z` added.

use crate::color::COLOR_NAMES;
use crate::export::Wall;
use crate::export::nbt::{COMPOUND, Tag, compound, gzip, varint, write_root};

/// Minecraft data version the exports declare: 1.21.4. Both formats' readers
/// only use it to decide whether they must upgrade the block states, and ours
/// are already in the modern spelling.
pub const DATA_VERSION: i32 = 4189;

const LITEMATIC_VERSION: i32 = 6;
/// Litematica schematic sub-version, which version 6 carries.
const LITEMATIC_SUBVERSION: i32 = 1;

/// Air, which is palette entry 0 in both formats: everything not written is
/// air, and Litematica in particular assumes index 0 is empty space.
const AIR: &str = "minecraft:air";

/// Depth of the build: the block wall, and the banners in front of it.
const LENGTH: usize = 2;

struct Cell<'a> {
    name: &'a str,
    /// Block-state properties, `(key, value)`.
    props: &'a [(&'a str, &'a str)],
    /// The banner's pattern layers, if this is a banner.
    banner: Option<&'a crate::solver::Solution>,
}

/// Walk the build in `(y, z, x)` order — the order both formats' block arrays
/// use — handing each position to `f`.
fn walk<'a>(wall: &'a Wall<'a>, mut f: impl FnMut(usize, usize, usize, Option<Cell<'a>>)) {
    let height = wall.block_rows();
    for y in 0..height {
        // y counts up from the bottom; block/banner rows count down from the
        // top, so row = rows - y for banners and (height - 1 - y) for blocks.
        for z in 0..LENGTH {
            for x in 0..wall.columns {
                let cell = if z == 0 {
                    Some(Cell {
                        name: wall.block(height - 1 - y, x),
                        props: &[],
                        banner: None,
                    })
                } else if y == 0 {
                    None // the lip below the last banner row has nothing in front of it
                } else {
                    let solution = wall.cell(wall.rows - y, x);
                    Some(Cell {
                        name: BANNER_NAMES[solution.base],
                        props: &[("facing", "south")],
                        banner: Some(solution),
                    })
                };
                f(x, y, z, cell);
            }
        }
    }
}

/// `minecraft:<colour>_wall_banner` per dye, in [`COLOR_NAMES`] order.
static BANNER_NAMES: [&str; crate::color::NUM_COLORS] = [
    "minecraft:white_wall_banner",
    "minecraft:orange_wall_banner",
    "minecraft:magenta_wall_banner",
    "minecraft:light_blue_wall_banner",
    "minecraft:yellow_wall_banner",
    "minecraft:lime_wall_banner",
    "minecraft:pink_wall_banner",
    "minecraft:gray_wall_banner",
    "minecraft:light_gray_wall_banner",
    "minecraft:cyan_wall_banner",
    "minecraft:purple_wall_banner",
    "minecraft:blue_wall_banner",
    "minecraft:brown_wall_banner",
    "minecraft:green_wall_banner",
    "minecraft:red_wall_banner",
    "minecraft:black_wall_banner",
];

fn patterns_tag(wall: &Wall<'_>, solution: &crate::solver::Solution) -> Tag {
    Tag::List(
        COMPOUND,
        solution
            .layers
            .iter()
            .map(|&(p, dye)| {
                compound! {
                    "pattern" => Tag::Str(format!("minecraft:{}", wall.patterns.names[p])),
                    "color" => Tag::Str(COLOR_NAMES[dye].to_string()),
                }
            })
            .collect(),
    )
}

/// A block-state string, `minecraft:foo[k=v,...]` — Sponge's palette key.
fn state_string(cell: &Cell<'_>) -> String {
    if cell.props.is_empty() {
        return cell.name.to_string();
    }
    let props: Vec<String> = cell.props.iter().map(|(k, v)| format!("{k}={v}")).collect();
    format!("{}[{}]", cell.name, props.join(","))
}

/// Build the `.schem` (Sponge Schematic v2), gzipped.
pub fn schem(wall: &Wall<'_>) -> Vec<u8> {
    let (width, height) = (wall.columns, wall.block_rows());

    let mut palette: Vec<String> = vec![AIR.to_string()];
    let mut block_data: Vec<u8> = Vec::with_capacity(width * height * LENGTH);
    let mut entities: Vec<Tag> = Vec::new();

    walk(wall, |x, y, _z, cell| {
        let id = match &cell {
            None => 0,
            Some(cell) => {
                let state = state_string(cell);
                match palette.iter().position(|s| *s == state) {
                    Some(i) => i,
                    None => {
                        palette.push(state);
                        palette.len() - 1
                    }
                }
            }
        };
        varint(id as u32, &mut block_data);

        if let Some(solution) = cell.and_then(|c| c.banner) {
            entities.push(compound! {
                "Id" => Tag::Str("minecraft:banner".to_string()),
                "Pos" => Tag::IntArray(vec![x as i32, y as i32, 1]),
                "patterns" => patterns_tag(wall, solution),
            });
        }
    });

    let palette_tag = Tag::Compound(
        palette
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), Tag::Int(i as i32)))
            .collect(),
    );

    let root = compound! {
        "Version" => Tag::Int(2),
        "DataVersion" => Tag::Int(DATA_VERSION),
        "Width" => Tag::Short(width as i16),
        "Height" => Tag::Short(height as i16),
        "Length" => Tag::Short(LENGTH as i16),
        "PaletteMax" => Tag::Int(palette.len() as i32),
        "Palette" => palette_tag,
        "BlockData" => Tag::ByteArray(block_data),
        "Offset" => Tag::IntArray(vec![0, 0, 0]),
        "BlockEntities" => Tag::List(COMPOUND, entities),
    };

    gzip(&write_root("Schematic", &root))
}

/// Build the `.litematic` (Litematica schematic version 6), gzipped.
///
/// `name` becomes the schematic's metadata name — the label Litematica shows in
/// its load list — and `created` is the creation/modification timestamp in
/// milliseconds since the epoch.
pub fn litematic(wall: &Wall<'_>, name: &str, created: i64) -> Vec<u8> {
    let (width, height) = (wall.columns, wall.block_rows());
    let volume = width * height * LENGTH;

    let mut palette: Vec<Tag> = vec![compound! { "Name" => Tag::Str(AIR.to_string()) }];
    let mut keys: Vec<String> = vec![AIR.to_string()];
    let mut indices: Vec<u32> = Vec::with_capacity(volume);
    let mut entities: Vec<Tag> = Vec::new();
    let mut solid = 0usize;

    walk(wall, |x, y, z, cell| {
        let id = match &cell {
            None => 0,
            Some(cell) => {
                solid += 1;
                let key = state_string(cell);
                match keys.iter().position(|s| *s == key) {
                    Some(i) => i,
                    None => {
                        let mut entry = vec![("Name".to_string(), Tag::Str(cell.name.to_string()))];
                        if !cell.props.is_empty() {
                            entry.push((
                                "Properties".to_string(),
                                Tag::Compound(
                                    cell.props
                                        .iter()
                                        .map(|(k, v)| (k.to_string(), Tag::Str(v.to_string())))
                                        .collect(),
                                ),
                            ));
                        }
                        palette.push(Tag::Compound(entry));
                        keys.push(key);
                        keys.len() - 1
                    }
                }
            }
        };
        // walk() already visits in y, z, x order — which is exactly the
        // format's index order — so pushing in sequence *is* the indexing.
        indices.push(id as u32);

        if let Some(solution) = cell.and_then(|c| c.banner) {
            entities.push(compound! {
                "id" => Tag::Str("minecraft:banner".to_string()),
                "x" => Tag::Int(x as i32),
                "y" => Tag::Int(y as i32),
                "z" => Tag::Int(z as i32),
                "patterns" => patterns_tag(wall, solution),
            });
        }
    });

    let bits = bits_per_entry(palette.len());
    let region = compound! {
        "Position" => compound! { "x" => Tag::Int(0), "y" => Tag::Int(0), "z" => Tag::Int(0) },
        "Size" => compound! {
            "x" => Tag::Int(width as i32),
            "y" => Tag::Int(height as i32),
            "z" => Tag::Int(LENGTH as i32),
        },
        "BlockStatePalette" => Tag::List(COMPOUND, palette),
        "Entities" => Tag::List(COMPOUND, vec![]),
        "TileEntities" => Tag::List(COMPOUND, entities),
        "PendingBlockTicks" => Tag::List(COMPOUND, vec![]),
        "PendingFluidTicks" => Tag::List(COMPOUND, vec![]),
        "BlockStates" => Tag::LongArray(pack(&indices, bits)),
    };

    let root = compound! {
        "Version" => Tag::Int(LITEMATIC_VERSION),
        "SubVersion" => Tag::Int(LITEMATIC_SUBVERSION),
        "MinecraftDataVersion" => Tag::Int(DATA_VERSION),
        "Metadata" => compound! {
            "Name" => Tag::Str(name.to_string()),
            "Author" => Tag::Str("bannerify".to_string()),
            "Description" => Tag::Str(String::new()),
            "Software" => Tag::Str("bannerify".to_string()),
            "RegionCount" => Tag::Int(1),
            "TimeCreated" => Tag::Long(created),
            "TimeModified" => Tag::Long(created),
            "TotalBlocks" => Tag::Int(solid as i32),
            "TotalVolume" => Tag::Int(volume as i32),
            "EnclosingSize" => compound! {
                "x" => Tag::Int(width as i32),
                "y" => Tag::Int(height as i32),
                "z" => Tag::Int(LENGTH as i32),
            },
        },
        "Regions" => compound! { name => region },
    };

    gzip(&write_root("", &root))
}

/// Bits Litematica spends per palette index: `max(2, ceil(log2(len)))`.
fn bits_per_entry(palette_len: usize) -> usize {
    let bits = usize::BITS - palette_len.saturating_sub(1).leading_zeros();
    (bits as usize).max(2)
}

/// Pack `values` into Litematica's bit array.
///
/// Little-endian within each long, **and entries straddle long boundaries**:
/// an entry that does not fit in the current long continues in the low bits of
/// the next one. Written as one forward sweep with a carry rather than the
/// reference implementation's random-access `setAt`, which is the same layout
/// reached in order.
fn pack(values: &[u32], bits: usize) -> Vec<i64> {
    let longs = (values.len() * bits).div_ceil(64);
    let mut out = vec![0u64; longs];
    for (i, &v) in values.iter().enumerate() {
        let start = i * bits;
        let (word, offset) = (start / 64, start % 64);
        out[word] |= u64::from(v) << offset;
        if offset + bits > 64 {
            out[word + 1] |= u64::from(v) >> (64 - offset);
        }
    }
    out.into_iter().map(|w| w as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_sizes_map_to_the_reference_bit_widths() {
        // max(2, ceil(log2(n))): 1..=4 entries fit in 2 bits, 5..=8 in 3, ...
        for (len, bits) in [(1, 2), (2, 2), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4)] {
            assert_eq!(bits_per_entry(len), bits, "palette of {len}");
        }
    }

    #[test]
    fn entries_straddle_long_boundaries() {
        // 13 five-bit entries: the 13th starts at bit 60 and spills one bit
        // into the second long — the packing detail Litematica readers assume.
        let values: Vec<u32> = (0..13).map(|i| (i as u32 % 32) | 1).collect();
        let packed = pack(&values, 5);
        assert_eq!(packed.len(), (13 * 5usize).div_ceil(64));
        for (i, &v) in values.iter().enumerate() {
            let start = i * 5;
            let (word, offset) = (start / 64, start % 64);
            let mut got = (packed[word] as u64) >> offset;
            if offset + 5 > 64 {
                got |= (packed[word + 1] as u64) << (64 - offset);
            }
            assert_eq!(got & 0x1F, u64::from(v), "entry {i}");
        }
    }
}
