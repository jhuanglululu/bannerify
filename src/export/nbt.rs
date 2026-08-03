//! A minimal, write-only NBT encoder.
//!
//! Ported from the Python build's `schematic.py` `_write_*` functions, which
//! exist for the same reason this does: both schematic formats we emit are one
//! fixed structure each, so a full NBT library (parsing, typed trees, schema
//! validation) would be several thousand lines of dependency to serialise a
//! shape we already know. Everything is big-endian, as NBT is.
//!
//! Only the tags the two writers use are here; adding one is a variant and an
//! arm. There is no reader — nothing in the pipeline consumes NBT.

/// One NBT tag value.
///
/// [`Tag::List`] carries its element type explicitly because an empty list must
/// still declare one, and the element type of a non-empty list must match every
/// element (asserted in debug builds when it is written).
#[derive(Clone, Debug)]
pub enum Tag {
    /// `TAG_Byte`.
    Byte(i8),
    /// `TAG_Short`.
    Short(i16),
    /// `TAG_Int`.
    Int(i32),
    /// `TAG_Long`.
    Long(i64),
    /// `TAG_String`.
    Str(String),
    /// `TAG_Byte_Array`.
    ByteArray(Vec<u8>),
    /// `TAG_Int_Array`.
    IntArray(Vec<i32>),
    /// `TAG_Long_Array`.
    LongArray(Vec<i64>),
    /// `TAG_List`: element tag id, then the elements.
    List(u8, Vec<Tag>),
    /// `TAG_Compound`: named entries, written in the order given.
    Compound(Vec<(String, Tag)>),
}

/// `TAG_End`, the compound terminator.
const END: u8 = 0;
/// `TAG_Compound`, the tag id every root here has.
pub const COMPOUND: u8 = 10;

impl Tag {
    /// This tag's type id.
    pub fn id(&self) -> u8 {
        match self {
            Tag::Byte(_) => 1,
            Tag::Short(_) => 2,
            Tag::Int(_) => 3,
            Tag::Long(_) => 4,
            Tag::Str(_) => 8,
            Tag::List(..) => 9,
            Tag::Compound(_) => COMPOUND,
            Tag::ByteArray(_) => 7,
            Tag::IntArray(_) => 11,
            Tag::LongArray(_) => 12,
        }
    }

    /// Append this tag's *payload* (no id, no name) to `out`.
    fn write_payload(&self, out: &mut Vec<u8>) {
        match self {
            Tag::Byte(v) => out.push(*v as u8),
            Tag::Short(v) => out.extend_from_slice(&v.to_be_bytes()),
            Tag::Int(v) => out.extend_from_slice(&v.to_be_bytes()),
            Tag::Long(v) => out.extend_from_slice(&v.to_be_bytes()),
            Tag::Str(s) => write_string(out, s),
            Tag::ByteArray(b) => {
                out.extend_from_slice(&(b.len() as i32).to_be_bytes());
                out.extend_from_slice(b);
            }
            Tag::IntArray(v) => {
                out.extend_from_slice(&(v.len() as i32).to_be_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_be_bytes());
                }
            }
            Tag::LongArray(v) => {
                out.extend_from_slice(&(v.len() as i32).to_be_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_be_bytes());
                }
            }
            Tag::List(elem, items) => {
                debug_assert!(
                    items.iter().all(|t| t.id() == *elem),
                    "every element of a TAG_List has the list's element type"
                );
                out.push(*elem);
                out.extend_from_slice(&(items.len() as i32).to_be_bytes());
                for item in items {
                    item.write_payload(out);
                }
            }
            Tag::Compound(entries) => {
                for (name, tag) in entries {
                    out.push(tag.id());
                    write_string(out, name);
                    tag.write_payload(out);
                }
                out.push(END);
            }
        }
    }
}

/// Serialise a named root tag — the whole file, uncompressed.
///
/// Both formats want a compound root; Sponge names it `Schematic`, Litematica
/// leaves the name empty.
pub fn write_root(name: &str, root: &Tag) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(root.id());
    write_string(&mut out, name);
    root.write_payload(&mut out);
    out
}

/// Modified-UTF-8 in principle, plain UTF-8 in practice: every string these
/// writers emit is a Minecraft id or a short ASCII label, and the two encodings
/// agree on everything below U+0080 with no NUL bytes.
fn write_string(out: &mut Vec<u8>, s: &str) {
    debug_assert!(s.is_ascii(), "NBT strings here are ASCII ids and labels");
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Compound-building shorthand: `compound![("Name", Tag::Int(1)), ...]`.
macro_rules! compound {
    ($($name:expr => $tag:expr),* $(,)?) => {
        $crate::export::nbt::Tag::Compound(vec![$(($name.to_string(), $tag)),*])
    };
}

pub(crate) use compound;

/// LEB128-style varint, the encoding Sponge v2 packs `BlockData` in.
pub fn varint(mut value: u32, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// gzip a serialised NBT document — both formats are gzipped, always.
pub fn gzip(data: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data).expect("writing to a Vec cannot fail");
    enc.finish().expect("finishing a Vec encoder cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_match_the_reference_encoding() {
        let mut out = Vec::new();
        varint(0, &mut out);
        varint(127, &mut out);
        varint(128, &mut out);
        varint(300, &mut out);
        assert_eq!(out, vec![0x00, 0x7F, 0x80, 0x01, 0xAC, 0x02]);
    }

    #[test]
    fn a_compound_round_trips_to_the_documented_bytes() {
        let root = compound! { "n" => Tag::Short(5) };
        // 0x0a root id, "" name, then 0x02 'n' 0x0005, then TAG_End.
        assert_eq!(
            write_root("", &root),
            vec![0x0a, 0x00, 0x00, 0x02, 0x00, 0x01, b'n', 0x00, 0x05, 0x00]
        );
    }

    #[test]
    fn an_empty_list_still_declares_its_element_type() {
        let root = compound! { "l" => Tag::List(COMPOUND, vec![]) };
        let bytes = write_root("", &root);
        assert_eq!(&bytes[bytes.len() - 6..], &[COMPOUND, 0, 0, 0, 0, END]);
    }
}
