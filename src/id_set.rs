//! Id sets: named groups of block ids or of pattern ids, and the `--blocks`
//! and `--patterns` selections that are written in terms of them.
//!
//! A selection is a list of entries applied left to right. An entry names a
//! set or a block id and adds it; with a leading `-` it removes it instead:
//!
//! ```text
//! default,-falling,-beacon      every block that does not fall, minus the beacon
//! wools,concretes,-black_wool   two colour families, minus one block
//! ```
//!
//! A set is itself such a list, so a custom set from the config file may name
//! other sets and remove from them. A name that is not a set is taken as an
//! id without being checked here: whether an id has an embedded asset is only
//! asked of the ids the whole selection ends up with (see
//! [`Checker::select`](crate::cli::config)), so removing an id also removes
//! the complaint about it.
//!
//! Blocks and patterns have separate sets; the syntax is the same.

use std::collections::BTreeMap;

use crate::block;
use crate::color::COLOR_NAMES;
use crate::pattern;

/// The set a selection starts from when none is given: every embedded id.
pub const DEFAULT: &str = "default";

/// The patterns a loom only offers with a banner pattern item in its pattern
/// slot; every other pattern needs nothing but dye.
pub const ITEM_PATTERNS: [&str; 10] = [
    "bricks",
    "creeper",
    "curly_border",
    "flow",
    "flower",
    "globe",
    "guster",
    "mojang",
    "piglin",
    "skull",
];

/// Why a selection could not be resolved.
#[derive(Debug, PartialEq, Eq)]
pub enum Problem {
    /// A set reaches itself; the names are the path, first and last equal.
    Cycle(Vec<String>),
    /// An entry with no name: `""` or a lone `"-"`.
    EmptyEntry,
}

/// The ids a selection ends up with: each id, and the set that named it
/// (`None` when the selection named the id itself).
pub type Selection = BTreeMap<String, Option<String>>;

/// Every set a selection may name: the built-in ones and the config file's.
pub struct Sets {
    /// Set name to its entries, as written.
    sets: BTreeMap<String, Vec<String>>,
}

impl Sets {
    /// The built-in sets of one kind plus `custom`. The caller has already
    /// refused custom names that collide with a built-in set or an id.
    pub fn new(
        builtin: BTreeMap<String, Vec<String>>,
        custom: BTreeMap<String, Vec<String>>,
    ) -> Self {
        let mut sets = builtin;
        sets.extend(custom);
        Self { sets }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.sets.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.sets.keys()
    }

    /// Apply `entries` left to right.
    pub fn resolve(&self, entries: &[String]) -> Result<Selection, Problem> {
        self.apply(entries, None, &mut Vec::new())
    }

    /// `owner` is the set `entries` belongs to and `stack` the sets being
    /// expanded around it, outermost first.
    fn apply(
        &self,
        entries: &[String],
        owner: Option<&str>,
        stack: &mut Vec<String>,
    ) -> Result<Selection, Problem> {
        let mut out = Selection::new();
        for entry in entries {
            let (remove, name) = match entry.strip_prefix('-') {
                Some(name) => (true, name),
                None => (false, entry.as_str()),
            };
            if name.is_empty() {
                return Err(Problem::EmptyEntry);
            }
            let Some(inner) = self.sets.get(name) else {
                if remove {
                    out.remove(name);
                } else {
                    out.insert(name.to_string(), owner.map(str::to_string));
                }
                continue;
            };
            if let Some(start) = stack.iter().position(|s| s == name) {
                let mut path = stack[start..].to_vec();
                path.push(name.to_string());
                return Err(Problem::Cycle(path));
            }
            stack.push(name.to_string());
            let members = self.apply(inner, Some(name), stack)?;
            stack.pop();
            if remove {
                for id in members.keys() {
                    out.remove(id);
                }
            } else {
                out.extend(members);
            }
        }
        Ok(out)
    }
}

/// The built-in pattern sets, by name: [`DEFAULT`], every embedded pattern,
/// and `dye_only`, the same without [`ITEM_PATTERNS`].
pub fn pattern_sets() -> BTreeMap<String, Vec<String>> {
    let all = pattern::ids();
    let dye_only = all
        .iter()
        .filter(|id| !ITEM_PATTERNS.contains(&id.as_str()))
        .cloned()
        .collect();
    BTreeMap::from([
        (DEFAULT.to_string(), all),
        ("dye_only".to_string(), dye_only),
    ])
}

/// The built-in block sets, by name. [`DEFAULT`] is every embedded block; the
/// rest are fixed lists.
pub fn block_sets() -> BTreeMap<String, Vec<String>> {
    let dyed = |suffix: &str| -> Vec<String> {
        COLOR_NAMES
            .iter()
            .map(|color| format!("{color}_{suffix}"))
            .collect()
    };
    let list = |ids: &[&str]| -> Vec<String> { ids.iter().map(|id| id.to_string()).collect() };
    let wood = |kinds: &[&str], suffix: &str| -> Vec<String> {
        kinds.iter().map(|k| format!("{k}_{suffix}")).collect()
    };

    let mut terracottas = dyed("terracotta");
    terracottas.push("terracotta".to_string());

    let mut falling = dyed("concrete_powder");
    falling.extend(list(&["gravel", "red_sand", "sand"]));

    const OVERWORLD_WOODS: [&str; 9] = [
        "acacia", "birch", "cherry", "dark_oak", "jungle", "mangrove", "oak", "pale_oak", "spruce",
    ];
    let mut planks = wood(&OVERWORLD_WOODS, "planks");
    planks.extend(list(&["bamboo_planks", "crimson_planks", "warped_planks"]));

    let mut ores = wood(
        &[
            "coal", "copper", "diamond", "emerald", "gold", "iron", "lapis", "redstone",
        ],
        "ore",
    );
    ores.extend(ores.clone().iter().map(|ore| format!("deepslate_{ore}")));
    ores.extend(list(&["nether_gold_ore", "nether_quartz_ore"]));

    BTreeMap::from(
        [
            (DEFAULT, block::ids()),
            ("wools", dyed("wool")),
            ("concretes", dyed("concrete")),
            ("concrete_powders", dyed("concrete_powder")),
            ("terracottas", terracottas),
            ("glazed_terracottas", dyed("glazed_terracotta")),
            ("logs", wood(&OVERWORLD_WOODS, "log")),
            ("planks", planks),
            ("ores", ores),
            // Blocks that fall when nothing is under them.
            ("falling", falling),
            (
                "expensive",
                list(&[
                    "ancient_debris",
                    "beacon",
                    "diamond_block",
                    "emerald_block",
                    "gold_block",
                    "netherite_block",
                ]),
            ),
        ]
        .map(|(name, ids)| (name.to_string(), ids)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn ids(selection: &Selection) -> Vec<&str> {
        selection.keys().map(String::as_str).collect()
    }

    /// A built-in set that names an id with no asset is only reported when
    /// someone selects that set, so it is checked here for all of them; a set
    /// that shares a name with an id would make that id unselectable.
    #[test]
    fn builtin_sets_name_only_embedded_ids_and_no_id() {
        for (ids, sets) in [
            (block::ids(), block_sets()),
            (pattern::ids(), pattern_sets()),
        ] {
            for (name, members) in sets {
                assert!(!ids.contains(&name), "set '{name}' is also an id");
                assert!(!members.is_empty(), "set '{name}' is empty");
                for id in members {
                    assert!(ids.contains(&id), "set '{name}' names unknown '{id}'");
                }
            }
        }
    }

    /// A misspelt item pattern would stay in `dye_only` without any error.
    #[test]
    fn dye_only_leaves_out_exactly_the_item_patterns() {
        let sets = pattern_sets();
        assert_eq!(
            sets["dye_only"].len() + ITEM_PATTERNS.len(),
            sets[DEFAULT].len()
        );
    }

    #[test]
    fn entries_apply_left_to_right() {
        let custom = BTreeMap::from([("pair".to_string(), strings(&["sand", "gravel"]))]);
        let sets = Sets::new(BTreeMap::new(), custom);

        let removed = sets.resolve(&strings(&["pair", "stone", "-sand"])).unwrap();
        assert_eq!(ids(&removed), ["gravel", "stone"]);

        // A removal only affects what is already selected.
        let readded = sets.resolve(&strings(&["-sand", "pair"])).unwrap();
        assert_eq!(ids(&readded), ["gravel", "sand"]);

        let minus_set = sets.resolve(&strings(&["pair", "stone", "-pair"])).unwrap();
        assert_eq!(ids(&minus_set), ["stone"]);
    }

    #[test]
    fn a_set_may_name_and_remove_from_other_sets() {
        let custom = BTreeMap::from([
            ("inner".to_string(), strings(&["sand", "gravel", "typo"])),
            ("outer".to_string(), strings(&["inner", "-typo", "stone"])),
        ]);
        let selection = Sets::new(BTreeMap::new(), custom)
            .resolve(&strings(&["outer"]))
            .unwrap();
        assert_eq!(
            selection,
            Selection::from([
                ("gravel".to_string(), Some("inner".to_string())),
                ("sand".to_string(), Some("inner".to_string())),
                ("stone".to_string(), Some("outer".to_string())),
            ])
        );
    }

    #[test]
    fn a_set_that_reaches_itself_is_a_cycle() {
        let custom = BTreeMap::from([
            ("a".to_string(), strings(&["b"])),
            ("b".to_string(), strings(&["stone", "-a"])),
        ]);
        assert_eq!(
            Sets::new(BTreeMap::new(), custom).resolve(&strings(&["a"])),
            Err(Problem::Cycle(strings(&["a", "b", "a"])))
        );
    }
}
