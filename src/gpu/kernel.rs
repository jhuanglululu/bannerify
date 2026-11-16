//! What the solver and the device layer share: the kernel list, the buffer
//! binding indices, the device limits a batch is sized by and the marker for
//! plain-old-data buffer contents.

/// Every kernel in `kernels.wgsl`, in the order the pipelines are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kernel {
    GreedyBase,
    GreedyMoments,
    GreedyPick,
    PrefixRange,
    IdentitySuffix,
    Rung1,
    SelectTop,
    ShortlistTop,
    Rung2,
    BeamStep,
    BeamSuffix,
    CommitWindow,
    ChainSuffix,
    ReoptExact,
    ReoptPick,
    ReoptClosed,
    Finish,
}

impl Kernel {
    pub(super) const ALL: [Kernel; 17] = [
        Kernel::GreedyBase,
        Kernel::GreedyMoments,
        Kernel::GreedyPick,
        Kernel::PrefixRange,
        Kernel::IdentitySuffix,
        Kernel::Rung1,
        Kernel::SelectTop,
        Kernel::ShortlistTop,
        Kernel::Rung2,
        Kernel::BeamStep,
        Kernel::BeamSuffix,
        Kernel::CommitWindow,
        Kernel::ChainSuffix,
        Kernel::ReoptExact,
        Kernel::ReoptPick,
        Kernel::ReoptClosed,
        Kernel::Finish,
    ];

    /// The kernel's function name in the shader source.
    pub(super) fn name(self) -> &'static str {
        match self {
            Kernel::GreedyBase => "greedy_base",
            Kernel::GreedyMoments => "greedy_moments",
            Kernel::GreedyPick => "greedy_pick",
            Kernel::PrefixRange => "prefix_range",
            Kernel::IdentitySuffix => "identity_suffix",
            Kernel::Rung1 => "rung1",
            Kernel::SelectTop => "select_top",
            Kernel::ShortlistTop => "shortlist_top",
            Kernel::Rung2 => "rung2",
            Kernel::BeamStep => "beam_step",
            Kernel::BeamSuffix => "beam_suffix",
            Kernel::CommitWindow => "commit_window",
            Kernel::ChainSuffix => "chain_suffix",
            Kernel::ReoptExact => "reopt_exact",
            Kernel::ReoptPick => "reopt_pick",
            Kernel::ReoptClosed => "reopt_closed",
            Kernel::Finish => "finish",
        }
    }
}

/// Buffer binding indices, shared by every kernel (`kernels.wgsl` declares
/// the same numbers).
#[derive(Clone, Copy, Debug)]
pub(super) enum Bind {
    Cells = 0,
    Alpha = 1,
    Alpha2 = 2,
    Slots = 3,
    Sols = 4,
    Layers = 5,
    Prefixes = 6,
    Suffixes = 7,
    Beam = 8,
    Chains = 9,
    Best = 10,
    Shortlist = 11,
    Rung1 = 12,
    Rung2 = 13,
    Reopt = 14,
}

/// Types whose every bit pattern is a valid value and which have no padding,
/// so a shared buffer's bytes can be viewed as a slice of them.
///
/// # Safety
///
/// Implementors must be `#[repr(C)]` (or primitive), contain no padding and
/// accept any bit pattern.
pub(super) unsafe trait Plain: Copy {}

// SAFETY: primitive numeric types accept every bit pattern and have no padding.
unsafe impl Plain for f32 {}
// SAFETY: as above.
unsafe impl Plain for u32 {}

/// What a device allows, as far as batch sizing cares.
#[derive(Clone, Copy, Debug)]
pub(super) struct Limits {
    /// Largest single buffer, in bytes.
    pub max_buffer_bytes: usize,
    /// Most threads one dispatch may cover.
    pub max_threads: usize,
}
