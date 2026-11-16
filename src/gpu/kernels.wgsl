// The GPU solver's kernels: greedy fill, windowed beam refinement and the
// per-slot bookkeeping between them (see `src/gpu/schedule.rs` for the order
// they run in).
//
// The host prepends a prelude with the workgroup size `GROUP` and the constant
// tables (`PL`, `NC`, `COLORS`, `WSQ`, `W0..W2`, `LIN`, `M1_*`, `M2_*`),
// generated from the same Rust constants the CPU solver uses.
//
// Every per-pixel expression is the CPU kernel's, with `fma` where the CPU
// calls `mul_add` and one accumulator lane per pixel of a `vec4`. The driver
// compiles in fast math, so it may still fuse a multiply and an add or
// approximate a square root: results agree with the CPU solver to a few parts
// in 10^7, not bit for bit.
//
// Things WGSL or the drivers impose:
//
// - Empty list entries carry `BIG` instead of infinity, which a driver is
//   allowed to assume never occurs. Real errors are far below it.
// - A dispatch is two-dimensional, because one dimension is limited to 65,535
//   workgroups: a thread's index is `x + y * P.row`.
// - WGSL cannot return a pointer into one of two buffers, so the suffix a
//   beam candidate carries (`suffixes` at window step 0, `beam` after) is
//   bound by the host as `sfx_src` for the two hot kernels.
//
// Planes are `PL` `vec4`s. Per slot, in order of the buffers: prefixes
// `[nl + 1][3]` planes, suffixes `[nl + 1][4]` planes (mul, then the three
// adds), beam suffixes `[2 generations][cand][4]` planes. Per cell: nine
// planes, the blended sRGB target, the OKLab target and the OKLab feature map.

struct Params {
    n_slots: u32,    // slots in this batch
    nl: u32,         // layers per slot
    n_pat: u32,      // patterns
    cand: u32,       // beam width
    n_exact: u32,    // shortlist length; 0 when the exact rung is off
    window: u32,     // `window_size`
    layer: u32,      // the layer a step re-chooses, or the first of a range
    step: u32,       // window step `k`, or the end of a layer range
    generation: u32, // beam generation the step reads; it writes `generation ^ 1`
    start: u32,      // the window's start layer
    flag: u32,       // kernel-specific switch
    row: u32,        // threads per dispatch row: thread id = x + y * row
    lambda: f32,     // feature weight
    threshold: f32,  // `error_threshold`
    pad1: f32,
    pad2: f32,
}

struct Slot {
    cell: u32,
    start: u32,
}

struct Sol {
    base: u32,
    n_cand: u32,
    greedy_err: f32,
    rank_err: f32,
    lab_err: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Scored {
    err: f32,
    cand: u32,
    p: u32,
    c: u32,
}

@group(0) @binding(0) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> alpha: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> alpha2: array<f32>;
@group(0) @binding(3) var<storage, read> slots: array<Slot>;
@group(0) @binding(4) var<storage, read_write> sols: array<Sol>;
@group(0) @binding(5) var<storage, read_write> layers: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> prefixes: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> suffixes: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> beam: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> chains: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read_write> best: array<Scored>;
@group(0) @binding(11) var<storage, read_write> shortlist: array<Scored>;
@group(0) @binding(12) var<storage, read_write> r1: array<f32>;
@group(0) @binding(13) var<storage, read_write> r2: array<f32>;
@group(0) @binding(14) var<storage, read_write> reopt: array<f32>;
@group(0) @binding(15) var<storage, read> sfx_src: array<vec4<f32>>;

var<immediate> P: Params;

/// The error of an empty list entry.
const BIG: f32 = 3.0e38;

// ------------------------------------------------------------------ addressing

fn prefix_at(slot: u32, layer: u32, ch: u32) -> u32 {
    return ((slot * (P.nl + 1u) + layer) * 3u + ch) * PL;
}

fn suffix_at(slot: u32, layer: u32) -> u32 {
    return (slot * (P.nl + 1u) + layer) * 4u * PL;
}

fn beam_at(slot: u32, generation: u32, cand: u32) -> u32 {
    return ((slot * 2u + generation) * P.cand + cand) * 4u * PL;
}

fn chain_at(slot: u32, generation: u32, cand: u32) -> u32 {
    return ((slot * 2u + generation) * P.cand + cand) * P.window;
}

fn cell_at(cell: u32, q: u32) -> u32 {
    return (cell * 9u + q) * PL;
}

/// Where the suffix beam candidate `cand` carries into the current window step
/// starts, in `suffixes` at step 0 and in `beam` after.
fn prev_suffix_at(slot: u32, cand: u32) -> u32 {
    if (P.step == 0u) {
        return suffix_at(slot, P.start + 1u);
    }
    return beam_at(slot, P.generation, cand);
}

/// Beam candidates entering the current step: one at step 0.
fn entering(slot: u32) -> u32 {
    if (P.step == 0u) {
        return 1u;
    }
    return sols[slot].n_cand;
}

// ----------------------------------------------------------------- arithmetic

fn hsum(v: vec4<f32>) -> f32 {
    return (v.x + v.y) + (v.z + v.w);
}

fn linearize(v: vec4<f32>) -> vec4<f32> {
    let i = vec4<u32>(clamp(v + 0.5, vec4<f32>(0.0), vec4<f32>(255.0)));
    return vec4<f32>(LIN[i.x], LIN[i.y], LIN[i.z], LIN[i.w]);
}

/// The cube root, per lane: a `pow` seed followed by one Newton step whose
/// residual `y³ − x` has the rounding error of `y²` added back, which brings
/// the result to within a last-digit difference of the CPU's `cbrt`. Inputs
/// are LMS values, never negative; `0` maps to `0`.
fn cube_root(x: vec4<f32>) -> vec4<f32> {
    let y = pow(x, vec4<f32>(1.0 / 3.0));
    let y2 = y * y;
    let e = fma(y2, y, -x) + fma(y, y, -y2) * y;
    let r = y - e / (3.0 * y2);
    return select(r, vec4<f32>(0.0), x == vec4<f32>(0.0));
}

fn matrix_row(r0: f32, r1_: f32, r2_: f32, x: vec4<f32>, y: vec4<f32>, z: vec4<f32>) -> vec4<f32> {
    return fma(x, vec4<f32>(r0), fma(y, vec4<f32>(r1_), z * vec4<f32>(r2_)));
}

struct Lab {
    l: vec4<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
}

fn oklab(r: vec4<f32>, g: vec4<f32>, b: vec4<f32>) -> Lab {
    let lr = linearize(r);
    let lg = linearize(g);
    let lb = linearize(b);
    let l = cube_root(matrix_row(M1_00, M1_01, M1_02, lr, lg, lb));
    let m = cube_root(matrix_row(M1_10, M1_11, M1_12, lr, lg, lb));
    let s = cube_root(matrix_row(M1_20, M1_21, M1_22, lr, lg, lb));
    return Lab(
        matrix_row(M2_00, M2_01, M2_02, l, m, s),
        matrix_row(M2_10, M2_11, M2_12, l, m, s),
        matrix_row(M2_20, M2_21, M2_22, l, m, s),
    );
}

fn delta_e(a: Lab, l: vec4<f32>, aa: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let dl = a.l - l;
    let da = a.a - aa;
    let db = a.b - b;
    return sqrt(fma(dl, dl, fma(da, da, db * db)));
}

/// One lane step of the ranking error (`lambda > 0`) or the pure-target error
/// (`lambda == 0`) for sRGB pixels `(r, g, b)` at plane position `j` of `cell`.
fn ranking_term(cell: u32, j: u32, lambda: f32, r: vec4<f32>, g: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let lab = oklab(r, g, b);
    let tde = delta_e(lab, cells[cell_at(cell, 3u) + j], cells[cell_at(cell, 4u) + j],
                      cells[cell_at(cell, 5u) + j]);
    if (lambda == 0.0) {
        return tde;
    }
    let feature = delta_e(lab, cells[cell_at(cell, 6u) + j], cells[cell_at(cell, 7u) + j],
                          cells[cell_at(cell, 8u) + j]);
    return fma(feature, vec4<f32>(lambda), tde);
}

/// The closed-form dye sweep shared by every rung-1 kernel.
fn sweep(m2: vec3<f32>, m1: vec3<f32>, q: f32, d: u32) -> f32 {
    return W0 * (m2.x + m1.x * COLORS[d].x)
         + W1 * (m2.y + m1.y * COLORS[d].y)
         + W2 * (m2.z + m1.z * COLORS[d].z)
         + q * WSQ[d];
}

/// `refine::push_top` on the `n` entries of `best` starting at `at`.
fn push_best(at: u32, n: u32, e: Scored) {
    if (e.err >= best[at + n - 1u].err) {
        return;
    }
    var i = n - 1u;
    while (i > 0u && best[at + i - 1u].err > e.err) {
        best[at + i] = best[at + i - 1u];
        i -= 1u;
    }
    best[at + i] = e;
}

// ---------------------------------------------------------------- greedy fill

/// One thread per slot: `greedy::best_base`.
@compute @workgroup_size(GROUP)
fn greedy_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    let s = slots[id];
    var t2 = vec3<f32>(0.0);
    var n2t = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch++) {
        let tar = cell_at(s.cell, ch);
        var acc2 = vec4<f32>(0.0);
        var acc = vec4<f32>(0.0);
        for (var j = s.start / 4u; j < PL; j++) {
            let t = cells[tar + j];
            acc2 = fma(t, t, acc2);
            acc = acc + t;
        }
        t2[ch] = hsum(acc2);
        n2t[ch] = -2.0 * hsum(acc);
    }
    let hw = f32(PL * 4u - s.start);
    var base = 0u;
    var min_err = BIG;
    for (var c = 0u; c < NC; c++) {
        let err = sweep(t2, n2t, hw, c);
        if (err < min_err) {
            base = c;
            min_err = err;
        }
    }
    sols[id].base = base;
    sols[id].greedy_err = min_err;
}

/// One thread per (slot, pattern): the greedy layer sweep for layer
/// `P.layer`, that pattern's best dye. Writes `(error, dye)` to `r1`.
@compute @workgroup_size(GROUP)
fn greedy_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * P.n_pat) { return; }
    let p = id % P.n_pat;
    let slot = id / P.n_pat;
    let s = slots[slot];
    let al = p * PL;
    let pr = prefix_at(slot, P.layer, 0u);
    let ta = cell_at(s.cell, 0u);
    var a2r = vec4<f32>(0.0);
    var a2g = vec4<f32>(0.0);
    var a2b = vec4<f32>(0.0);
    var aar = vec4<f32>(0.0);
    var aag = vec4<f32>(0.0);
    var aab = vec4<f32>(0.0);
    for (var j = s.start / 4u; j < PL; j++) {
        let alp = alpha[al + j];
        let om = 1.0 - alp;
        let res_r = fma(prefixes[pr + j], om, -cells[ta + j]);
        let res_g = fma(prefixes[pr + PL + j], om, -cells[ta + PL + j]);
        let res_b = fma(prefixes[pr + 2u * PL + j], om, -cells[ta + 2u * PL + j]);
        a2r = fma(res_r, res_r, a2r);
        aar = fma(res_r, alp, aar);
        a2g = fma(res_g, res_g, a2g);
        aag = fma(res_g, alp, aag);
        a2b = fma(res_b, res_b, a2b);
        aab = fma(res_b, alp, aab);
    }
    let m2 = vec3<f32>(hsum(a2r), hsum(a2g), hsum(a2b));
    let m1 = 2.0 * vec3<f32>(hsum(aar), hsum(aag), hsum(aab));
    var a2_at = p;
    if (s.start != 0u) {
        a2_at = P.n_pat + p;
    }
    let a2 = alpha2[a2_at];
    var best_c = 0u;
    var min_err = BIG;
    for (var c = 0u; c < NC; c++) {
        let err = sweep(m2, m1, a2, c);
        if (err < min_err) {
            best_c = c;
            min_err = err;
        }
    }
    r1[id * 2u] = min_err;
    r1[id * 2u + 1u] = f32(best_c);
}

/// One thread per slot: the greedy argmin over patterns for layer `P.layer`.
@compute @workgroup_size(GROUP)
fn greedy_pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    var pick = vec2<u32>(0u, 0u);
    var min_err = BIG;
    for (var p = 0u; p < P.n_pat; p++) {
        let i = id * P.n_pat + p;
        if (r1[i * 2u] < min_err) {
            min_err = r1[i * 2u];
            pick = vec2<u32>(p, u32(r1[i * 2u + 1u]));
        }
    }
    layers[id * P.nl + P.layer] = pick;
    sols[id].greedy_err = min_err;
}

// ------------------------------------------------------------- prefix chains

/// One thread per (slot, plane position): composite layers `P.layer` up to
/// (not including) `P.step` forward, writing `prefixes[l + 1]` for each. With
/// `P.flag` set the chain starts from a flat fill of the slot's base dye,
/// written to `prefixes[0]`; otherwise from `prefixes[P.layer]` as stored.
@compute @workgroup_size(GROUP)
fn prefix_range(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * PL) { return; }
    let slot = id / PL;
    let j = id % PL;
    if (j < slots[slot].start / 4u) { return; }
    var vr: vec4<f32>;
    var vg: vec4<f32>;
    var vb: vec4<f32>;
    if (P.flag != 0u) {
        let c = COLORS[sols[slot].base];
        vr = vec4<f32>(c.x);
        vg = vec4<f32>(c.y);
        vb = vec4<f32>(c.z);
        let at = prefix_at(slot, 0u, 0u);
        prefixes[at + j] = vr;
        prefixes[at + PL + j] = vg;
        prefixes[at + 2u * PL + j] = vb;
    } else {
        let at = prefix_at(slot, P.layer, 0u);
        vr = prefixes[at + j];
        vg = prefixes[at + PL + j];
        vb = prefixes[at + 2u * PL + j];
    }
    for (var l = P.layer; l < P.step; l++) {
        let pc = layers[slot * P.nl + l];
        let alp = alpha[pc.x * PL + j];
        let om = 1.0 - alp;
        let c = COLORS[pc.y];
        vr = fma(vr, om, vec4<f32>(c.x) * alp);
        vg = fma(vg, om, vec4<f32>(c.y) * alp);
        vb = fma(vb, om, vec4<f32>(c.z) * alp);
        let at = prefix_at(slot, l + 1u, 0u);
        prefixes[at + j] = vr;
        prefixes[at + PL + j] = vg;
        prefixes[at + 2u * PL + j] = vb;
    }
}

/// One thread per (slot, plane position): `suffixes[nl]` is the identity.
@compute @workgroup_size(GROUP)
fn identity_suffix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * PL) { return; }
    let slot = id / PL;
    let j = id % PL;
    if (j < slots[slot].start / 4u) { return; }
    let at = suffix_at(slot, P.nl);
    suffixes[at + j] = vec4<f32>(1.0);
    for (var q = 1u; q < 4u; q++) {
        suffixes[at + q * PL + j] = vec4<f32>(0.0);
    }
}

// ---------------------------------------------------------- refinement window

/// One thread per (slot, beam candidate, pattern): rung 1 of window step
/// `P.step` at layer `P.layer`. With the exact rung on it writes that
/// pattern's best `(error, dye)`; with it off, all 16 dye errors.
@compute @workgroup_size(GROUP)
fn rung1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * P.cand * P.n_pat) { return; }
    let p = id % P.n_pat;
    let rest = id / P.n_pat;
    let cand = rest % P.cand;
    let slot = rest / P.cand;
    if (cand >= entering(slot)) { return; }
    let s = slots[slot];

    let sf = prev_suffix_at(slot, cand);
    let al = p * PL;
    let pr = prefix_at(slot, P.layer, 0u);
    let ta = cell_at(s.cell, 0u);

    var a2r = vec4<f32>(0.0);
    var a2g = vec4<f32>(0.0);
    var a2b = vec4<f32>(0.0);
    var aar = vec4<f32>(0.0);
    var aag = vec4<f32>(0.0);
    var aab = vec4<f32>(0.0);
    var acc_e = vec4<f32>(0.0);
    for (var j = s.start / 4u; j < PL; j++) {
        let alp = alpha[al + j];
        let mul = sfx_src[sf + j];
        let eff = alp * mul;
        let om = 1.0 - alp;
        acc_e = fma(eff, eff, acc_e);
        let res_r = fma(prefixes[pr + j] * om, mul, sfx_src[sf + PL + j]) - cells[ta + j];
        let res_g = fma(prefixes[pr + PL + j] * om, mul, sfx_src[sf + 2u * PL + j]) - cells[ta + PL + j];
        let res_b = fma(prefixes[pr + 2u * PL + j] * om, mul, sfx_src[sf + 3u * PL + j]) - cells[ta + 2u * PL + j];
        a2r = fma(res_r, res_r, a2r);
        aar = fma(res_r, eff, aar);
        a2g = fma(res_g, res_g, a2g);
        aag = fma(res_g, eff, aag);
        a2b = fma(res_b, res_b, a2b);
        aab = fma(res_b, eff, aab);
    }
    let m2 = vec3<f32>(hsum(a2r), hsum(a2g), hsum(a2b));
    let m1 = 2.0 * vec3<f32>(hsum(aar), hsum(aag), hsum(aab));
    let eff_a2 = hsum(acc_e);

    if (P.n_exact == 0u) {
        for (var c = 0u; c < NC; c++) {
            r1[id * NC + c] = sweep(m2, m1, eff_a2, c);
        }
    } else {
        var best_c = 0u;
        var min_err = BIG;
        for (var c = 0u; c < NC; c++) {
            let err = sweep(m2, m1, eff_a2, c);
            if (err < min_err) {
                best_c = c;
                min_err = err;
            }
        }
        r1[id * 2u] = min_err;
        r1[id * 2u + 1u] = f32(best_c);
    }
}

/// One thread per (slot, candidate, pattern): the exact rung's shortlist, the
/// `n_exact` best of each (candidate, pattern)'s best dye, each entry placed at
/// its stable rank.
///
/// The CPU builds the list by inserting in (candidate, pattern) order and
/// keeping the earlier entry on equal error, which leaves the first `n_exact`
/// entries of a stable sort by error. So each entry's place is the number of
/// entries with a lower error or an equal one inserted earlier, and every
/// thread computes its own entry's rank independently. Places no real entry
/// fills get the empty sentinel.
@compute @workgroup_size(GROUP)
fn shortlist_top(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    let per = P.cand * P.n_pat;
    if (id >= P.n_slots * per) { return; }
    let slot = id / per;
    let i = id % per;
    let n = entering(slot) * P.n_pat;
    let row = slot * per * 2u;
    let list = slot * P.n_exact;

    var err = BIG;
    if (i < n) {
        err = r1[row + i * 2u];
    }
    var rank = 0u;
    var finite = 0u;
    for (var j = 0u; j < n; j++) {
        let e = r1[row + j * 2u];
        finite += u32(e < BIG);
        rank += u32(e < err || (e == err && j < i));
    }
    if (err < BIG && rank < P.n_exact) {
        shortlist[list + rank] = Scored(err, i / P.n_pat, i % P.n_pat, u32(r1[row + i * 2u + 1u]));
    }
    if (i >= finite && i < P.n_exact) {
        shortlist[list + i] = Scored(BIG, 0u, 0u, 0u);
    }
}

/// One thread per slot, exact rung off: every rung-1 dye error straight into
/// the beam's `best` list, inserted in (candidate, pattern, dye) order.
@compute @workgroup_size(GROUP)
fn select_top(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    let nc = entering(id);
    let list = id * P.cand;
    for (var i = 0u; i < P.cand; i++) {
        best[list + i] = Scored(BIG, 0u, 0u, 0u);
    }
    for (var cand = 0u; cand < nc; cand++) {
        for (var p = 0u; p < P.n_pat; p++) {
            let i = ((id * P.cand + cand) * P.n_pat + p) * NC;
            for (var c = 0u; c < NC; c++) {
                push_best(list, P.cand, Scored(r1[i + c], cand, p, c));
            }
        }
    }
}

/// One thread per (slot, shortlist entry): rung 2, the entry's full composite
/// scored by the ranking error.
@compute @workgroup_size(GROUP)
fn rung2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * P.n_exact) { return; }
    let slot = id / P.n_exact;
    let e = shortlist[id];
    if (!(e.err < BIG)) { return; }
    let s = slots[slot];

    let sf = prev_suffix_at(slot, e.cand);
    let al = e.p * PL;
    let pr = prefix_at(slot, P.layer, 0u);
    let col = COLORS[e.c];
    let c0 = vec4<f32>(col.x);
    let c1 = vec4<f32>(col.y);
    let c2 = vec4<f32>(col.z);

    var acc = vec4<f32>(0.0);
    for (var j = s.start / 4u; j < PL; j++) {
        let alp = alpha[al + j];
        let om = 1.0 - alp;
        let mul = sfx_src[sf + j];
        let r = fma(fma(prefixes[pr + j], om, c0 * alp), mul, sfx_src[sf + PL + j]);
        let g = fma(fma(prefixes[pr + PL + j], om, c1 * alp), mul, sfx_src[sf + 2u * PL + j]);
        let b = fma(fma(prefixes[pr + 2u * PL + j], om, c2 * alp), mul, sfx_src[sf + 3u * PL + j]);
        acc = acc + ranking_term(s.cell, j, P.lambda, r, g, b);
    }
    r2[id] = hsum(acc);
}

/// One thread per slot: the beam update of a window step. Ranks the exact
/// scores in shortlist order (or takes rung 1's list as is), prunes by
/// `error_threshold`, and writes the surviving candidates' layer chains into
/// generation `P.generation ^ 1`.
@compute @workgroup_size(GROUP)
fn beam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    let list = id * P.cand;
    if (P.n_exact > 0u) {
        for (var i = 0u; i < P.cand; i++) {
            best[list + i] = Scored(BIG, 0u, 0u, 0u);
        }
        for (var j = 0u; j < P.n_exact; j++) {
            var e = shortlist[id * P.n_exact + j];
            if (!(e.err < BIG)) { break; }
            e.err = r2[id * P.n_exact + j];
            push_best(list, P.cand, e);
        }
    }

    var finite = 0u;
    while (finite < P.cand && best[list + finite].err < BIG) {
        finite++;
    }
    finite = max(finite, 1u);
    var n = P.cand;
    if (P.threshold > 0.0) {
        let bound = best[list].err / P.threshold;
        n = 0u;
        while (n < P.cand && best[list + n].err <= bound) {
            n++;
        }
    }
    n = clamp(n, 1u, finite);

    for (var ci = 0u; ci < n; ci++) {
        let e = best[list + ci];
        let src_chain = chain_at(id, P.generation, e.cand);
        let dst_chain = chain_at(id, P.generation ^ 1u, ci);
        for (var i = 0u; i < P.step; i++) {
            chains[dst_chain + i] = chains[src_chain + i];
        }
        chains[dst_chain + P.step] = vec2<u32>(e.p, e.c);
        if (P.layer == 0u) { break; }
    }
    sols[id].n_cand = n;
}

/// One plane element of the suffix a candidate carried into this step.
fn carried(at: u32) -> vec4<f32> {
    if (P.step == 0u) {
        return suffixes[at];
    }
    return beam[at];
}

/// One thread per (slot, beam candidate, plane position): the suffix each
/// surviving candidate carries into the next step, into generation
/// `P.generation ^ 1`.
@compute @workgroup_size(GROUP)
fn beam_suffix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * P.cand * PL) { return; }
    let j = id % PL;
    let rest = id / PL;
    let cand = rest % P.cand;
    let slot = rest / P.cand;
    if (cand >= sols[slot].n_cand || j < slots[slot].start / 4u) { return; }
    let e = best[slot * P.cand + cand];
    let src = prev_suffix_at(slot, e.cand);
    let dst = beam_at(slot, P.generation ^ 1u, cand);
    let alp = alpha[e.p * PL + j];
    let smul = carried(src + j);
    let col = COLORS[e.c];
    let add_r = fma(vec4<f32>(col.x) * alp, smul, carried(src + PL + j));
    let add_g = fma(vec4<f32>(col.y) * alp, smul, carried(src + 2u * PL + j));
    let add_b = fma(vec4<f32>(col.z) * alp, smul, carried(src + 3u * PL + j));
    beam[dst + PL + j] = add_r;
    beam[dst + 2u * PL + j] = add_g;
    beam[dst + 3u * PL + j] = add_b;
    beam[dst + j] = (1.0 - alp) * smul;
}

/// One thread per slot: write the window's winning chain (candidate 0 of
/// generation `P.generation`, `P.step` layers, deepest first) into the layer list.
@compute @workgroup_size(GROUP)
fn commit_window(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    let src_chain = chain_at(id, P.generation, 0u);
    for (var i = 0u; i < P.step; i++) {
        layers[id * P.nl + P.start - i] = chains[src_chain + i];
    }
}

/// One thread per (slot, plane position): `suffixes[P.start]` from
/// `suffixes[P.start + 1]` and the window's now-fixed top layer.
@compute @workgroup_size(GROUP)
fn chain_suffix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * PL) { return; }
    let slot = id / PL;
    let j = id % PL;
    if (j < slots[slot].start / 4u) { return; }
    let pc = layers[slot * P.nl + P.start];
    let src = suffix_at(slot, P.start + 1u);
    let dst = suffix_at(slot, P.start);
    let alp = alpha[pc.x * PL + j];
    let smul = suffixes[src + j];
    let col = COLORS[pc.y];
    suffixes[dst + PL + j] = fma(vec4<f32>(col.x) * alp, smul, suffixes[src + PL + j]);
    suffixes[dst + 2u * PL + j] = fma(vec4<f32>(col.y) * alp, smul, suffixes[src + 2u * PL + j]);
    suffixes[dst + 3u * PL + j] = fma(vec4<f32>(col.z) * alp, smul, suffixes[src + 3u * PL + j]);
    suffixes[dst + j] = (1.0 - alp) * smul;
}

// ------------------------------------------------------------- end of a pass

/// One thread per (slot, dye): the exact score of base dye `c` under the whole
/// layer stack, `c·mul + add` with `suffixes[0]`.
@compute @workgroup_size(GROUP)
fn reopt_exact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots * NC) { return; }
    let slot = id / NC;
    let c = id % NC;
    let s = slots[slot];
    let at = suffix_at(slot, 0u);
    let col = COLORS[c];
    let c0 = vec4<f32>(col.x);
    let c1 = vec4<f32>(col.y);
    let c2 = vec4<f32>(col.z);
    var acc = vec4<f32>(0.0);
    for (var j = s.start / 4u; j < PL; j++) {
        let mul = suffixes[at + j];
        let r = fma(c0, mul, suffixes[at + PL + j]);
        let g = fma(c1, mul, suffixes[at + 2u * PL + j]);
        let b = fma(c2, mul, suffixes[at + 3u * PL + j]);
        acc = acc + ranking_term(s.cell, j, P.lambda, r, g, b);
    }
    reopt[id] = hsum(acc);
}

/// One thread per slot: the first minimum of `reopt_exact`'s 16 scores.
@compute @workgroup_size(GROUP)
fn reopt_pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    var pick = sols[id].base;
    var min_err = BIG;
    for (var c = 0u; c < NC; c++) {
        let err = reopt[id * NC + c];
        if (err < min_err) {
            min_err = err;
            pick = c;
        }
    }
    sols[id].base = pick;
}

/// One thread per slot: the closed-form base re-optimisation used with the
/// exact rung off.
@compute @workgroup_size(GROUP)
fn reopt_closed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    let s = slots[id];
    let at = suffix_at(id, 0u);
    var acc = vec4<f32>(0.0);
    for (var j = s.start / 4u; j < PL; j++) {
        let m = suffixes[at + j];
        acc = fma(m, m, acc);
    }
    let mul2 = hsum(acc);
    var d2 = vec3<f32>(0.0);
    var d_2m = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch++) {
        var acc2 = vec4<f32>(0.0);
        var acc_m = vec4<f32>(0.0);
        let tar = cell_at(s.cell, ch);
        for (var j = s.start / 4u; j < PL; j++) {
            let m = suffixes[at + j];
            let d = suffixes[at + (1u + ch) * PL + j] - cells[tar + j];
            acc2 = fma(d, d, acc2);
            acc_m = fma(d, m, acc_m);
        }
        d2[ch] = hsum(acc2);
        d_2m[ch] = 2.0 * hsum(acc_m);
    }
    var pick = sols[id].base;
    var min_err = BIG;
    for (var c = 0u; c < NC; c++) {
        let err = sweep(d2, d_2m, mul2, c);
        if (err < min_err) {
            min_err = err;
            pick = c;
        }
    }
    sols[id].base = pick;
}

/// One thread per slot: the ranking error and the pure-target error of the
/// final composite `prefixes[nl]`, in one pass.
@compute @workgroup_size(GROUP)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * P.row;
    if (id >= P.n_slots) { return; }
    let s = slots[id];
    let at = prefix_at(id, P.nl, 0u);
    var rank = vec4<f32>(0.0);
    var pure = vec4<f32>(0.0);
    for (var j = s.start / 4u; j < PL; j++) {
        let lab = oklab(prefixes[at + j], prefixes[at + PL + j], prefixes[at + 2u * PL + j]);
        let tde = delta_e(lab, cells[cell_at(s.cell, 3u) + j], cells[cell_at(s.cell, 4u) + j],
                          cells[cell_at(s.cell, 5u) + j]);
        pure = pure + tde;
        if (P.lambda != 0.0) {
            let feature = delta_e(lab, cells[cell_at(s.cell, 6u) + j],
                                  cells[cell_at(s.cell, 7u) + j], cells[cell_at(s.cell, 8u) + j]);
            rank = rank + fma(feature, vec4<f32>(P.lambda), tde);
        }
    }
    sols[id].lab_err = hsum(pure);
    if (P.lambda != 0.0) {
        sols[id].rank_err = hsum(rank);
    } else {
        sols[id].rank_err = hsum(pure);
    }
}
