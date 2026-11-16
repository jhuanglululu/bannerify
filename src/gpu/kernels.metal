// The GPU solver's kernels: greedy fill, windowed beam refinement and the
// per-slot bookkeeping between them (see `src/gpu/schedule.rs` for the order they run in).
//
// The host prepends a prelude with the constant tables (`HW`, `PL`, `NC`,
// `COLORS`, `WSQ`, `W0..W2`, `LIN`, `M1_*`, `M2_*`), generated from the same
// Rust constants the CPU solver uses.
//
// Every per-pixel expression is the CPU kernel's, with `fma` where the CPU
// calls `mul_add` and four accumulators per pixel sum. The source is compiled
// in fast math, so the compiler may still fuse, reorder or approximate:
// results agree with the CPU solver to a few parts in 10^7, not bit for bit.
//
// Buffer bindings are the same for every kernel (`Bind` in `src/gpu/metal.rs`):
// a kernel declares only the ones it reads or writes.

#include <metal_stdlib>
using namespace metal;

/// Per-dispatch parameters. Field meaning per kernel is documented at the
/// kernel; the layout matches `Params` in `src/gpu/state.rs`.
struct Params {
    uint n_slots;   // slots in this batch
    uint nl;        // layers per slot (every slot of a batch has the same count)
    uint n_pat;     // patterns
    uint cand;      // beam width, `refinement_candidate` (at least 1)
    uint n_exact;   // shortlist length; 0 when the exact rung is off
    uint window;    // `window_size`
    uint layer;     // the layer a step re-chooses, or the first layer of a range
    uint step;      // window step `k`, or the end of a layer range
    uint generation; // beam generation the step reads; it writes `generation ^ 1`
    uint start;     // the window's start layer
    uint flag;      // kernel-specific switch
    uint row;       // used by the WGSL kernels only
    float lambda;   // feature weight
    float threshold;// `error_threshold`
    float pad1;
    float pad2;
};

/// Which cell a slot solves and the first plane element it solves (0 for
/// banner row 0, 320 for every other row).
struct Slot {
    uint cell;
    uint start;
};

/// A slot's scalar solver state.
struct Sol {
    uint base;
    uint n_cand;      // beam candidates entering the next window step
    float greedy_err; // the greedy fill's closed-form error
    float rank_err;   // ranking error of the final composite
    float lab_err;    // pure-target `Σ ΔE` of the final composite
    uint pad0;
    uint pad1;
    uint pad2;
};

/// `(error, beam candidate, pattern, dye)`, as the CPU's `Scored`.
struct Scored {
    float err;
    uint cand;
    uint p;
    uint c;
};

// ------------------------------------------------------------------ addressing
//
// Planes are `PL` float4s (`HW` floats). Per slot, in order of the buffers:
// prefixes `[nl + 1][3]` planes, suffixes `[nl + 1][4]` planes (mul, then the
// three adds), beam suffixes `[2 generations][cand][4]` planes. Per cell: nine
// planes, the blended sRGB target, the OKLab target and the OKLab feature map.

inline uint prefix_at(constant Params& P, uint slot, uint layer, uint ch) {
    return ((slot * (P.nl + 1) + layer) * 3 + ch) * PL;
}

inline uint suffix_at(constant Params& P, uint slot, uint layer) {
    return (slot * (P.nl + 1) + layer) * 4 * PL;
}

inline uint beam_at(constant Params& P, uint slot, uint generation, uint cand) {
    return ((slot * 2 + generation) * P.cand + cand) * 4 * PL;
}

inline uint chain_at(constant Params& P, uint slot, uint generation, uint cand) {
    return ((slot * 2 + generation) * P.cand + cand) * P.window;
}

inline uint cell_at(uint cell, uint q) {
    return (cell * 9 + q) * PL;
}

/// The suffix beam candidate `cand` carries into the current window step. At
/// step 0 there is one candidate, the layers above the window as they are, so
/// it is read straight from the suffix chain instead of being copied.
inline device const float4* prev_suffix(device const float4* suffixes,
                                        device const float4* beam,
                                        constant Params& P, uint slot, uint cand) {
    if (P.step == 0) {
        return suffixes + suffix_at(P, slot, P.start + 1);
    }
    return beam + beam_at(P, slot, P.generation, cand);
}

/// Beam candidates entering the current step: one at step 0.
inline uint entering(device const Sol* sols, constant Params& P, uint slot) {
    return P.step == 0 ? 1u : sols[slot].n_cand;
}

// ----------------------------------------------------------------- arithmetic

/// `F32s::hsum` on NEON (`vaddvq_f32`).
inline float hsum(float4 v) {
    return (v.x + v.y) + (v.z + v.w);
}

/// `oklab::linearize`: round to the nearest byte, clamp, look up in `lin`
/// (`LIN`, copied to threadgroup memory by the kernel: every lane reads its own
/// entry, which `constant` memory serves one address at a time).
inline float4 linearize(threadgroup const float* lin, float4 v) {
    uint4 i = uint4(clamp(v + 0.5f, float4(0.0f), float4(255.0f)));
    return float4(lin[i.x], lin[i.y], lin[i.z], lin[i.w]);
}

/// Copy `LIN` into threadgroup memory. Every thread of the group must call
/// this before any of them returns.
inline void load_lin(threadgroup float* lin, uint tid, uint size) {
    for (uint i = tid; i < 256; i += size) {
        lin[i] = LIN[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// `f32::cbrt`, per lane. A fast `pow` seed (relative error around 1e-5)
/// followed by one Newton step whose residual `y³ − x` is computed with the
/// rounding error of `y²` added back (`fma(y, y, −y²)` is that error exactly),
/// so the step lands on the nearest float as libm's cube root does. Precise
/// `pow(x, 1/3)` is both slower and further from libm: its exponent `1/3` is
/// itself rounded. Inputs are LMS values, never negative; `0` maps to `0`.
inline float4 cube_root(float4 x) {
    float4 y = fast::pow(x, float4(1.0f / 3.0f));
    float4 y2 = y * y;
    float4 e = fma(y2, y, -x) + fma(y, y, -y2) * y;
    float4 r = y - e / (3.0f * y2);
    return select(r, float4(0.0f), x == 0.0f);
}

/// `oklab::matrix_row`.
inline float4 matrix_row(float r0, float r1, float r2, float4 x, float4 y, float4 z) {
    return fma(x, float4(r0), fma(y, float4(r1), z * float4(r2)));
}

struct Lab {
    float4 l;
    float4 a;
    float4 b;
};

/// `oklab::srgb_to_oklab`.
inline Lab oklab(threadgroup const float* lin, float4 r, float4 g, float4 b) {
    float4 lr = linearize(lin, r);
    float4 lg = linearize(lin, g);
    float4 lb = linearize(lin, b);
    float4 l = cube_root(matrix_row(M1_00, M1_01, M1_02, lr, lg, lb));
    float4 m = cube_root(matrix_row(M1_10, M1_11, M1_12, lr, lg, lb));
    float4 s = cube_root(matrix_row(M1_20, M1_21, M1_22, lr, lg, lb));
    Lab out;
    out.l = matrix_row(M2_00, M2_01, M2_02, l, m, s);
    out.a = matrix_row(M2_10, M2_11, M2_12, l, m, s);
    out.b = matrix_row(M2_20, M2_21, M2_22, l, m, s);
    return out;
}

/// `refine::delta_e`.
inline float4 delta_e(Lab a, float4 l, float4 aa, float4 b) {
    float4 dl = a.l - l;
    float4 da = a.a - aa;
    float4 db = a.b - b;
    return sqrt(fma(dl, dl, fma(da, da, db * db)));
}

/// One lane step of `refine::ranking_error` (`lambda > 0`) or
/// `refine::lab_error` (`lambda == 0`): the term added to the accumulator for
/// sRGB pixels `(r, g, b)` at plane position `j` of cell `cell`.
inline float4 ranking_term(threadgroup const float* lin, device const float4* cells, uint cell,
                           uint j, float lambda, float4 r, float4 g, float4 b) {
    Lab lab = oklab(lin, r, g, b);
    float4 target = delta_e(lab, cells[cell_at(cell, 3) + j], cells[cell_at(cell, 4) + j],
                            cells[cell_at(cell, 5) + j]);
    if (lambda == 0.0f) {
        return target;
    }
    float4 feature = delta_e(lab, cells[cell_at(cell, 6) + j], cells[cell_at(cell, 7) + j],
                             cells[cell_at(cell, 8) + j]);
    return fma(feature, float4(lambda), target);
}

/// The closed-form dye sweep shared by every rung-1 kernel:
/// `Σ_ch w·(m2 + m1·c) + q·Σ w·c²`, evaluated left to right as on the CPU.
inline float sweep(float3 m2, float3 m1, float q, uint d) {
    return W0 * (m2.x + m1.x * COLORS[d][0])
         + W1 * (m2.y + m1.y * COLORS[d][1])
         + W2 * (m2.z + m1.z * COLORS[d][2])
         + q * WSQ[d];
}

/// `refine::push_top`, on a list in device memory, with the CPU's exact
/// comparisons (so NaN and ties behave the same).
inline void push_top(device Scored* list, uint n, Scored e) {
    if (e.err >= list[n - 1].err) {
        return;
    }
    uint i = n - 1;
    while (i > 0 && list[i - 1].err > e.err) {
        list[i] = list[i - 1];
        i -= 1;
    }
    list[i] = e;
}

inline Scored no_cand() {
    Scored s;
    s.err = INFINITY;
    s.cand = 0;
    s.p = 0;
    s.c = 0;
    return s;
}

// ---------------------------------------------------------------- greedy fill

/// One thread per slot: `greedy::best_base`.
kernel void greedy_base(device const float4* cells [[buffer(0)]],
                        device const Slot* slots [[buffer(3)]],
                        device Sol* sols [[buffer(4)]],
                        constant Params& P [[buffer(15)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    Slot s = slots[id];
    float t2[3];
    float n2t[3];
    for (uint ch = 0; ch < 3; ch++) {
        device const float4* tar = cells + cell_at(s.cell, ch);
        float4 acc2 = float4(0.0f);
        float4 acc = float4(0.0f);
        for (uint j = s.start / 4; j < PL; j++) {
            float4 t = tar[j];
            acc2 = fma(t, t, acc2);
            acc = acc + t;
        }
        t2[ch] = hsum(acc2);
        n2t[ch] = -2.0f * hsum(acc);
    }
    float hw = float(HW - s.start);
    uint base = 0;
    float min_err = INFINITY;
    for (uint c = 0; c < NC; c++) {
        float err = sweep(float3(t2[0], t2[1], t2[2]), float3(n2t[0], n2t[1], n2t[2]), hw, c);
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
kernel void greedy_moments(device const float4* cells [[buffer(0)]],
                           device const float4* alpha [[buffer(1)]],
                           device const float* alpha2 [[buffer(2)]],
                           device const Slot* slots [[buffer(3)]],
                           device const float4* prefixes [[buffer(6)]],
                           device float* r1 [[buffer(12)]],
                           constant Params& P [[buffer(15)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots * P.n_pat) return;
    uint p = id % P.n_pat;
    uint slot = id / P.n_pat;
    Slot s = slots[slot];
    device const float4* alp_p = alpha + p * PL;
    float4 acc2[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 acc_a[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint j = s.start / 4; j < PL; j++) {
        float4 alp = alp_p[j];
        float4 om = 1.0f - alp;
        for (uint ch = 0; ch < 3; ch++) {
            float4 pre = prefixes[prefix_at(P, slot, P.layer, ch) + j];
            float4 tar = cells[cell_at(s.cell, ch) + j];
            float4 res = fma(pre, om, -tar);
            acc2[ch] = fma(res, res, acc2[ch]);
            acc_a[ch] = fma(res, alp, acc_a[ch]);
        }
    }
    float3 m2 = float3(hsum(acc2[0]), hsum(acc2[1]), hsum(acc2[2]));
    float3 m1 = float3(2.0f * hsum(acc_a[0]), 2.0f * hsum(acc_a[1]), 2.0f * hsum(acc_a[2]));
    float a2 = alpha2[(s.start == 0 ? 0u : P.n_pat) + p];
    uint best = 0;
    float min_err = INFINITY;
    for (uint c = 0; c < NC; c++) {
        float err = sweep(m2, m1, a2, c);
        if (err < min_err) {
            best = c;
            min_err = err;
        }
    }
    r1[id * 2] = min_err;
    r1[id * 2 + 1] = float(best);
}

/// One thread per slot: the greedy argmin over patterns for layer `P.layer`.
kernel void greedy_pick(device Sol* sols [[buffer(4)]],
                        device uint2* layers [[buffer(5)]],
                        device const float* r1 [[buffer(12)]],
                        constant Params& P [[buffer(15)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    uint2 best = uint2(0, 0);
    float min_err = INFINITY;
    for (uint p = 0; p < P.n_pat; p++) {
        uint i = id * P.n_pat + p;
        if (r1[i * 2] < min_err) {
            min_err = r1[i * 2];
            best = uint2(p, uint(r1[i * 2 + 1]));
        }
    }
    layers[id * P.nl + P.layer] = best;
    sols[id].greedy_err = min_err;
}

// ------------------------------------------------------------- prefix chains

/// One thread per (slot, plane position): composite layers `P.layer` up to
/// (not including) `P.step` forward, writing `prefixes[l + 1]` for each. With
/// `P.flag` set the chain starts from a flat fill of the slot's base dye,
/// written to `prefixes[0]` (`workspace::rebuild_prefixes`); otherwise from
/// `prefixes[P.layer]` as stored (`greedy::build_prefix`).
kernel void prefix_range(device const float4* alpha [[buffer(1)]],
                         device const Slot* slots [[buffer(3)]],
                         device const Sol* sols [[buffer(4)]],
                         device const uint2* layers [[buffer(5)]],
                         device float4* prefixes [[buffer(6)]],
                         constant Params& P [[buffer(15)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots * PL) return;
    uint slot = id / PL;
    uint j = id % PL;
    if (j < slots[slot].start / 4) return;
    float4 v[3];
    if (P.flag != 0) {
        uint base = sols[slot].base;
        for (uint ch = 0; ch < 3; ch++) {
            v[ch] = float4(COLORS[base][ch]);
            prefixes[prefix_at(P, slot, 0, ch) + j] = v[ch];
        }
    } else {
        for (uint ch = 0; ch < 3; ch++) {
            v[ch] = prefixes[prefix_at(P, slot, P.layer, ch) + j];
        }
    }
    for (uint l = P.layer; l < P.step; l++) {
        uint2 pc = layers[slot * P.nl + l];
        float4 alp = alpha[pc.x * PL + j];
        float4 om = 1.0f - alp;
        for (uint ch = 0; ch < 3; ch++) {
            v[ch] = fma(v[ch], om, float4(COLORS[pc.y][ch]) * alp);
            prefixes[prefix_at(P, slot, l + 1, ch) + j] = v[ch];
        }
    }
}

/// One thread per (slot, plane position): `suffixes[nl]` is the identity
/// (`refine::empty_suffix`).
kernel void identity_suffix(device const Slot* slots [[buffer(3)]],
                            device float4* suffixes [[buffer(7)]],
                            constant Params& P [[buffer(15)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots * PL) return;
    uint slot = id / PL;
    uint j = id % PL;
    if (j < slots[slot].start / 4) return;
    uint at = suffix_at(P, slot, P.nl);
    suffixes[at + j] = float4(1.0f);
    for (uint q = 1; q < 4; q++) {
        suffixes[at + q * PL + j] = float4(0.0f);
    }
}

// ---------------------------------------------------------- refinement window

/// One thread per (slot, beam candidate, pattern): rung 1 of window step
/// `P.step` at layer `P.layer` (`refine::moments_eff`, `refine::moments` and
/// the dye sweep). With the exact rung on it writes that pattern's best
/// `(error, dye)`; with it off, all 16 dye errors.
kernel void rung1(device const float4* cells [[buffer(0)]],
                  device const float4* alpha [[buffer(1)]],
                  device const Slot* slots [[buffer(3)]],
                  device const Sol* sols [[buffer(4)]],
                  device const float4* prefixes [[buffer(6)]],
                  device const float4* suffixes [[buffer(7)]],
                  device const float4* beam [[buffer(8)]],
                  device float* r1 [[buffer(12)]],
                  constant Params& P [[buffer(15)]],
                  uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots * P.cand * P.n_pat) return;
    uint p = id % P.n_pat;
    uint rest = id / P.n_pat;
    uint cand = rest % P.cand;
    uint slot = rest / P.cand;
    if (cand >= entering(sols, P, slot)) return;
    Slot s = slots[slot];

    device const float4* sfx = prev_suffix(suffixes, beam, P, slot, cand);
    device const float4* alp_p = alpha + p * PL;
    uint pre0 = prefix_at(P, slot, P.layer, 0);
    uint tar0 = cell_at(s.cell, 0);

    float4 acc2[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 acc_a[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 acc_e = float4(0.0f);
    for (uint j = s.start / 4; j < PL; j++) {
        float4 alp = alp_p[j];
        float4 mul = sfx[j];
        float4 eff = alp * mul;
        float4 om = 1.0f - alp;
        acc_e = fma(eff, eff, acc_e);
        for (uint ch = 0; ch < 3; ch++) {
            float4 pre = prefixes[pre0 + ch * PL + j];
            float4 tar = cells[tar0 + ch * PL + j];
            float4 add = sfx[(1 + ch) * PL + j];
            float4 res = fma(pre * om, mul, add) - tar;
            acc2[ch] = fma(res, res, acc2[ch]);
            acc_a[ch] = fma(res, eff, acc_a[ch]);
        }
    }
    float3 m2 = float3(hsum(acc2[0]), hsum(acc2[1]), hsum(acc2[2]));
    float3 m1 = float3(2.0f * hsum(acc_a[0]), 2.0f * hsum(acc_a[1]), 2.0f * hsum(acc_a[2]));
    float eff_a2 = hsum(acc_e);

    if (P.n_exact == 0) {
        for (uint c = 0; c < NC; c++) {
            r1[id * NC + c] = sweep(m2, m1, eff_a2, c);
        }
    } else {
        uint best = 0;
        float min_err = INFINITY;
        for (uint c = 0; c < NC; c++) {
            float err = sweep(m2, m1, eff_a2, c);
            if (err < min_err) {
                best = c;
                min_err = err;
            }
        }
        r1[id * 2] = min_err;
        r1[id * 2 + 1] = float(best);
    }
}

/// One thread per (slot, candidate, pattern): the exact rung's shortlist, the
/// `n_exact` best of each (candidate, pattern)'s best dye.
///
/// The CPU builds it with `push_top` in (candidate, pattern) order, which for
/// finite errors leaves exactly the first `n_exact` entries of a *stable* sort
/// by error. So each entry's place is its stable rank, the number of entries
/// with a lower error or an equal one inserted earlier, and every thread
/// computes its own entry's rank independently. Places no finite entry fills
/// keep the CPU's sentinel. (Errors are sums of squares, never NaN.)
kernel void shortlist_top(device const Sol* sols [[buffer(4)]],
                          device Scored* shortlist [[buffer(11)]],
                          device const float* r1 [[buffer(12)]],
                          constant Params& P [[buffer(15)]],
                          uint id [[thread_position_in_grid]]) {
    uint per = P.cand * P.n_pat;
    if (id >= P.n_slots * per) return;
    uint slot = id / per;
    uint i = id % per;
    uint n = entering(sols, P, slot) * P.n_pat;
    device const float* row = r1 + slot * per * 2;
    device Scored* list = shortlist + slot * P.n_exact;

    float err = i < n ? row[i * 2] : INFINITY;
    uint rank = 0;
    uint finite = 0;
    for (uint j = 0; j < n; j++) {
        float e = row[j * 2];
        finite += e < INFINITY ? 1u : 0u;
        rank += (e < err || (e == err && j < i)) ? 1u : 0u;
    }
    if (err < INFINITY && rank < P.n_exact) {
        Scored e;
        e.err = err;
        e.cand = i / P.n_pat;
        e.p = i % P.n_pat;
        e.c = uint(row[i * 2 + 1]);
        list[rank] = e;
    }
    if (i >= finite && i < P.n_exact) {
        list[i] = no_cand();
    }
}

/// One thread per slot, exact rung off: every rung-1 dye error straight into
/// the beam's `best` list, inserted in the CPU's (candidate, pattern, dye)
/// order.
kernel void select_top(device const Sol* sols [[buffer(4)]],
                       device Scored* best [[buffer(10)]],
                       device const float* r1 [[buffer(12)]],
                       constant Params& P [[buffer(15)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    uint nc = entering(sols, P, id);
    device Scored* list = best + id * P.cand;
    for (uint i = 0; i < P.cand; i++) list[i] = no_cand();
    for (uint cand = 0; cand < nc; cand++) {
        for (uint p = 0; p < P.n_pat; p++) {
            uint i = ((id * P.cand + cand) * P.n_pat + p) * NC;
            for (uint c = 0; c < NC; c++) {
                Scored e;
                e.err = r1[i + c];
                e.cand = cand;
                e.p = p;
                e.c = c;
                push_top(list, P.cand, e);
            }
        }
    }
}

/// One thread per (slot, shortlist entry): rung 2, the entry's full composite
/// (`refine::composite`) scored by `refine::ranking_error`.
kernel void rung2(device const float4* cells [[buffer(0)]],
                  device const float4* alpha [[buffer(1)]],
                  device const Slot* slots [[buffer(3)]],
                  device const float4* prefixes [[buffer(6)]],
                  device const float4* suffixes [[buffer(7)]],
                  device const float4* beam [[buffer(8)]],
                  device const Scored* shortlist [[buffer(11)]],
                  device float* r2 [[buffer(13)]],
                  constant Params& P [[buffer(15)]],
                  uint id [[thread_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint size [[threads_per_threadgroup]]) {
    threadgroup float lin[256];
    load_lin(lin, tid, size);
    if (id >= P.n_slots * P.n_exact) return;
    uint slot = id / P.n_exact;
    Scored e = shortlist[id];
    if (!isfinite(e.err)) return;
    Slot s = slots[slot];

    device const float4* sfx = prev_suffix(suffixes, beam, P, slot, e.cand);
    device const float4* alp_p = alpha + e.p * PL;
    uint pre0 = prefix_at(P, slot, P.layer, 0);
    float4 c0 = float4(COLORS[e.c][0]);
    float4 c1 = float4(COLORS[e.c][1]);
    float4 c2 = float4(COLORS[e.c][2]);

    float4 acc = float4(0.0f);
    for (uint j = s.start / 4; j < PL; j++) {
        float4 alp = alp_p[j];
        float4 om = 1.0f - alp;
        float4 mul = sfx[j];
        float4 r = fma(fma(prefixes[pre0 + j], om, c0 * alp), mul, sfx[PL + j]);
        float4 g = fma(fma(prefixes[pre0 + PL + j], om, c1 * alp), mul, sfx[2 * PL + j]);
        float4 b = fma(fma(prefixes[pre0 + 2 * PL + j], om, c2 * alp), mul, sfx[3 * PL + j]);
        acc = acc + ranking_term(lin, cells, s.cell, j, P.lambda, r, g, b);
    }
    r2[id] = hsum(acc);
}

/// One thread per slot: the beam update of a window step. Ranks the exact
/// scores in shortlist order (or takes rung 1's list as is), prunes by
/// `error_threshold`, and writes the surviving candidates' layer chains into
/// generation `P.generation ^ 1`.
kernel void beam_step(device Sol* sols [[buffer(4)]],
                      device uint2* chains [[buffer(9)]],
                      device Scored* best [[buffer(10)]],
                      device const Scored* shortlist [[buffer(11)]],
                      device const float* r2 [[buffer(13)]],
                      constant Params& P [[buffer(15)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    device Scored* list = best + id * P.cand;
    if (P.n_exact > 0) {
        for (uint i = 0; i < P.cand; i++) list[i] = no_cand();
        for (uint j = 0; j < P.n_exact; j++) {
            Scored e = shortlist[id * P.n_exact + j];
            if (!isfinite(e.err)) break;
            e.err = r2[id * P.n_exact + j];
            push_top(list, P.cand, e);
        }
    }

    uint finite = 0;
    while (finite < P.cand && isfinite(list[finite].err)) finite++;
    finite = max(finite, 1u);
    uint n;
    if (P.threshold > 0.0f) {
        float bound = list[0].err / P.threshold;
        n = 0;
        while (n < P.cand && list[n].err <= bound) n++;
    } else {
        n = P.cand;
    }
    n = clamp(n, 1u, finite);

    for (uint ci = 0; ci < n; ci++) {
        Scored e = list[ci];
        uint from = chain_at(P, id, P.generation, e.cand);
        uint to = chain_at(P, id, P.generation ^ 1u, ci);
        for (uint i = 0; i < P.step; i++) chains[to + i] = chains[from + i];
        chains[to + P.step] = uint2(e.p, e.c);
        if (P.layer == 0) break;
    }
    sols[id].n_cand = n;
}

/// One thread per (slot, beam candidate, plane position): the suffix each
/// surviving candidate carries into the next step (`refine::build_suffix`),
/// into generation `P.generation ^ 1`.
kernel void beam_suffix(device const float4* alpha [[buffer(1)]],
                        device const Slot* slots [[buffer(3)]],
                        device const Sol* sols [[buffer(4)]],
                        device const float4* suffixes [[buffer(7)]],
                        device float4* beam [[buffer(8)]],
                        device const Scored* best [[buffer(10)]],
                        constant Params& P [[buffer(15)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots * P.cand * PL) return;
    uint j = id % PL;
    uint rest = id / PL;
    uint cand = rest % P.cand;
    uint slot = rest / P.cand;
    if (cand >= sols[slot].n_cand || j < slots[slot].start / 4) return;
    Scored e = best[slot * P.cand + cand];
    device const float4* src = prev_suffix(suffixes, beam, P, slot, e.cand);
    uint dst = beam_at(P, slot, P.generation ^ 1u, cand);
    float4 alp = alpha[e.p * PL + j];
    float4 smul = src[j];
    for (uint ch = 0; ch < 3; ch++) {
        beam[dst + (1 + ch) * PL + j] = fma(float4(COLORS[e.c][ch]) * alp, smul, src[(1 + ch) * PL + j]);
    }
    beam[dst + j] = (1.0f - alp) * smul;
}

/// One thread per slot: write the window's winning chain (candidate 0 of
/// generation `P.generation`, `P.step` layers, deepest first) into the layer list.
kernel void commit_window(device uint2* layers [[buffer(5)]],
                          device const uint2* chains [[buffer(9)]],
                          constant Params& P [[buffer(15)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    uint from = chain_at(P, id, P.generation, 0);
    for (uint i = 0; i < P.step; i++) {
        layers[id * P.nl + P.start - i] = chains[from + i];
    }
}

/// One thread per (slot, plane position): `suffixes[P.start]` from
/// `suffixes[P.start + 1]` and the window's now-fixed top layer.
kernel void chain_suffix(device const float4* alpha [[buffer(1)]],
                         device const Slot* slots [[buffer(3)]],
                         device const uint2* layers [[buffer(5)]],
                         device float4* suffixes [[buffer(7)]],
                         constant Params& P [[buffer(15)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots * PL) return;
    uint slot = id / PL;
    uint j = id % PL;
    if (j < slots[slot].start / 4) return;
    uint2 pc = layers[slot * P.nl + P.start];
    uint src = suffix_at(P, slot, P.start + 1);
    uint dst = suffix_at(P, slot, P.start);
    float4 alp = alpha[pc.x * PL + j];
    float4 smul = suffixes[src + j];
    for (uint ch = 0; ch < 3; ch++) {
        suffixes[dst + (1 + ch) * PL + j] =
            fma(float4(COLORS[pc.y][ch]) * alp, smul, suffixes[src + (1 + ch) * PL + j]);
    }
    suffixes[dst + j] = (1.0f - alp) * smul;
}

// ------------------------------------------------------------- end of a pass

/// One thread per (slot, dye): the exact score of base dye `c` under the whole
/// layer stack, `c·mul + add` with `suffixes[0]` (`refine::reopt_base`).
kernel void reopt_exact(device const float4* cells [[buffer(0)]],
                        device const Slot* slots [[buffer(3)]],
                        device const float4* suffixes [[buffer(7)]],
                        device float* reopt [[buffer(14)]],
                        constant Params& P [[buffer(15)]],
                        uint id [[thread_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint size [[threads_per_threadgroup]]) {
    threadgroup float lin[256];
    load_lin(lin, tid, size);
    if (id >= P.n_slots * NC) return;
    uint slot = id / NC;
    uint c = id % NC;
    Slot s = slots[slot];
    uint at = suffix_at(P, slot, 0);
    float4 c0 = float4(COLORS[c][0]);
    float4 c1 = float4(COLORS[c][1]);
    float4 c2 = float4(COLORS[c][2]);
    float4 acc = float4(0.0f);
    for (uint j = s.start / 4; j < PL; j++) {
        float4 mul = suffixes[at + j];
        float4 r = fma(c0, mul, suffixes[at + PL + j]);
        float4 g = fma(c1, mul, suffixes[at + 2 * PL + j]);
        float4 b = fma(c2, mul, suffixes[at + 3 * PL + j]);
        acc = acc + ranking_term(lin, cells, s.cell, j, P.lambda, r, g, b);
    }
    reopt[id] = hsum(acc);
}

/// One thread per slot: the first minimum of `reopt_exact`'s 16 scores.
kernel void reopt_pick(device Sol* sols [[buffer(4)]],
                       device const float* reopt [[buffer(14)]],
                       constant Params& P [[buffer(15)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    uint best = sols[id].base;
    float min_err = INFINITY;
    for (uint c = 0; c < NC; c++) {
        float err = reopt[id * NC + c];
        if (err < min_err) {
            min_err = err;
            best = c;
        }
    }
    sols[id].base = best;
}

/// One thread per slot: the closed-form base re-optimisation used with the
/// exact rung off.
kernel void reopt_closed(device const float4* cells [[buffer(0)]],
                         device const Slot* slots [[buffer(3)]],
                         device Sol* sols [[buffer(4)]],
                         device const float4* suffixes [[buffer(7)]],
                         constant Params& P [[buffer(15)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= P.n_slots) return;
    Slot s = slots[id];
    uint at = suffix_at(P, id, 0);
    float4 acc = float4(0.0f);
    for (uint j = s.start / 4; j < PL; j++) {
        float4 m = suffixes[at + j];
        acc = fma(m, m, acc);
    }
    float mul2 = hsum(acc);
    float d2[3];
    float d_2m[3];
    for (uint ch = 0; ch < 3; ch++) {
        float4 acc2 = float4(0.0f);
        float4 acc_m = float4(0.0f);
        for (uint j = s.start / 4; j < PL; j++) {
            float4 m = suffixes[at + j];
            float4 d = suffixes[at + (1 + ch) * PL + j] - cells[cell_at(s.cell, ch) + j];
            acc2 = fma(d, d, acc2);
            acc_m = fma(d, m, acc_m);
        }
        d2[ch] = hsum(acc2);
        d_2m[ch] = 2.0f * hsum(acc_m);
    }
    uint best = sols[id].base;
    float min_err = INFINITY;
    for (uint c = 0; c < NC; c++) {
        float err = sweep(float3(d2[0], d2[1], d2[2]), float3(d_2m[0], d_2m[1], d_2m[2]), mul2, c);
        if (err < min_err) {
            min_err = err;
            best = c;
        }
    }
    sols[id].base = best;
}

/// One thread per slot: the ranking error and the pure-target error of the
/// final composite `prefixes[nl]`, in one pass (both sums are the CPU's,
/// accumulated separately).
kernel void finish(device const float4* cells [[buffer(0)]],
                   device const Slot* slots [[buffer(3)]],
                   device Sol* sols [[buffer(4)]],
                   device const float4* prefixes [[buffer(6)]],
                   constant Params& P [[buffer(15)]],
                   uint id [[thread_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint size [[threads_per_threadgroup]]) {
    threadgroup float lin[256];
    load_lin(lin, tid, size);
    if (id >= P.n_slots) return;
    Slot s = slots[id];
    uint at = prefix_at(P, id, P.nl, 0);
    float4 rank = float4(0.0f);
    float4 pure = float4(0.0f);
    for (uint j = s.start / 4; j < PL; j++) {
        Lab lab = oklab(lin, prefixes[at + j], prefixes[at + PL + j], prefixes[at + 2 * PL + j]);
        float4 target = delta_e(lab, cells[cell_at(s.cell, 3) + j], cells[cell_at(s.cell, 4) + j],
                                cells[cell_at(s.cell, 5) + j]);
        pure = pure + target;
        if (P.lambda != 0.0f) {
            float4 feature = delta_e(lab, cells[cell_at(s.cell, 6) + j],
                                     cells[cell_at(s.cell, 7) + j], cells[cell_at(s.cell, 8) + j]);
            rank = rank + fma(feature, float4(P.lambda), target);
        }
    }
    sols[id].lab_err = hsum(pure);
    sols[id].rank_err = P.lambda != 0.0f ? hsum(rank) : hsum(pure);
}
