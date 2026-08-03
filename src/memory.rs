//! Minimal allocation tracking, for `--debug` memory reporting.
//!
//! A pass-through [`GlobalAlloc`] over [`System`] that keeps two atomic
//! counters: bytes currently live and the high-water mark. No dependencies, no
//! allocator replacement — the point is to answer "did this run keep memory
//! bounded", which is the property the pipeline design is built around.
//!
//! Cost is one relaxed add and one `fetch_max` per allocation; it is not a
//! profiler and does not attribute allocations to call sites.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bytes currently allocated.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of [`LIVE`].
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// System allocator plus live/peak byte counters.
pub struct Tracking;

impl Tracking {
    #[inline]
    fn record_alloc(size: usize) {
        let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System` with the caller's own pointer and
// layout, so all of `GlobalAlloc`'s requirements are exactly `System`'s. The
// counter updates touch only atomics and never allocate, so they cannot
// re-enter the allocator.
unsafe impl GlobalAlloc for Tracking {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` comes from the caller and is forwarded unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            Self::record_alloc(layout.size());
        }
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::record_alloc(layout.size());
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` was allocated by this allocator (hence by `System`)
        // with `layout`, as `GlobalAlloc::dealloc` requires.
        unsafe { System.dealloc(ptr, layout) };
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`/`layout`/`new_size` are the caller's and are forwarded
        // unchanged to the same underlying allocator.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            Self::record_alloc(new_size);
        }
        new
    }
}

/// Bytes currently allocated.
pub fn live_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Highest number of bytes allocated at once so far.
pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Format a byte count for logs.
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
