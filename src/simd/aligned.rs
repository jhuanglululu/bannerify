//! Runtime-length 64-byte-aligned `f32` buffer.

use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, handle_alloc_error};

use super::{F32s, LANES};

/// Alignment of every [`AlignedVec`] allocation, in bytes.
const ALIGN: usize = 64;

/// Runtime-length 64-byte-aligned `f32` buffer.
///
/// The length is asserted to be a multiple of 16 at construction, so lane views
/// never need a remainder path. Nothing here touches memory implicitly: only
/// the constructor you name writes to the buffer, and [`AlignedVec::new_uninit`]
/// writes nothing at all.
pub struct AlignedVec {
    /// Dangling (but 64-byte aligned) when `len == 0`.
    ptr: NonNull<f32>,
    len: usize,
}

// SAFETY: `AlignedVec` uniquely owns its allocation and hands out access only
// through `&self`/`&mut self`; `f32` is `Send + Sync`.
unsafe impl Send for AlignedVec {}
// SAFETY: as above.
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    fn layout(len: usize) -> Layout {
        assert!(
            len.is_multiple_of(16),
            "AlignedVec: len must be a multiple of 16, got {len}"
        );
        Layout::from_size_align(len * size_of::<f32>(), ALIGN).expect("AlignedVec: layout overflow")
    }

    /// `len` elements, all zero.
    ///
    /// Uses `alloc_zeroed`, so for large buffers the zeroing is the allocator's
    /// (typically fresh zero pages), not an explicit memset.
    pub fn zeroed(len: usize) -> Self {
        let layout = Self::layout(len);
        if len == 0 {
            return Self::empty();
        }
        // SAFETY: `layout` has non-zero size (checked above) and valid alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        Self {
            ptr: NonNull::new(ptr.cast::<f32>()).unwrap_or_else(|| handle_alloc_error(layout)),
            len,
        }
    }

    /// `len` elements, **contents unspecified** — nothing is written at
    /// allocation time (no memset), which is the point: this is the write-once
    /// output buffer constructor.
    ///
    /// # Safety
    ///
    /// The caller must write every element (e.g. through [`AlignedVec::lanes_mut`]
    /// or [`DerefMut`]) before any read of the buffer, including reads through
    /// [`AlignedVec::lanes`], [`Deref`], or `Debug`. Reading an element that has
    /// not been written is undefined behaviour.
    pub unsafe fn new_uninit(len: usize) -> Self {
        let layout = Self::layout(len);
        if len == 0 {
            return Self::empty();
        }
        // SAFETY: `layout` has non-zero size (checked above) and valid alignment.
        let ptr = unsafe { alloc(layout) };
        Self {
            ptr: NonNull::new(ptr.cast::<f32>()).unwrap_or_else(|| handle_alloc_error(layout)),
            len,
        }
    }

    /// `len` elements, each lane produced by `f` — a safe write-once
    /// constructor: allocates uninitialized and writes every lane exactly once.
    pub fn from_lane_fn(len: usize, mut f: impl FnMut(usize) -> F32s) -> Self {
        // SAFETY: every lane — and therefore every element — is written below
        // before this function returns, so no unwritten element is observable.
        let v = unsafe { Self::new_uninit(len) };
        let base = v.ptr.as_ptr().cast::<F32s>();
        for i in 0..len / LANES {
            // SAFETY: `i < len / LANES`, so `base.add(i)` is in bounds of the
            // allocation and 64-byte-aligned base keeps every lane aligned.
            unsafe { base.add(i).write(f(i)) };
        }
        v
    }

    /// `len` elements, all `x`.
    pub fn splat(len: usize, x: f32) -> Self {
        let v = F32s::splat(x);
        Self::from_lane_fn(len, |_| v)
    }

    fn empty() -> Self {
        Self {
            // A 64-byte-aligned dangling pointer: never dereferenced (`len == 0`)
            // but still non-null and aligned, as `slice::from_raw_parts` requires.
            ptr: NonNull::new(ALIGN as *mut f32).expect("non-null"),
            len: 0,
        }
    }

    /// Number of `f32` elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Overwrite every element with `x`.
    #[inline]
    pub fn fill(&mut self, x: f32) {
        let v = F32s::splat(x);
        for lane in self.lanes_mut() {
            *lane = v;
        }
    }

    /// Lane view; length `len() / LANES`.
    #[inline]
    pub fn lanes(&self) -> &[F32s] {
        // SAFETY: the allocation is 64-byte aligned and `len % 16 == 0`, so it
        // holds exactly `len / LANES` correctly aligned `F32s` (a transparent
        // wrapper over `LANES` f32s with no invalid bit patterns); the borrow
        // ties the view's lifetime to `self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr().cast::<F32s>(), self.len / LANES) }
    }

    /// Mutable lane view; length `len() / LANES`.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [F32s] {
        // SAFETY: as `lanes`, and `&mut self` makes the view exclusive.
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<F32s>(), self.len / LANES)
        }
    }
}

impl Deref for AlignedVec {
    type Target = [f32];
    #[inline]
    fn deref(&self) -> &[f32] {
        // SAFETY: `ptr` points to `len` contiguous, aligned `f32`s owned by
        // `self` (dangling but never dereferenced when `len == 0`).
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as `deref`, and `&mut self` makes the slice exclusive.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout = Layout::from_size_align(self.len * size_of::<f32>(), ALIGN)
            .expect("layout was valid at construction");
        // SAFETY: `ptr` came from `alloc`/`alloc_zeroed` with exactly this
        // layout and has not been freed before (this runs once).
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
    }
}

impl core::fmt::Debug for AlignedVec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
