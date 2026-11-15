//! The Metal layer: device, compiled kernels, shared buffers and the encoder
//! the schedule records dispatches into. Every `unsafe` block of the GPU
//! backend is in this file.
//!
//! All buffers are shared-storage, so the CPU writes inputs and reads results
//! in place. Every command buffer is waited for before control returns to the
//! caller ([`Encoder::finish`]), so no GPU work is in flight whenever Rust code
//! holds a slice of a buffer; that invariant is what the `unsafe` views below
//! rest on.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Duration;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions,
    MTLMathMode, MTLResourceOptions, MTLSize,
};

use super::GpuError;

// `MTLCreateSystemDefaultDevice` returns nil unless CoreGraphics is linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Every kernel in `kernels.metal`, in the order the pipelines are stored.
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
    const ALL: [Kernel; 17] = [
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

    /// The MSL function name.
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

/// Buffer binding indices, shared by every kernel (`kernels.metal` declares
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

/// Binding index of the per-dispatch parameter block.
const PARAMS_INDEX: usize = 15;

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

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A shared-storage buffer.
pub(super) struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
}

impl Buffer {
    /// The buffer's contents as `T`s, for the CPU to read or write.
    ///
    /// `&mut self` makes the view exclusive on the Rust side; the GPU side is
    /// covered by the module invariant (nothing in flight outside
    /// [`Encoder::finish`]).
    pub(super) fn slice_mut<T: Plain>(&mut self) -> &mut [T] {
        let len = self.bytes / size_of::<T>();
        // SAFETY: `contents` points at `bytes` bytes of shared memory owned by
        // `raw`, page aligned (so aligned for any `T` here), alive as long as
        // `self`. `T: Plain` accepts every bit pattern. No GPU work is in
        // flight (module invariant), and `&mut self` excludes other CPU views.
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<T>(), len) }
    }

    /// Read-only [`Buffer::slice_mut`].
    pub(super) fn slice<T: Plain>(&self) -> &[T] {
        let len = self.bytes / size_of::<T>();
        // SAFETY: as `slice_mut`; the shared borrow excludes CPU writers.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), len) }
    }
}

/// The device, its queue and every compiled kernel.
pub(super) struct Gpu {
    device: Device,
    queue: Queue,
    pipelines: Vec<Pipeline>,
    /// Device name, for reports.
    pub(super) name: String,
}

impl Gpu {
    /// Open the default device and compile `source`.
    pub(super) fn new(source: &str) -> Result<Self, GpuError> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| GpuError::Setup("could not create a command queue".into()))?;

            let options = MTLCompileOptions::new();
            // No reassociation, no approximate division or square root: the
            // kernels are written to evaluate exactly what the CPU does.
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                .map_err(|e| GpuError::Compile(e.localizedDescription().to_string()))?;

            let mut pipelines = Vec::with_capacity(Kernel::ALL.len());
            for kernel in Kernel::ALL {
                let function = library
                    .newFunctionWithName(&NSString::from_str(kernel.name()))
                    .ok_or_else(|| {
                        GpuError::Compile(format!("missing kernel {}", kernel.name()))
                    })?;
                let pipeline = device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| {
                        GpuError::Compile(format!(
                            "{}: {}",
                            kernel.name(),
                            e.localizedDescription()
                        ))
                    })?;
                pipelines.push(pipeline);
            }

            let name = device.name().to_string();
            Ok(Self {
                device,
                queue,
                pipelines,
                name,
            })
        })
    }

    /// A zeroed shared buffer of at least `bytes` bytes.
    pub(super) fn buffer(&self, bytes: usize) -> Result<Buffer, GpuError> {
        // Metal refuses zero-length buffers.
        let bytes = bytes.max(16);
        let raw = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(GpuError::OutOfMemory(bytes))?;
        Ok(Buffer { raw, bytes })
    }

    /// Bytes the device currently has allocated, for memory reports.
    pub(super) fn allocated(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    /// Start recording a command buffer with `bound` bound at their indices.
    ///
    /// With `profile` set every dispatch is committed and waited for on its
    /// own, and its GPU time is added to `profile`'s entry for the kernel. That
    /// serialises the GPU, so it is for measurement only.
    pub(super) fn encoder<'a>(
        &'a self,
        bound: &'a [(Bind, &'a Buffer)],
        profile: Option<&'a mut BTreeMap<Kernel, Duration>>,
    ) -> Encoder<'a> {
        let mut encoder = Encoder {
            gpu: self,
            bound,
            profile,
            open: None,
        };
        encoder.open();
        encoder
    }
}

type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ComputeEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// One command buffer being recorded. Dispatches run in recording order, each
/// seeing every write of the ones before it (a serial compute encoder).
pub(super) struct Encoder<'a> {
    gpu: &'a Gpu,
    bound: &'a [(Bind, &'a Buffer)],
    profile: Option<&'a mut BTreeMap<Kernel, Duration>>,
    open: Option<(CommandBuffer, ComputeEncoder)>,
}

impl Encoder<'_> {
    fn open(&mut self) {
        let cb = self
            .gpu
            .queue
            .commandBuffer()
            .expect("Metal could not create a command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal could not create a compute encoder");
        for (bind, buffer) in self.bound {
            // SAFETY: the buffer outlives the encoder (`'a`), and the offset 0
            // is in bounds.
            unsafe { enc.setBuffer_offset_atIndex(Some(&buffer.raw), 0, *bind as usize) };
        }
        self.open = Some((cb, enc));
    }

    /// Record `kernel` over `threads` threads with `params`.
    pub(super) fn dispatch<P: Plain>(&mut self, kernel: Kernel, params: &P, threads: usize) {
        if threads == 0 {
            return;
        }
        let pipeline = &self.gpu.pipelines[kernel as usize];
        let (_, enc) = self.open.as_ref().expect("encoder is open");
        enc.setComputePipelineState(pipeline);
        // SAFETY: `params` is a live `Plain` value of exactly that many bytes;
        // Metal copies them before `setBytes` returns.
        unsafe {
            enc.setBytes_length_atIndex(
                NonNull::from(params).cast::<c_void>(),
                size_of::<P>(),
                PARAMS_INDEX,
            );
        }
        // Four SIMD groups per threadgroup. Measured on the M1 Pro, one to eight
        // make no difference beyond 3%, and 32 is clearly slower.
        let width = pipeline.threadExecutionWidth();
        let group = (pipeline.maxTotalThreadsPerThreadgroup() / width).min(4) * width;
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group.max(1),
                height: 1,
                depth: 1,
            },
        );

        if self.profile.is_some() {
            let gpu_time = self.submit();
            let profile = self.profile.as_mut().expect("checked above");
            *profile.entry(kernel).or_default() += gpu_time;
            self.open();
        }
    }

    /// Commit the open command buffer and wait for it; returns its GPU time.
    fn submit(&mut self) -> Duration {
        let (cb, enc) = self.open.take().expect("encoder is open");
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            let reason = cb
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_default();
            panic!("GPU command buffer failed: {reason}");
        }
        Duration::from_secs_f64((cb.GPUEndTime() - cb.GPUStartTime()).max(0.0))
    }

    /// Run everything recorded and wait for it.
    pub(super) fn finish(mut self) {
        self.submit();
    }
}
