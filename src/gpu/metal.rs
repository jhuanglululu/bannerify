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
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

use super::GpuError;
use super::kernel::{Bind, Device, DeviceBuffer, Encode, Kernel, Limits, Plain};
use super::source;
use super::state::Params;

// `MTLCreateSystemDefaultDevice` returns nil unless CoreGraphics is linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Binding index of the per-dispatch parameter block.
const PARAMS_INDEX: usize = 15;

type MetalDevice = Retained<ProtocolObject<dyn MTLDevice>>;
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
    device: MetalDevice,
    queue: Queue,
    pipelines: Vec<Pipeline>,
    /// Device name, for reports.
    pub(super) name: String,
}

impl Gpu {
    /// Open the default device and compile the kernels.
    pub(super) fn new() -> Result<Self, GpuError> {
        let source = source::msl();
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| GpuError::Setup("could not create a command queue".into()))?;

            let options = MTLCompileOptions::new();
            // Fast math: the compiler may fuse, reorder and approximate, so
            // results agree with the CPU solver closely, not bit for bit.
            options.setMathMode(MTLMathMode::Fast);
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
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

    /// Metal sets no limit a memory budget can reach: buffers may be as large
    /// as the device's memory and a dispatch may cover any number of threads.
    pub(super) fn limits(&self) -> Limits {
        Limits {
            max_buffer_bytes: usize::MAX,
            max_threads: usize::MAX,
        }
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
    pub(super) fn dispatch(&mut self, kernel: Kernel, params: &Params, threads: usize) {
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
                size_of::<Params>(),
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

impl Device for Gpu {
    type Buffer = Buffer;
    type Encoder<'a> = Encoder<'a>;

    fn open() -> Result<Self, GpuError> {
        Gpu::new()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn limits(&self) -> Limits {
        Gpu::limits(self)
    }

    fn buffer(&self, bytes: usize) -> Result<Buffer, GpuError> {
        Gpu::buffer(self, bytes)
    }

    fn allocated(&self) -> usize {
        Gpu::allocated(self)
    }

    fn encoder<'a>(
        &'a self,
        bound: &'a [(Bind, &'a Buffer)],
        profile: Option<&'a mut BTreeMap<Kernel, Duration>>,
    ) -> Encoder<'a> {
        Gpu::encoder(self, bound, profile)
    }
}

impl DeviceBuffer<Gpu> for Buffer {
    fn slice_mut<T: Plain>(&mut self) -> &mut [T] {
        Buffer::slice_mut(self)
    }

    fn slice<T: Plain>(&self) -> &[T] {
        Buffer::slice(self)
    }

    /// Shared storage: the kernels' writes are already visible.
    fn fetch(&mut self, _gpu: &Gpu) {}
}

impl Encode for Encoder<'_> {
    fn dispatch(&mut self, kernel: Kernel, params: &Params, threads: usize) {
        Encoder::dispatch(self, kernel, params, threads)
    }

    fn finish(self) {
        Encoder::finish(self)
    }
}
