//! The device layer, on wgpu: Vulkan, DX12 or Metal through one API. Device,
//! compiled kernels, buffers with a CPU-side copy, and the encoder the
//! schedule records dispatches into. Every `unsafe` block of the GPU backend
//! is in this file.
//!
//! wgpu buffers are not shared with the CPU. A buffer the CPU touches keeps a
//! copy in host memory: writes go to the copy and are uploaded when the next
//! encoder is created, and [`Buffer::fetch`] reads the GPU's contents
//! back into it. Buffers the CPU never touches have no copy.
//!
//! On Metal the kernels are translated to MSL here and handed to wgpu as raw
//! MSL, so the translation can be controlled: no bounds checks, no loop
//! guards, and without the `max_total_threads_per_threadgroup` attribute naga
//! puts on every compute entry point. Every other backend takes the WGSL as
//! it is.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::GpuError;
use super::kernel::{Bind, Kernel, Limits, Plain};
use super::source;
use super::state::Params;

/// Threads per workgroup, as `kernels.wgsl` declares on every entry point.
const GROUP: usize = 64;

/// The binding the two hot kernels read a candidate's suffix through: the
/// suffix chain at window step 0, the beam afterwards.
const SRC: u32 = 15;

/// The bindings each kernel uses, as `kernels.wgsl` declares them. Bindings 0
/// to 3 and [`SRC`] are read-only, the rest writable.
fn bindings(kernel: Kernel) -> &'static [u32] {
    const CELLS: u32 = Bind::Cells as u32;
    const ALPHA: u32 = Bind::Alpha as u32;
    const ALPHA2: u32 = Bind::Alpha2 as u32;
    const SLOTS: u32 = Bind::Slots as u32;
    const SOLS: u32 = Bind::Sols as u32;
    const LAYERS: u32 = Bind::Layers as u32;
    const PREFIXES: u32 = Bind::Prefixes as u32;
    const SUFFIXES: u32 = Bind::Suffixes as u32;
    const BEAM: u32 = Bind::Beam as u32;
    const CHAINS: u32 = Bind::Chains as u32;
    const BEST: u32 = Bind::Best as u32;
    const SHORTLIST: u32 = Bind::Shortlist as u32;
    const R1: u32 = Bind::Rung1 as u32;
    const R2: u32 = Bind::Rung2 as u32;
    const REOPT: u32 = Bind::Reopt as u32;
    match kernel {
        Kernel::GreedyBase => &[CELLS, SLOTS, SOLS],
        Kernel::GreedyMoments => &[CELLS, ALPHA, ALPHA2, SLOTS, PREFIXES, R1],
        Kernel::GreedyPick => &[SOLS, LAYERS, R1],
        Kernel::PrefixRange => &[ALPHA, SLOTS, SOLS, LAYERS, PREFIXES],
        Kernel::IdentitySuffix => &[SLOTS, SUFFIXES],
        Kernel::Rung1 => &[CELLS, ALPHA, SLOTS, SOLS, PREFIXES, R1, SRC],
        Kernel::SelectTop => &[SOLS, BEST, R1],
        Kernel::ShortlistTop => &[SOLS, SHORTLIST, R1],
        Kernel::Rung2 => &[CELLS, ALPHA, SLOTS, PREFIXES, SHORTLIST, R2, SRC],
        Kernel::BeamStep => &[SOLS, CHAINS, BEST, SHORTLIST, R2],
        Kernel::BeamSuffix => &[ALPHA, SLOTS, SOLS, SUFFIXES, BEAM, BEST],
        Kernel::CommitWindow => &[LAYERS, CHAINS],
        Kernel::ChainSuffix => &[ALPHA, SLOTS, LAYERS, SUFFIXES],
        Kernel::ReoptExact => &[CELLS, SLOTS, SUFFIXES, REOPT],
        Kernel::ReoptPick => &[SOLS, REOPT],
        Kernel::ReoptClosed => &[CELLS, SLOTS, SOLS, SUFFIXES],
        Kernel::Finish => &[CELLS, SLOTS, SOLS, PREFIXES],
    }
}

fn read_only(binding: u32) -> bool {
    binding <= Bind::Slots as u32 || binding == SRC
}

/// Wait for everything submitted so far.
fn wait(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU device was lost");
}

/// Run `f` and return the validation error it raised on `device`, if any.
fn validated<R>(device: &wgpu::Device, f: impl FnOnce() -> R) -> Result<R, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = f();
    match pollster::block_on(scope.pop()) {
        Some(e) => Err(e.to_string()),
        None => Ok(out),
    }
}

/// One kernel's pipeline and the layout its bind groups are made from.
struct Pipeline {
    raw: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The device, its queue and every compiled kernel.
pub(super) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<Pipeline>,
    name: String,
    limits: Limits,
    /// Most workgroups one dimension of a dispatch may have.
    row_groups: usize,
    /// Bytes of GPU buffers alive, shared with every [`Buffer`].
    allocated: Arc<AtomicUsize>,
}

/// A GPU buffer and, once the CPU has touched it, its host copy.
pub(super) struct Buffer {
    raw: wgpu::Buffer,
    /// Whole `u32`s, so the copy is aligned for every [`Plain`] type.
    copy: Vec<u32>,
    /// The copy has CPU writes the GPU has not seen.
    dirty: Cell<bool>,
    bytes: usize,
    allocated: Arc<AtomicUsize>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.raw.destroy();
        self.allocated.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

impl Buffer {
    fn ensure_copy(&mut self) {
        if self.copy.is_empty() {
            self.copy = vec![0; self.bytes / 4];
        }
    }

    fn copy_bytes(&self) -> &[u8] {
        // SAFETY: any `u32` slice is valid as bytes of four times the length.
        unsafe { std::slice::from_raw_parts(self.copy.as_ptr().cast::<u8>(), self.copy.len() * 4) }
    }
}

impl Buffer {
    /// The buffer's contents as `T`s, for the CPU to write.
    pub(super) fn slice_mut<T: Plain>(&mut self) -> &mut [T] {
        self.ensure_copy();
        self.dirty.set(true);
        assert!(align_of::<T>() <= align_of::<u32>());
        let len = self.copy.len() * 4 / size_of::<T>();
        // SAFETY: the copy is `len * size_of::<T>()` initialised bytes at most,
        // aligned for `T` (checked above), exclusively borrowed through
        // `&mut self`, and `T: Plain` accepts every bit pattern.
        unsafe { std::slice::from_raw_parts_mut(self.copy.as_mut_ptr().cast::<T>(), len) }
    }

    /// The buffer's contents as `T`s, as of the last [`Buffer::fetch`] or CPU
    /// write. Empty until the CPU has written the buffer or fetched it.
    pub(super) fn slice<T: Plain>(&self) -> &[T] {
        assert!(align_of::<T>() <= align_of::<u32>());
        let len = self.copy.len() * 4 / size_of::<T>();
        // SAFETY: as `slice_mut`, with a shared borrow.
        unsafe { std::slice::from_raw_parts(self.copy.as_ptr().cast::<T>(), len) }
    }

    /// Read what the kernels wrote back, for [`Buffer::slice`].
    pub(super) fn fetch(&mut self, gpu: &Gpu) {
        self.ensure_copy();
        let size = self.bytes as u64;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.raw, 0, &staging, 0, size);
        gpu.queue.submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| {
            r.expect("could not map a GPU buffer")
        });
        wait(&gpu.device);
        {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .expect("the buffer was mapped above");
            // SAFETY: a `u32` slice viewed as its bytes; every byte pattern is
            // a valid `u32`.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(self.copy.as_mut_ptr().cast::<u8>(), self.bytes)
            };
            dst.copy_from_slice(&view);
        }
        staging.destroy();
        self.dirty.set(false);
    }
}

/// How suitable an adapter is: real GPUs only, discrete first.
fn rank(info: &wgpu::AdapterInfo) -> Option<u32> {
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => Some(0),
        wgpu::DeviceType::IntegratedGpu => Some(1),
        wgpu::DeviceType::VirtualGpu => Some(2),
        wgpu::DeviceType::Other => Some(3),
        // A software rasteriser (llvmpipe, WARP) is slower than the CPU solver.
        wgpu::DeviceType::Cpu => None,
    }
}

/// The MSL naga writes for `kernel`, with the buffer slots wgpu's Metal
/// backend assigns to this kernel's pipeline layout: the parameter block
/// first, then the bindings in order. Returns the source and the entry
/// point's name in it.
#[cfg(target_os = "macos")]
fn msl(
    module: &wgpu::naga::Module,
    info: &wgpu::naga::valid::ModuleInfo,
    kernel: Kernel,
) -> Result<(String, String), GpuError> {
    use wgpu::naga::back::msl;
    use wgpu::naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};

    let mut resources = msl::BindingMap::default();
    for (i, &binding) in bindings(kernel).iter().enumerate() {
        resources.insert(
            wgpu::naga::ResourceBinding { group: 0, binding },
            msl::BindTarget {
                buffer: Some(1 + i as msl::Slot),
                mutable: !read_only(binding),
                ..Default::default()
            },
        );
    }
    let unchecked = BoundsCheckPolicy::Unchecked;
    let options = msl::Options {
        lang_version: (2, 4),
        per_entry_point_map: msl::EntryPointResourceMap::from([(
            kernel.name().to_owned(),
            msl::EntryPointResources {
                resources,
                immediates_buffer: Some(0),
                sizes_buffer: None,
            },
        )]),
        bounds_check_policies: BoundsCheckPolicies {
            index: unchecked,
            buffer: unchecked,
            image_load: unchecked,
            binding_array: unchecked,
        },
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: false,
        ..Default::default()
    };
    let pipeline = msl::PipelineOptions {
        entry_point: Some((wgpu::naga::ShaderStage::Compute, kernel.name().to_owned())),
        ..Default::default()
    };
    let (text, names) = msl::write_string(module, info, &options, &pipeline)
        .map_err(|e| GpuError::Compile(format!("{}: {e}", kernel.name())))?;
    let entry = names
        .entry_point_names
        .into_iter()
        .next()
        .ok_or_else(|| GpuError::Compile(format!("{}: no entry point", kernel.name())))?
        .map_err(|e| GpuError::Compile(format!("{}: {e}", kernel.name())))?;
    // The attribute caps the threadgroup at the WGSL workgroup size; Metal
    // needs no cap to run these kernels.
    let attribute = format!("[[max_total_threads_per_threadgroup({GROUP})]] ");
    Ok((text.replace(&attribute, ""), entry))
}

impl Gpu {
    /// One shader module and entry-point name per kernel.
    fn modules(
        device: &wgpu::Device,
        backend: wgpu::Backend,
        raw_msl: bool,
    ) -> Result<Vec<(wgpu::ShaderModule, String)>, GpuError> {
        let wgsl = source::wgsl(GROUP);
        #[cfg(target_os = "macos")]
        if backend == wgpu::Backend::Metal && raw_msl {
            let module = wgpu::naga::front::wgsl::parse_str(&wgsl)
                .map_err(|e| GpuError::Compile(e.emit_to_string(&wgsl)))?;
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|e| GpuError::Compile(e.emit_to_string(&wgsl)))?;
            return Kernel::ALL
                .iter()
                .map(|&kernel| {
                    let (text, entry) = msl(&module, &info, kernel)?;
                    // SAFETY: the MSL is naga's translation of validated
                    // WGSL, its buffer slots are the ones wgpu assigns to the
                    // layout the pipeline is created with, and the kernels
                    // index every buffer within the sizes the host allocates.
                    let module = unsafe {
                        device.create_shader_module_passthrough(
                            wgpu::ShaderModuleDescriptorPassthrough {
                                label: Some(kernel.name()),
                                entry_points: vec![wgpu::PassthroughShaderEntryPoint {
                                    name: std::borrow::Cow::Owned(entry.clone()),
                                    workgroup_size: (GROUP as u32, 1, 1),
                                }]
                                .into(),
                                msl: Some(std::borrow::Cow::Owned(text)),
                                ..Default::default()
                            },
                        )
                    };
                    Ok((module, entry))
                })
                .collect();
        }
        let _ = (backend, raw_msl);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernels.wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        Ok(Kernel::ALL
            .iter()
            .map(|k| (module.clone(), k.name().to_owned()))
            .collect())
    }

    /// Open the best real GPU and compile every kernel.
    pub(super) fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY));
        let needed = wgpu::Features::IMMEDIATES;
        let adapter = adapters
            .into_iter()
            .filter(|a| a.features().contains(needed))
            .filter(|a| a.limits().max_immediate_size as usize >= size_of::<Params>())
            .filter_map(|a| rank(&a.get_info()).map(|r| (r, a)))
            .min_by_key(|(r, _)| *r)
            .map(|(_, a)| a)
            .ok_or(GpuError::NoDevice)?;
        let info = adapter.get_info();
        let raw_msl = info.backend == wgpu::Backend::Metal
            && adapter
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS);
        let mut features = needed;
        if raw_msl {
            features |= wgpu::Features::PASSTHROUGH_SHADERS;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bannerify"),
            required_features: features,
            // Everything the adapter offers: the solver's buffers are far
            // larger than WebGPU's default limits.
            required_limits: adapter.limits(),
            experimental_features: if raw_msl {
                // SAFETY: the one experimental feature used is raw MSL, whose
                // requirements `Gpu::modules` documents.
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            ..Default::default()
        }))
        .map_err(|e| GpuError::Setup(e.to_string()))?;

        let pipelines = validated(&device, || -> Result<Vec<Pipeline>, GpuError> {
            let modules = Self::modules(&device, info.backend, raw_msl)?;
            Ok(Kernel::ALL
                .iter()
                .zip(&modules)
                .map(|(&kernel, (module, entry))| {
                    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings(kernel)
                        .iter()
                        .map(|&binding| wgpu::BindGroupLayoutEntry {
                            binding,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: read_only(binding),
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        })
                        .collect();
                    let layout =
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some(kernel.name()),
                            entries: &entries,
                        });
                    let pipeline_layout =
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(kernel.name()),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: size_of::<Params>() as u32,
                        });
                    let raw = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(kernel.name()),
                        layout: Some(&pipeline_layout),
                        module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    });
                    Pipeline { raw, layout }
                })
                .collect())
        })
        .map_err(GpuError::Compile)??;

        let limits = device.limits();
        Ok(Self {
            name: format!("{} ({:?})", info.name, info.backend),
            limits: Limits {
                max_buffer_bytes: limits
                    .max_buffer_size
                    .min(limits.max_storage_buffer_binding_size)
                    .min(usize::MAX as u64) as usize,
                // Two dimensions of workgroups, and thread indices are `u32`.
                max_threads: (limits.max_compute_workgroups_per_dimension as usize)
                    .saturating_pow(2)
                    .saturating_mul(GROUP)
                    .min(u32::MAX as usize),
            },
            row_groups: limits.max_compute_workgroups_per_dimension as usize,
            device,
            queue,
            pipelines,
            allocated: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl Gpu {
    /// The device's name, for reports.
    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn limits(&self) -> Limits {
        self.limits
    }

    /// A zeroed buffer of at least `bytes` bytes.
    pub(super) fn buffer(&self, bytes: usize) -> Result<Buffer, GpuError> {
        let bytes = bytes.max(16).next_multiple_of(4);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if pollster::block_on(scope.pop()).is_some() {
            return Err(GpuError::OutOfMemory(bytes));
        }
        self.allocated.fetch_add(bytes, Ordering::Relaxed);
        Ok(Buffer {
            raw,
            copy: Vec::new(),
            dirty: Cell::new(false),
            bytes,
            allocated: Arc::clone(&self.allocated),
        })
    }

    /// Bytes of device memory the solver's buffers currently take.
    pub(super) fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Start recording dispatches that see `bound` at their binding indices,
    /// with whatever the CPU wrote to those buffers uploaded first.
    ///
    /// With `profile` set every dispatch is submitted and waited for on its
    /// own and its wall time is added to `profile`'s entry for the kernel, so
    /// it is for measurement only.
    pub(super) fn encoder<'a>(
        &'a self,
        bound: &'a [(Bind, &'a Buffer)],
        profile: Option<&'a mut BTreeMap<Kernel, Duration>>,
    ) -> Encoder<'a> {
        for (_, buffer) in bound {
            if buffer.dirty.replace(false) {
                self.queue.write_buffer(&buffer.raw, 0, buffer.copy_bytes());
            }
        }
        let find = |bind: Bind| -> &wgpu::Buffer {
            let (_, buffer) = bound
                .iter()
                .find(|(b, _)| *b as u32 == bind as u32)
                .expect("every buffer of the set is bound");
            &buffer.raw
        };
        let group = |kernel: Kernel, src: Bind| -> wgpu::BindGroup {
            let entries: Vec<wgpu::BindGroupEntry> = bindings(kernel)
                .iter()
                .map(|&binding| {
                    let raw = if binding == SRC {
                        find(src)
                    } else {
                        let (_, buffer) = bound
                            .iter()
                            .find(|(b, _)| *b as u32 == binding)
                            .expect("every buffer of the set is bound");
                        &buffer.raw
                    };
                    wgpu::BindGroupEntry {
                        binding,
                        resource: raw.as_entire_binding(),
                    }
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines[kernel as usize].layout,
                entries: &entries,
            })
        };
        Encoder {
            gpu: self,
            groups: Kernel::ALL
                .iter()
                .map(|&k| group(k, Bind::Suffixes))
                .collect(),
            from_beam: [
                group(Kernel::Rung1, Bind::Beam),
                group(Kernel::Rung2, Bind::Beam),
            ],
            profile,
            pending: Vec::new(),
        }
    }
}

/// One recorded dispatch: `rows` rows of `params.row / GROUP` workgroups.
struct Pending {
    kernel: Kernel,
    params: Params,
    rows: u32,
}

/// Dispatches recorded against one set of bind groups, submitted as a single
/// compute pass by [`Encoder::finish`]. They run in recording order, each
/// seeing every write of the ones before it.
pub(super) struct Encoder<'a> {
    gpu: &'a Gpu,
    /// Per kernel, with [`SRC`] bound to the suffix chain.
    groups: Vec<wgpu::BindGroup>,
    /// Rung 1 and rung 2 with [`SRC`] bound to the beam.
    from_beam: [wgpu::BindGroup; 2],
    profile: Option<&'a mut BTreeMap<Kernel, Duration>>,
    pending: Vec<Pending>,
}

impl Encoder<'_> {
    /// Submit everything pending as one compute pass and wait for it.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in self.pending.drain(..) {
                // The candidate's suffix is in the beam from step 1 on.
                let group = match p.kernel {
                    Kernel::Rung1 if p.params.step > 0 => &self.from_beam[0],
                    Kernel::Rung2 if p.params.step > 0 => &self.from_beam[1],
                    k => &self.groups[k as usize],
                };
                pass.set_pipeline(&self.gpu.pipelines[p.kernel as usize].raw);
                pass.set_bind_group(0, group, &[]);
                // SAFETY: `Params` is `Plain`: `repr(C)`, no padding.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        (&raw const p.params).cast::<u8>(),
                        size_of::<Params>(),
                    )
                };
                pass.set_immediates(0, bytes);
                pass.dispatch_workgroups(p.params.row / GROUP as u32, p.rows, 1);
            }
        }
        self.gpu.queue.submit([enc.finish()]);
        wait(&self.gpu.device);
    }
}

impl Encoder<'_> {
    /// Record `kernel` over `threads` threads with `params`.
    pub(super) fn dispatch(&mut self, kernel: Kernel, params: &Params, threads: usize) {
        if threads == 0 {
            return;
        }
        // One dimension holds `row_groups` workgroups at most; the rest of the
        // threads go to further rows, and the kernels find their index from
        // the row width.
        let groups = threads.div_ceil(GROUP);
        let per_row = groups.min(self.gpu.row_groups);
        self.pending.push(Pending {
            kernel,
            params: Params {
                row: (per_row * GROUP) as u32,
                ..*params
            },
            rows: groups.div_ceil(per_row) as u32,
        });
        if self.profile.is_some() {
            let t = Instant::now();
            self.flush();
            let profile = self.profile.as_mut().expect("checked above");
            *profile.entry(kernel).or_default() += t.elapsed();
        }
    }

    /// Run everything recorded and wait for it.
    pub(super) fn finish(mut self) {
        self.flush();
    }
}
