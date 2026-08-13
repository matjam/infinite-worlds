//! The globe pipeline: static per-chunk geometry, dynamic per-cell data, and
//! the starfield background pass.
//!
//! Geometry is built once from the `Mesh` (a triangle fan per cell over its
//! corners) into one vertex and one index buffer, with a draw range per chunk
//! for culling. Per-cell elevation and colour live in a storage buffer indexed
//! by the cell id carried on every vertex, so a data update never rebuilds
//! geometry.

use anyhow::{anyhow, Result};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use iw_mesh::Mesh;

use crate::buffer::{upload_device_local, Buffer};
use crate::camera::ViewMode;
use crate::cull::{chunk_visible, Frustum};
use crate::gpu::Gpu;

/// Frames the renderer keeps in flight.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Vertical span assumed for culling and Mercator depth normalisation, metres.
const ELEV_NORM_M: f32 = 9_000.0;

/// One corner of one cell. `cells[0]` owns the corner and supplies the flat
/// colour; `cells[1..3]` are the other cells sharing it (repeating `cells[0]`
/// where a corner is shared by fewer than three). Displacement uses the mean
/// of the three so neighbouring cells meet instead of leaving open cliffs.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobeVertex {
    pos: [f32; 3],
    cells: [u32; 3],
}

/// Per-cell dynamic data, std430-compatible (8 byte stride).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CellGpu {
    elevation_m: f32,
    color_rgba8: u32,
}

/// Per-cell static data: the cell centre's lat/lon, used to resolve the
/// Mercator seam without splitting cells.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CellStaticGpu {
    lat_rad: f32,
    lon_rad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobePush {
    view_proj: [f32; 16],
    /// xyz = camera position (km), w = vertical exaggeration.
    cam_pos_exag: [f32; 4],
    /// x = radius_km, y = base_offset_m, z = elevation normaliser, w = centre lon.
    params: [f32; 4],
    /// x = 0 globe / 1 mercator.
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StarPush {
    inv_view_proj: [f32; 16],
    /// x = seed, y = brightness.
    params: [f32; 4],
}

/// A chunk's draw range and bounding cone.
struct ChunkDraw {
    first_index: u32,
    index_count: u32,
    axis: Vec3,
    cos_radius: f32,
}

/// Per-frame parameters handed to the globe pass.
#[derive(Debug, Clone, Copy)]
pub struct GlobeParams {
    pub view_proj: Mat4,
    pub camera_pos_km: Vec3,
    pub exaggeration: f32,
    /// Constant radial offset added to every vertex, metres.
    pub base_offset_m: f32,
    pub radius_km: f32,
    pub mode: ViewMode,
    /// Mercator view-centre longitude, radians.
    pub center_lon_rad: f32,
    /// Enable frustum + horizon culling of chunks.
    pub cull: bool,
    pub star_seed: f32,
    pub star_brightness: f32,
}

/// Statistics from the last recorded frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrawStats {
    pub chunks_total: usize,
    pub chunks_drawn: usize,
    pub triangles_drawn: usize,
}

/// Globe geometry, per-cell data and pipelines.
pub struct GlobeRenderer {
    n_cells: usize,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
    chunks: Vec<ChunkDraw>,
    statics: Option<Buffer>,
    cell_buffers: Vec<Buffer>,
    staging: Vec<Buffer>,
    dirty: [bool; FRAMES_IN_FLIGHT],
    pending: Vec<CellGpu>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    star_layout: vk::PipelineLayout,
    star_pipeline: vk::Pipeline,
    pub stats: DrawStats,
}

fn load_shader(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    let mut cursor = std::io::Cursor::new(spv);
    let code = ash::util::read_spv(&mut cursor)?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    Ok(unsafe { device.create_shader_module(&ci, None) }?)
}

const GLOBE_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/globe.vert.spv"));
const GLOBE_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/globe.frag.spv"));
const STAR_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/star.vert.spv"));
const STAR_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/star.frag.spv"));

impl GlobeRenderer {
    /// Create the pipelines and descriptor objects. Geometry arrives later via
    /// [`GlobeRenderer::upload_mesh`].
    pub fn new(gpu: &Gpu, render_pass: vk::RenderPass) -> Result<GlobeRenderer> {
        let device = &gpu.device;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
        ];
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2 * FRAMES_IN_FLIGHT as u32)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        let layouts = [descriptor_layout; FRAMES_IN_FLIGHT];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }?;

        let set_layouts = [descriptor_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<GlobePush>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        }?;
        let star_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<StarPush>() as u32)];
        let star_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&star_ranges),
                None,
            )
        }?;

        let pipeline = create_globe_pipeline(gpu, render_pass, pipeline_layout)?;
        let star_pipeline = create_star_pipeline(gpu, render_pass, star_layout)?;

        Ok(GlobeRenderer {
            n_cells: 0,
            vertices: None,
            indices: None,
            chunks: Vec::new(),
            statics: None,
            cell_buffers: Vec::new(),
            staging: Vec::new(),
            dirty: [false; FRAMES_IN_FLIGHT],
            pending: Vec::new(),
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            star_layout,
            star_pipeline,
            stats: DrawStats::default(),
        })
    }

    /// Number of cells the current geometry was built for.
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// Build and upload the geometry for `mesh`. Replaces any previous mesh.
    pub fn upload_mesh(&mut self, gpu: &Gpu, mesh: &Mesh) -> Result<()> {
        unsafe { gpu.device.device_wait_idle() }?;
        self.free_mesh(gpu);

        let n_cells = mesh.n_cells();
        let mut vertices: Vec<GlobeVertex> = Vec::with_capacity(n_cells * 6);
        let mut indices: Vec<u32> = Vec::with_capacity(n_cells * 12);
        let mut chunks = Vec::with_capacity(mesh.chunks.len());

        // Which cells touch each corner vertex (three, on a Goldberg mesh).
        let mut corner_cells: Vec<[u32; 3]> = vec![[u32::MAX; 3]; mesh.vertices.len()];
        for cell in 0..n_cells as u32 {
            for &v in mesh.corners_of(cell) {
                let slot = &mut corner_cells[v as usize];
                if let Some(s) = slot.iter_mut().find(|s| **s == u32::MAX) {
                    *s = cell;
                }
            }
        }

        for chunk in &mesh.chunks {
            let first_index = indices.len() as u32;
            let axis = chunk.center.normalize();
            // The stored cone bounds cell centres; widen it to cover corners.
            let mut cos_radius = chunk.cos_radius.min(1.0);
            for &cell in &chunk.cells {
                let corners = mesh.corners_of(cell);
                if corners.len() < 3 {
                    continue;
                }
                let base = vertices.len() as u32;
                for &c in corners {
                    let v = mesh.vertices[c as usize];
                    cos_radius = cos_radius.min(axis.dot(v.normalize()));
                    // Owning cell first, then the others sharing this corner.
                    let mut ids = [cell; 3];
                    let mut n = 1;
                    for &other in &corner_cells[c as usize] {
                        if other != u32::MAX && other != cell && n < 3 {
                            ids[n] = other;
                            n += 1;
                        }
                    }
                    vertices.push(GlobeVertex {
                        pos: v.to_array(),
                        cells: ids,
                    });
                }
                for i in 1..corners.len() as u32 - 1 {
                    indices.push(base);
                    indices.push(base + i);
                    indices.push(base + i + 1);
                }
            }
            chunks.push(ChunkDraw {
                first_index,
                index_count: indices.len() as u32 - first_index,
                axis,
                cos_radius: cos_radius.clamp(-1.0, 1.0),
            });
        }
        log::info!(
            "globe geometry: {} cells, {} vertices, {} triangles, {} chunks",
            n_cells,
            vertices.len(),
            indices.len() / 3,
            chunks.len()
        );

        let statics: Vec<CellStaticGpu> = (0..n_cells)
            .map(|i| {
                let ll = mesh.latlon[i];
                CellStaticGpu {
                    lat_rad: ll[0],
                    lon_rad: ll[1],
                }
            })
            .collect();

        self.vertices = Some(upload_device_local(
            gpu,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "globe vertices",
        )?);
        self.indices = Some(upload_device_local(
            gpu,
            &indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
            "globe indices",
        )?);
        self.statics = Some(upload_device_local(
            gpu,
            &statics,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "cell statics",
        )?);
        self.chunks = chunks;
        self.n_cells = n_cells;

        // Per-frame-in-flight cell buffers so a 10 Hz update never waits on the
        // GPU: the frame about to be recorded owns its own copy.
        let bytes = (n_cells * std::mem::size_of::<CellGpu>()) as u64;
        for i in 0..FRAMES_IN_FLIGHT {
            self.cell_buffers.push(Buffer::new(
                gpu,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "cell data",
            )?);
            self.staging.push(Buffer::new(
                gpu,
                bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "cell staging",
            )?);
            let cells_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.cell_buffers[i].handle)
                .range(vk::WHOLE_SIZE)];
            let statics_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.statics.as_ref().unwrap().handle)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&cells_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&statics_info),
            ];
            unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
        }

        self.pending = vec![CellGpu::default(); n_cells];
        self.dirty = [true; FRAMES_IN_FLIGHT];
        Ok(())
    }

    /// Replace the per-cell elevation and colour. Cheap and non-blocking:
    /// the data is staged per frame in flight, so this may be called at the
    /// snapshot rate (up to ~10 Hz) without stalling the GPU.
    pub fn update_cells(&mut self, elevation_m: &[f32], color_rgba8: &[[u8; 4]]) -> Result<()> {
        if self.n_cells == 0 {
            return Ok(());
        }
        if elevation_m.len() != self.n_cells || color_rgba8.len() != self.n_cells {
            return Err(anyhow!(
                "update_cells: expected {} cells, got {} elevations and {} colours",
                self.n_cells,
                elevation_m.len(),
                color_rgba8.len()
            ));
        }
        for (dst, (e, c)) in self
            .pending
            .iter_mut()
            .zip(elevation_m.iter().zip(color_rgba8.iter()))
        {
            dst.elevation_m = *e;
            dst.color_rgba8 = u32::from_le_bytes(*c);
        }
        self.dirty = [true; FRAMES_IN_FLIGHT];
        Ok(())
    }

    /// Record the pending cell-data copy for `frame`, if any.
    pub fn record_uploads(&mut self, gpu: &Gpu, cb: vk::CommandBuffer, frame: usize) -> Result<()> {
        if !self.dirty[frame] || self.cell_buffers.is_empty() {
            return Ok(());
        }
        self.staging[frame].write(&self.pending)?;
        let size = self.staging[frame].size;
        unsafe {
            gpu.device.cmd_copy_buffer(
                cb,
                self.staging[frame].handle,
                self.cell_buffers[frame].handle,
                &[vk::BufferCopy::default().size(size)],
            );
            let barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.cell_buffers[frame].handle)
                .size(vk::WHOLE_SIZE)];
            gpu.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barrier,
                &[],
            );
        }
        self.dirty[frame] = false;
        Ok(())
    }

    /// Record the starfield background (fullscreen triangle, no depth write).
    pub fn record_starfield(&self, gpu: &Gpu, cb: vk::CommandBuffer, params: &GlobeParams) {
        let push = StarPush {
            inv_view_proj: params.view_proj.inverse().to_cols_array(),
            params: [params.star_seed, params.star_brightness, 0.0, 0.0],
        };
        unsafe {
            gpu.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.star_pipeline);
            gpu.device.cmd_push_constants(
                cb,
                self.star_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            gpu.device.cmd_draw(cb, 3, 1, 0, 0);
        }
    }

    /// Record the globe (or Mercator) pass, culling chunks on the CPU.
    pub fn record_globe(
        &mut self,
        gpu: &Gpu,
        cb: vk::CommandBuffer,
        frame: usize,
        params: &GlobeParams,
    ) {
        let (Some(vertices), Some(indices)) = (&self.vertices, &self.indices) else {
            return;
        };
        let exaggeration = params.exaggeration;
        let push = GlobePush {
            view_proj: params.view_proj.to_cols_array(),
            cam_pos_exag: [
                params.camera_pos_km.x,
                params.camera_pos_km.y,
                params.camera_pos_km.z,
                exaggeration,
            ],
            params: [
                params.radius_km,
                params.base_offset_m,
                ELEV_NORM_M,
                params.center_lon_rad,
            ],
            flags: [
                match params.mode {
                    ViewMode::Globe => 0,
                    ViewMode::Mercator => 1,
                },
                0,
                0,
                0,
            ],
        };
        unsafe {
            gpu.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            gpu.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[frame]],
                &[],
            );
            gpu.device
                .cmd_bind_vertex_buffers(cb, 0, &[vertices.handle], &[0]);
            gpu.device
                .cmd_bind_index_buffer(cb, indices.handle, 0, vk::IndexType::UINT32);
            gpu.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
        }

        // Radial extent of the displaced geometry, for culling.
        let disp_km = (ELEV_NORM_M * exaggeration + params.base_offset_m) * 0.001;
        let r_min = params.radius_km - disp_km;
        let r_max = params.radius_km + disp_km;
        let frustum = Frustum::from_view_proj(params.view_proj);
        let cull = params.cull && params.mode == ViewMode::Globe;

        let mut drawn = 0usize;
        let mut tris = 0usize;
        for chunk in &self.chunks {
            if chunk.index_count == 0 {
                continue;
            }
            if cull
                && !chunk_visible(
                    &frustum,
                    params.camera_pos_km,
                    chunk.axis,
                    chunk.cos_radius,
                    r_min,
                    r_max,
                    r_min,
                    true,
                )
            {
                continue;
            }
            unsafe {
                gpu.device
                    .cmd_draw_indexed(cb, chunk.index_count, 1, chunk.first_index, 0, 0)
            };
            drawn += 1;
            tris += chunk.index_count as usize / 3;
        }
        self.stats = DrawStats {
            chunks_total: self.chunks.len(),
            chunks_drawn: drawn,
            triangles_drawn: tris,
        };
    }

    fn free_mesh(&mut self, gpu: &Gpu) {
        let mut allocator = gpu.alloc();
        for b in self
            .vertices
            .iter_mut()
            .chain(self.indices.iter_mut())
            .chain(self.statics.iter_mut())
        {
            b.destroy(&gpu.device, &mut allocator);
        }
        for b in self.cell_buffers.iter_mut().chain(self.staging.iter_mut()) {
            b.destroy(&gpu.device, &mut allocator);
        }
        drop(allocator);
        self.vertices = None;
        self.indices = None;
        self.statics = None;
        self.cell_buffers.clear();
        self.staging.clear();
        self.chunks.clear();
        self.pending.clear();
        self.n_cells = 0;
    }

    /// Destroy every GPU object owned here. Device must be idle.
    pub fn destroy(&mut self, gpu: &Gpu) {
        self.free_mesh(gpu);
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline(self.star_pipeline, None);
            gpu.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            gpu.device.destroy_pipeline_layout(self.star_layout, None);
            gpu.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            gpu.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

fn create_globe_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let device = &gpu.device;
    let vs = load_shader(device, GLOBE_VERT)?;
    let fs = load_shader(device, GLOBE_FRAG)?;
    let name = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs)
            .name(name),
    ];
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<GlobeVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attributes = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_UINT)
            .offset(12),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    // Back-face culling is off on purpose: the Mercator branch reprojects the
    // same triangles and can flip their winding, and reverse-Z makes the extra
    // overdraw cheap.
    let pipeline = build_pipeline(gpu, render_pass, layout, &stages, &vertex_input, true, true);
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

fn create_star_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let device = &gpu.device;
    let vs = load_shader(device, STAR_VERT)?;
    let fs = load_shader(device, STAR_FRAG)?;
    let name = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs)
            .name(name),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let pipeline = build_pipeline(
        gpu,
        render_pass,
        layout,
        &stages,
        &vertex_input,
        false,
        false,
    );
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

fn build_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    stages: &[vk::PipelineShaderStageCreateInfo],
    vertex_input: &vk::PipelineVertexInputStateCreateInfo,
    depth_test: bool,
    depth_write: bool,
) -> Result<vk::Pipeline> {
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Reverse-Z: the depth buffer is cleared to 0.0 and nearer fragments have
    // the larger depth value, hence GREATER.
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_test)
        .depth_write_enable(depth_write)
        .depth_compare_op(vk::CompareOp::GREATER)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(stages)
        .vertex_input_state(vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipelines = unsafe {
        gpu.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[ci], None)
    }
    .map_err(|(_, e)| anyhow!("vkCreateGraphicsPipelines: {e}"))?;
    Ok(pipelines[0])
}
