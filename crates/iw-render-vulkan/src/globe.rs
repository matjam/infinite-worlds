//! The globe pipeline: static per-chunk geometry, dynamic per-cell data, the
//! sky (starfield + atmospheric halo) pass and the optional cloud shell.
//!
//! Geometry is built once from the `Mesh` (a triangle fan per cell over its
//! corners) into one vertex and one index buffer, with a draw range per chunk
//! for culling. Per-cell elevation, colour and beauty shading inputs live in a
//! storage buffer indexed by the cell id carried on every vertex, so a data
//! update never rebuilds geometry.
//!
//! The cloud shell is a separate, static icosphere (see [`CLOUD_SHELL_LEVEL`])
//! whose per-vertex coverage is refilled from the CPU; its structure comes
//! from noise in the fragment stage, so it does not inherit the planet's cell
//! resolution.

use std::collections::HashMap;

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

/// Icosahedron subdivisions of the cloud shell: 10242 vertices, 20480
/// triangles. Fine enough that the interpolated coverage field shows no
/// facets, coarse enough to cost nothing next to the planet itself.
pub const CLOUD_SHELL_LEVEL: u32 = 5;

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

/// What a cell's surface is made of, for the beauty shader's water handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SurfaceKind {
    /// Anything above sea level, ice included.
    #[default]
    Land = 0,
    /// Open ocean: full-strength sun glint, sky reflection scaled by depth.
    Ocean = 1,
    /// Inland water: a smaller, softer highlight.
    Lake = 2,
}

/// Per-cell shading inputs for the beauty view. Everything the lit shader
/// needs beyond elevation and albedo; see `crates/iw-app/src/beauty.rs` for
/// how it is derived from a snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct CellShade {
    /// Elevation gradient in the cell's own tangent basis, metres per metre.
    /// Reconstructing the normal from a gradient (rather than baking one) keeps
    /// the shading exact under any vertical exaggeration.
    pub grad_east: f32,
    /// See [`CellShade::grad_east`].
    pub grad_north: f32,
    pub kind: SurfaceKind,
    /// Position on the ocean depth ramp, 0 at the shoreline, 1 at the abyss.
    pub depth_t: f32,
    /// Ice cover fraction, 0..1. Kills the glint and the water sky term.
    pub ice_t: f32,
}

/// Per-cell dynamic data, std430-compatible (16 byte stride).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CellGpu {
    elevation_m: f32,
    color_rgba8: u32,
    /// Two halves: d elevation / d east, d elevation / d north (m per m).
    gradient: u32,
    /// Byte 0 surface kind, byte 1 depth ramp, byte 2 ice, byte 3 reserved.
    material: u32,
}

/// Pack two floats as IEEE binary16, matching GLSL `unpackHalf2x16`.
fn pack_half2(x: f32, y: f32) -> u32 {
    (f16_bits(x) as u32) | ((f16_bits(y) as u32) << 16)
}

/// f32 -> binary16 bits, round half up, saturating on overflow.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf or NaN: keep NaN a NaN, clamp infinities to the largest finite
        // half (the shader only ever divides these, never inspects them).
        return sign | if mantissa != 0 { 0x7e00 } else { 0x7bff };
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1f {
        return sign | 0x7bff; // saturate rather than produce infinity
    }
    if unbiased <= 0 {
        // Subnormal half (or zero): shift the implicit 1 back in.
        if unbiased < -10 {
            return sign;
        }
        let m = mantissa | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let half = (m >> shift) + ((m >> (shift - 1)) & 1);
        return sign | half as u16;
    }
    let half = ((unbiased as u32) << 10) | (mantissa >> 13);
    let round = (mantissa >> 12) & 1;
    sign | (half + round) as u16
}

/// Pack four 0..1 values into RGBA8 the way GLSL `unpackUnorm4x8` reads them.
fn pack_unorm4(x: f32, y: f32, z: f32, w: f32) -> u32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    b(x) | (b(y) << 8) | (b(z) << 16) | (b(w) << 24)
}

/// Per-cell static data: the cell centre's lat/lon, used to resolve the
/// Mercator seam without splitting cells.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CellStaticGpu {
    lat_rad: f32,
    lon_rad: f32,
}

/// Exactly 128 bytes: the minimum `maxPushConstantsSize` every Vulkan
/// implementation is required to support.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobePush {
    view_proj: [f32; 16],
    /// xyz = camera position (km), w = vertical exaggeration.
    cam_pos_exag: [f32; 4],
    /// x = radius_km, y = base_offset_m, z = elevation normaliser, w = centre lon.
    params: [f32; 4],
    /// xyz = sun direction (unit), w = atmosphere fade 0..1.
    sun: [f32; 4],
    /// x = 0 globe / 1 mercator, y = 1 when beauty shading is on.
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StarPush {
    inv_view_proj: [f32; 16],
    /// x = seed, y = brightness, z = planet radius km.
    params: [f32; 4],
    /// xyz = camera position (km), w = halo strength 0..1.
    cam: [f32; 4],
    /// xyz = sun direction (unit).
    sun: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudPush {
    view_proj: [f32; 16],
    /// xyz = camera position (km), w = planet radius (km).
    cam_radius: [f32; 4],
    /// xyz = sun direction (unit), w = deck rotation phase (rad).
    sun_phase: [f32; 4],
    /// x = opacity fade 0..1, y = noise seed.
    misc: [f32; 4],
}

/// One vertex of a river ribbon (CPU-built quads following `flow_to`).
/// Elevation rides in the vertex so the ribbon hugs the displaced terrain
/// under any exaggeration without cell lookups.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RiverVertex {
    /// Unit sphere direction.
    pub pos: [f32; 3],
    /// Terrain elevation at this point, metres (pre-exaggeration).
    pub elevation_m: f32,
    /// Premultiplied-nothing RGBA; alpha carries the flux fade.
    pub color: [f32; 4],
}

/// Radial lift of the river ribbons above the terrain, metres
/// (pre-exaggeration, so it scales with the terrain and never z-fights).
const RIVER_LIFT_M: f32 = 30.0;

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
    /// Sea level, metres — the fragment shader's shoreline-crinkle reference
    /// (rides in the push-constant slot Mercator uses for depth normalising,
    /// so the crinkle is globe-only).
    pub sea_level_m: f32,
    pub radius_km: f32,
    pub mode: ViewMode,
    /// Mercator view-centre longitude, radians.
    pub center_lon_rad: f32,
    /// Enable frustum + horizon culling of chunks.
    pub cull: bool,
    pub star_seed: f32,
    pub star_brightness: f32,
    /// Lit beauty view (relief shading, glint, limb) instead of the flat
    /// data-layer look.
    pub beauty: bool,
    /// Unit vector towards the sun; see [`crate::beauty::sun_direction`].
    pub sun_dir: Vec3,
    /// Strength of the atmospheric halo and limb, 0..1; see
    /// [`crate::beauty::haze_fade`].
    pub atmosphere: f32,
    /// Opacity of the cloud shell, 0..1. Zero skips the pass entirely.
    pub cloud_opacity: f32,
    /// Cloud deck rotation phase, radians.
    pub cloud_phase_rad: f32,
    /// Seed offset for the cloud noise, so two planets get different weather.
    pub cloud_seed: f32,
    /// Mean cell pitch as an angle on the unit sphere, radians. The shoreline
    /// crinkle scales its noise wavelength by this, so coarse worlds get a
    /// calm wander instead of froth.
    pub cell_pitch_rad: f32,
}

impl Default for GlobeParams {
    fn default() -> Self {
        GlobeParams {
            view_proj: Mat4::IDENTITY,
            camera_pos_km: Vec3::ZERO,
            exaggeration: 1.0,
            base_offset_m: 0.0,
            sea_level_m: 0.0,
            radius_km: iw_mesh::EARTH_RADIUS_KM,
            mode: ViewMode::Globe,
            center_lon_rad: 0.0,
            cull: true,
            star_seed: 1.0,
            star_brightness: 1.0,
            beauty: false,
            sun_dir: Vec3::Z,
            atmosphere: 1.0,
            cloud_opacity: 0.0,
            cloud_phase_rad: 0.0,
            cloud_seed: 0.0,
            cell_pitch_rad: 0.016,
        }
    }
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
    shell_dirs: Vec<Vec3>,
    shell_vertices: Option<Buffer>,
    shell_indices: Option<Buffer>,
    shell_index_count: u32,
    coverage_buffers: Vec<Buffer>,
    coverage_staging: Vec<Buffer>,
    coverage_dirty: [bool; FRAMES_IN_FLIGHT],
    coverage_pending: Vec<f32>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    star_layout: vk::PipelineLayout,
    star_pipeline: vk::Pipeline,
    halo_pipeline: vk::Pipeline,
    cloud_layout: vk::PipelineLayout,
    cloud_pipeline: vk::Pipeline,
    river_pipeline: vk::Pipeline,
    river_vertices: Option<Buffer>,
    river_count: u32,
    river_capacity: u64,
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
const CLOUD_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cloud.vert.spv"));
const CLOUD_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cloud.frag.spv"));
const RIVER_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/river.vert.spv"));
const RIVER_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/river.frag.spv"));

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
            // Cloud shell coverage. Lives in the same set so the cloud pass can
            // reuse the layout the globe pass already binds.
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
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
            .descriptor_count(3 * FRAMES_IN_FLIGHT as u32)];
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
        // The fragment stage reads the sun, camera and flags out of the same
        // block, so the range covers both stages.
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
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
        let cloud_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<CloudPush>() as u32)];
        let cloud_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&cloud_ranges),
                None,
            )
        }?;

        let pipeline = create_globe_pipeline(gpu, render_pass, pipeline_layout)?;
        let star_pipeline = create_star_pipeline(gpu, render_pass, star_layout, false)?;
        let halo_pipeline = create_star_pipeline(gpu, render_pass, star_layout, true)?;
        let cloud_pipeline = create_cloud_pipeline(gpu, render_pass, cloud_layout)?;
        let river_pipeline = create_river_pipeline(gpu, render_pass, pipeline_layout)?;

        // The cloud shell never changes; only its coverage does.
        let (shell_dirs, shell_indices_cpu) = icosphere(CLOUD_SHELL_LEVEL);
        let shell_vertices = upload_device_local(
            gpu,
            &shell_dirs.iter().map(|d| d.to_array()).collect::<Vec<_>>(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "cloud shell vertices",
        )?;
        let shell_index_count = shell_indices_cpu.len() as u32;
        let shell_indices = upload_device_local(
            gpu,
            &shell_indices_cpu,
            vk::BufferUsageFlags::INDEX_BUFFER,
            "cloud shell indices",
        )?;
        let coverage_bytes = (shell_dirs.len() * std::mem::size_of::<f32>()) as u64;
        let mut coverage_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut coverage_staging = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            coverage_buffers.push(Buffer::new(
                gpu,
                coverage_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "cloud coverage",
            )?);
            coverage_staging.push(Buffer::new(
                gpu,
                coverage_bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "cloud coverage staging",
            )?);
        }
        let coverage_pending = vec![0.0f32; shell_dirs.len()];

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
            shell_dirs,
            shell_vertices: Some(shell_vertices),
            shell_indices: Some(shell_indices),
            shell_index_count,
            coverage_buffers,
            coverage_staging,
            coverage_dirty: [true; FRAMES_IN_FLIGHT],
            coverage_pending,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            star_layout,
            star_pipeline,
            halo_pipeline,
            cloud_layout,
            cloud_pipeline,
            river_pipeline,
            river_vertices: None,
            river_count: 0,
            river_capacity: 0,
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
            let coverage_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.coverage_buffers[i].handle)
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
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&coverage_info),
            ];
            unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
        }

        self.pending = vec![CellGpu::default(); n_cells];
        self.dirty = [true; FRAMES_IN_FLIGHT];
        Ok(())
    }

    /// Replace the per-cell elevation and colour, leaving the beauty shading
    /// inputs flat (no relief, everything treated as land). This is what the
    /// data layers want: their palettes are read off the screen.
    pub fn update_cells(&mut self, elevation_m: &[f32], color_rgba8: &[[u8; 4]]) -> Result<()> {
        self.update_cells_shaded(elevation_m, color_rgba8, None)
    }

    /// Replace the per-cell elevation, colour and (optionally) the beauty
    /// shading inputs. Cheap and non-blocking: the data is staged per frame in
    /// flight, so this may be called at the snapshot rate (up to ~10 Hz)
    /// without stalling the GPU.
    pub fn update_cells_shaded(
        &mut self,
        elevation_m: &[f32],
        color_rgba8: &[[u8; 4]],
        shade: Option<&[CellShade]>,
    ) -> Result<()> {
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
        if let Some(shade) = shade {
            if shade.len() != self.n_cells {
                return Err(anyhow!(
                    "update_cells: expected {} cells, got {} shading records",
                    self.n_cells,
                    shade.len()
                ));
            }
        }
        for (i, (dst, (e, c))) in self
            .pending
            .iter_mut()
            .zip(elevation_m.iter().zip(color_rgba8.iter()))
            .enumerate()
        {
            dst.elevation_m = *e;
            dst.color_rgba8 = u32::from_le_bytes(*c);
            match shade {
                Some(s) => {
                    let s = s[i];
                    dst.gradient = pack_half2(s.grad_east, s.grad_north);
                    dst.material =
                        pack_unorm4(s.kind as u8 as f32 / 255.0, s.depth_t, s.ice_t, 0.0);
                }
                None => {
                    dst.gradient = 0;
                    dst.material = 0;
                }
            }
        }
        self.dirty = [true; FRAMES_IN_FLIGHT];
        Ok(())
    }

    /// Unit directions of the cloud shell's vertices, in planet coordinates.
    /// The caller samples its coverage field at these points and hands the
    /// result back through [`GlobeRenderer::update_cloud_coverage`].
    pub fn cloud_shell_dirs(&self) -> &[Vec3] {
        &self.shell_dirs
    }

    /// Replace the per-shell-vertex cloud coverage (0..1).
    pub fn update_cloud_coverage(&mut self, coverage: &[f32]) -> Result<()> {
        if coverage.len() != self.coverage_pending.len() {
            return Err(anyhow!(
                "update_cloud_coverage: expected {} shell vertices, got {}",
                self.coverage_pending.len(),
                coverage.len()
            ));
        }
        self.coverage_pending.copy_from_slice(coverage);
        self.coverage_dirty = [true; FRAMES_IN_FLIGHT];
        Ok(())
    }

    /// Record the pending cell-data and cloud-coverage copies for `frame`.
    pub fn record_uploads(&mut self, gpu: &Gpu, cb: vk::CommandBuffer, frame: usize) -> Result<()> {
        let mut barriers: Vec<vk::BufferMemoryBarrier> = Vec::with_capacity(2);
        if self.dirty[frame] && !self.cell_buffers.is_empty() {
            self.staging[frame].write(&self.pending)?;
            let size = self.staging[frame].size;
            unsafe {
                gpu.device.cmd_copy_buffer(
                    cb,
                    self.staging[frame].handle,
                    self.cell_buffers[frame].handle,
                    &[vk::BufferCopy::default().size(size)],
                );
            }
            barriers.push(buffer_barrier(self.cell_buffers[frame].handle));
            self.dirty[frame] = false;
        }
        if self.coverage_dirty[frame] && !self.coverage_buffers.is_empty() {
            self.coverage_staging[frame].write(&self.coverage_pending)?;
            let size = self.coverage_staging[frame].size;
            unsafe {
                gpu.device.cmd_copy_buffer(
                    cb,
                    self.coverage_staging[frame].handle,
                    self.coverage_buffers[frame].handle,
                    &[vk::BufferCopy::default().size(size)],
                );
            }
            barriers.push(buffer_barrier(self.coverage_buffers[frame].handle));
            self.coverage_dirty[frame] = false;
        }
        if !barriers.is_empty() {
            unsafe {
                gpu.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
            }
        }
        Ok(())
    }

    /// Record the starfield background (fullscreen triangle, opaque, no depth
    /// write, drawn before everything else).
    pub fn record_starfield(&self, gpu: &Gpu, cb: vk::CommandBuffer, params: &GlobeParams) {
        self.record_sky(gpu, cb, params, false);
    }

    /// Record the atmospheric halo: the same fullscreen triangle, alpha
    /// blended over the finished globe so the haze fogs the limb (and anything
    /// vertical exaggeration pushes out past the silhouette) instead of being
    /// painted over by it.
    pub fn record_halo(&self, gpu: &Gpu, cb: vk::CommandBuffer, params: &GlobeParams) {
        // A globe-view effect: a Mercator map has no limb to scatter around.
        if params.mode != ViewMode::Globe || !params.beauty || params.atmosphere <= 0.0 {
            return;
        }
        self.record_sky(gpu, cb, params, true);
    }

    fn record_sky(&self, gpu: &Gpu, cb: vk::CommandBuffer, params: &GlobeParams, halo: bool) {
        let push = StarPush {
            inv_view_proj: params.view_proj.inverse().to_cols_array(),
            params: [
                params.star_seed,
                params.star_brightness,
                params.radius_km,
                halo as u32 as f32,
            ],
            cam: [
                params.camera_pos_km.x,
                params.camera_pos_km.y,
                params.camera_pos_km.z,
                params.atmosphere.clamp(0.0, 1.0),
            ],
            sun: [params.sun_dir.x, params.sun_dir.y, params.sun_dir.z, 0.0],
        };
        let pipeline = if halo {
            self.halo_pipeline
        } else {
            self.star_pipeline
        };
        unsafe {
            gpu.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
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
                // Globe: sea level for the fragment shoreline crinkle.
                // Mercator: the depth normaliser its vertex branch needs.
                match params.mode {
                    ViewMode::Globe => params.sea_level_m,
                    ViewMode::Mercator => ELEV_NORM_M,
                },
                params.center_lon_rad,
            ],
            sun: [
                params.sun_dir.x,
                params.sun_dir.y,
                params.sun_dir.z,
                params.atmosphere.clamp(0.0, 1.0),
            ],
            flags: [
                match params.mode {
                    ViewMode::Globe => 0,
                    ViewMode::Mercator => 1,
                },
                params.beauty as u32,
                params.cell_pitch_rad.to_bits(),
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
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
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

    /// Record the cloud shell over the globe: alpha blended, depth tested
    /// against the surface but never writing depth, back faces culled so the
    /// far side of the shell does not double up around the limb.
    ///
    /// Globe view only — a cloud deck drawn on a Mercator map would be a lie
    /// about where the weather is, and the shell has no projection anyway.
    /// Replace the river ribbon geometry. Call when a new sim view arrives —
    /// the buffer is host-visible and rewritten in place unless it must grow
    /// (growth stalls the device, which is fine at view-update cadence).
    pub fn set_rivers(&mut self, gpu: &Gpu, verts: &[RiverVertex]) -> Result<()> {
        self.river_count = verts.len() as u32;
        if verts.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::size_of_val(verts) as u64;
        if bytes > self.river_capacity {
            unsafe { gpu.device.device_wait_idle() }?;
            if let Some(mut b) = self.river_vertices.take() {
                b.destroy(&gpu.device, &mut gpu.alloc());
            }
            let cap = bytes.next_power_of_two();
            self.river_vertices = Some(Buffer::new(
                gpu,
                cap,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
                "river ribbons",
            )?);
            self.river_capacity = cap;
        }
        self.river_vertices
            .as_mut()
            .expect("river buffer sized above")
            .write(verts)
    }

    /// Draw the river ribbons over the globe surface (globe mode only).
    pub fn record_rivers(&self, gpu: &Gpu, cb: vk::CommandBuffer, params: &GlobeParams) {
        if params.mode != ViewMode::Globe || self.river_count == 0 {
            return;
        }
        let Some(vertices) = &self.river_vertices else {
            return;
        };
        let push = GlobePush {
            view_proj: params.view_proj.to_cols_array(),
            cam_pos_exag: [
                params.camera_pos_km.x,
                params.camera_pos_km.y,
                params.camera_pos_km.z,
                params.exaggeration,
            ],
            params: [
                params.radius_km,
                params.base_offset_m + RIVER_LIFT_M * params.exaggeration,
                ELEV_NORM_M,
                params.center_lon_rad,
            ],
            sun: [
                params.sun_dir.x,
                params.sun_dir.y,
                params.sun_dir.z,
                params.atmosphere.clamp(0.0, 1.0),
            ],
            flags: [0, params.beauty as u32, 0, 0],
        };
        unsafe {
            gpu.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.river_pipeline);
            gpu.device
                .cmd_bind_vertex_buffers(cb, 0, &[vertices.handle], &[0]);
            gpu.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            gpu.device.cmd_draw(cb, self.river_count, 1, 0, 0);
        }
    }

    pub fn record_clouds(&self, gpu: &Gpu, cb: vk::CommandBuffer, frame: usize, p: &GlobeParams) {
        if p.mode != ViewMode::Globe || p.cloud_opacity <= 0.0 || self.shell_index_count == 0 {
            return;
        }
        let (Some(vertices), Some(indices)) = (&self.shell_vertices, &self.shell_indices) else {
            return;
        };
        let push = CloudPush {
            view_proj: p.view_proj.to_cols_array(),
            cam_radius: [
                p.camera_pos_km.x,
                p.camera_pos_km.y,
                p.camera_pos_km.z,
                p.radius_km,
            ],
            sun_phase: [p.sun_dir.x, p.sun_dir.y, p.sun_dir.z, p.cloud_phase_rad],
            misc: [p.cloud_opacity.clamp(0.0, 1.0), p.cloud_seed, 0.0, 0.0],
        };
        unsafe {
            gpu.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.cloud_pipeline);
            gpu.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.cloud_layout,
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
                self.cloud_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            gpu.device
                .cmd_draw_indexed(cb, self.shell_index_count, 1, 0, 0, 0);
        }
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
        {
            let mut allocator = gpu.alloc();
            for b in self
                .shell_vertices
                .iter_mut()
                .chain(self.shell_indices.iter_mut())
            {
                b.destroy(&gpu.device, &mut allocator);
            }
            for b in self
                .coverage_buffers
                .iter_mut()
                .chain(self.coverage_staging.iter_mut())
            {
                b.destroy(&gpu.device, &mut allocator);
            }
        }
        self.shell_vertices = None;
        self.shell_indices = None;
        self.coverage_buffers.clear();
        self.coverage_staging.clear();
        if let Some(mut b) = self.river_vertices.take() {
            b.destroy(&gpu.device, &mut gpu.alloc());
        }
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline(self.star_pipeline, None);
            gpu.device.destroy_pipeline(self.halo_pipeline, None);
            gpu.device.destroy_pipeline(self.cloud_pipeline, None);
            gpu.device.destroy_pipeline(self.river_pipeline, None);
            gpu.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            gpu.device.destroy_pipeline_layout(self.star_layout, None);
            gpu.device.destroy_pipeline_layout(self.cloud_layout, None);
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
    let pipeline = build_pipeline(
        gpu,
        render_pass,
        layout,
        &stages,
        &vertex_input,
        PipelineOpts {
            depth_test: true,
            depth_write: true,
            ..PipelineOpts::default()
        },
    );
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

fn create_cloud_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let device = &gpu.device;
    let vs = load_shader(device, CLOUD_VERT)?;
    let fs = load_shader(device, CLOUD_FRAG)?;
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
        .stride(std::mem::size_of::<[f32; 3]>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attributes = [vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let pipeline = build_pipeline(
        gpu,
        render_pass,
        layout,
        &stages,
        &vertex_input,
        PipelineOpts {
            // Tested against the surface (so the far side of the shell is
            // hidden by the planet) but never written: the shell is a
            // translucent overlay.
            depth_test: true,
            depth_write: false,
            blend: true,
            // Only the near face of the shell contributes; without this the
            // rim beyond the silhouette would be drawn twice.
            cull_back: true,
        },
    );
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

fn create_river_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let device = &gpu.device;
    let vs = load_shader(device, RIVER_VERT)?;
    let fs = load_shader(device, RIVER_FRAG)?;
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
        .stride(std::mem::size_of::<RiverVertex>() as u32)
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
            .format(vk::Format::R32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let pipeline = build_pipeline(
        gpu,
        render_pass,
        layout,
        &stages,
        &vertex_input,
        PipelineOpts {
            // Tested against the terrain (rivers hide behind the limb) but
            // never written: they are a translucent overlay riding just above
            // the surface.
            depth_test: true,
            depth_write: false,
            blend: true,
            cull_back: false,
        },
    );
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

/// A buffer barrier from a transfer write to a vertex-stage read.
fn buffer_barrier(buffer: vk::Buffer) -> vk::BufferMemoryBarrier<'static> {
    vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .size(vk::WHOLE_SIZE)
}

/// Unit-sphere icosahedron subdivided `level` times: outward-facing (CCW)
/// triangles, shared vertices. Used for the cloud shell.
fn icosphere(level: u32) -> (Vec<Vec3>, Vec<u32>) {
    let t = (1.0 + 5.0f32.sqrt()) * 0.5;
    let mut verts: Vec<Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|v| Vec3::from_array(*v).normalize())
    .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..level {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::default();
        let mut next = Vec::with_capacity(faces.len() * 4);
        for f in &faces {
            let mut mid = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                *cache.entry(key).or_insert_with(|| {
                    let v = (verts[a as usize] + verts[b as usize]).normalize();
                    verts.push(v);
                    verts.len() as u32 - 1
                })
            };
            let a = mid(f[0], f[1], &mut verts);
            let b = mid(f[1], f[2], &mut verts);
            let c = mid(f[2], f[0], &mut verts);
            next.push([f[0], a, c]);
            next.push([f[1], b, a]);
            next.push([f[2], c, b]);
            next.push([a, b, c]);
        }
        faces = next;
    }
    let indices = faces.into_iter().flatten().collect();
    (verts, indices)
}

/// The sky pass. `blend` selects the halo variant: same shaders, alpha blended
/// over the finished frame instead of opaque under it.
fn create_star_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    blend: bool,
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
        PipelineOpts {
            blend,
            ..PipelineOpts::default()
        },
    );
    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }
    pipeline
}

/// The few fixed-function knobs the three passes disagree about.
#[derive(Debug, Clone, Copy, Default)]
struct PipelineOpts {
    depth_test: bool,
    depth_write: bool,
    /// Straight source-alpha blending (the cloud shell).
    blend: bool,
    cull_back: bool,
}

fn build_pipeline(
    gpu: &Gpu,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    stages: &[vk::PipelineShaderStageCreateInfo],
    vertex_input: &vk::PipelineVertexInputStateCreateInfo,
    opts: PipelineOpts,
) -> Result<vk::Pipeline> {
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(if opts.cull_back {
            vk::CullModeFlags::BACK
        } else {
            vk::CullModeFlags::NONE
        })
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Reverse-Z: the depth buffer is cleared to 0.0 and nearer fragments have
    // the larger depth value, hence GREATER.
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(opts.depth_test)
        .depth_write_enable(opts.depth_write)
        .depth_compare_op(vk::CompareOp::GREATER)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(opts.blend)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)];
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact bit patterns GLSL's `unpackHalf2x16` will read back.
    #[test]
    fn half_packing_matches_ieee_binary16() {
        for (v, bits) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (0.1, 0x2e66),
            (65504.0, 0x7bff),
            (1.0e9, 0x7bff), // saturates instead of going infinite
            (-1.0e9, 0xfbff),
            (1.0e-9, 0x0000), // underflows to zero
            (6.0e-8, 0x0001), // smallest subnormal
        ] {
            assert_eq!(f16_bits(v), bits, "{v} packed as {:#06x}", f16_bits(v));
        }
        assert!(f16_bits(f32::NAN) & 0x7fff > 0x7c00, "NaN must stay NaN");
        // Terrain gradients are small; they must survive the round trip.
        for g in [-0.4f32, -0.05, -0.001, 0.0, 0.001, 0.05, 0.4] {
            let packed = pack_half2(g, -g);
            let lo = half_to_f32((packed & 0xffff) as u16);
            let hi = half_to_f32((packed >> 16) as u16);
            assert!((lo - g).abs() < 1e-3 * g.abs().max(1e-3), "{g} -> {lo}");
            assert!((hi + g).abs() < 1e-3 * g.abs().max(1e-3), "{g} -> {hi}");
        }
    }

    fn half_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1f) as i32;
        let frac = (bits & 0x3ff) as f32;
        match exp {
            0 => sign * frac * 2f32.powi(-24),
            _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
        }
    }

    #[test]
    fn material_bytes_survive_the_round_trip() {
        let packed = pack_unorm4(SurfaceKind::Lake as u8 as f32 / 255.0, 0.5, 1.0, 0.0);
        let bytes = packed.to_le_bytes();
        assert_eq!(bytes[0], 2, "kind is byte 0");
        assert_eq!(bytes[1], 128);
        assert_eq!(bytes[2], 255);
        assert_eq!(bytes[3], 0);
        // Out-of-range input clamps rather than wrapping.
        assert_eq!(
            pack_unorm4(-1.0, 2.0, 0.0, 0.0).to_le_bytes(),
            [0, 255, 0, 0]
        );
    }

    #[test]
    fn cloud_shell_is_a_closed_unit_sphere() {
        let (verts, indices) = icosphere(2);
        assert_eq!(verts.len(), 10 * 4usize.pow(2) + 2);
        assert_eq!(indices.len(), 20 * 4usize.pow(2) * 3);
        for v in &verts {
            assert!((v.length() - 1.0).abs() < 1e-5);
        }
        // Every triangle faces outwards (CCW seen from outside), which is what
        // the cloud pipeline's back-face culling relies on.
        for f in indices.chunks(3) {
            let (a, b, c) = (
                verts[f[0] as usize],
                verts[f[1] as usize],
                verts[f[2] as usize],
            );
            let n = (b - a).cross(c - a);
            assert!(n.dot(a + b + c) > 0.0, "inward-facing triangle {f:?}");
        }
        assert_eq!(icosphere(CLOUD_SHELL_LEVEL).0.len(), 10242);
    }

    #[test]
    fn push_constants_fit_the_guaranteed_minimum() {
        assert_eq!(std::mem::size_of::<GlobePush>(), 128);
        assert!(std::mem::size_of::<StarPush>() <= 128);
        assert!(std::mem::size_of::<CloudPush>() <= 128);
        assert_eq!(std::mem::size_of::<CellGpu>(), 16);
    }
}
