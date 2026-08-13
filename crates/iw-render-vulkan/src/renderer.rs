//! The frame loop: swapchain acquire, one render pass (starfield, globe, egui),
//! submit, present. Two frames in flight.

use anyhow::{Context, Result};
use ash::vk;
use egui_ash_renderer::{allocator::GpuAllocator, Options, RenderMode};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::globe::{DrawStats, GlobeParams, GlobeRenderer, FRAMES_IN_FLIGHT};
use crate::gpu::Gpu;
use crate::swapchain::{create_render_pass, Swapchain};

/// What egui produced this frame. The texture delta is consumed (emptied) by
/// [`Renderer::render`], including on frames that are skipped.
pub struct EguiFrame<'a> {
    pub textures_delta: &'a mut egui::TexturesDelta,
    pub primitives: &'a [egui::ClippedPrimitive],
    pub pixels_per_point: f32,
}

struct FrameSync {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
}

/// Owns every Vulkan object and draws one frame per call to [`Renderer::render`].
pub struct Renderer {
    frames: Vec<FrameSync>,
    frame: usize,
    frame_counter: u64,
    render_pass: vk::RenderPass,
    swapchain: Option<Swapchain>,
    globe: Option<GlobeRenderer>,
    egui: Option<egui_ash_renderer::Renderer<GpuAllocator>>,
    pending_texture_frees: Vec<(u64, Vec<egui::TextureId>)>,
    needs_resize: bool,
    surface_size: (u32, u32),
    /// Declared last so it is dropped last.
    gpu: Gpu,
}

impl Renderer {
    /// Bring up Vulkan for `window` at the given pixel size.
    pub fn new<W>(window: &W, size: (u32, u32), app_name: &str) -> Result<Renderer>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        let gpu = Gpu::new(window, app_name)?;

        // The render pass needs a colour format, which needs a swapchain query.
        let formats = unsafe {
            gpu.surface_fn
                .get_physical_device_surface_formats(gpu.physical_device, gpu.surface)
        }?;
        let probe_format = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .or_else(|| {
                formats
                    .iter()
                    .find(|f| f.format == vk::Format::R8G8B8A8_UNORM)
            })
            .map(|f| f.format)
            .unwrap_or(formats[0].format);
        let render_pass = create_render_pass(&gpu, probe_format)?;

        let swapchain = Swapchain::new(&gpu, render_pass, size.0, size.1, None)?;
        debug_assert_eq!(swapchain.format, probe_format);

        let command_buffers = unsafe {
            gpu.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(gpu.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )
        }?;
        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for cb in command_buffers {
            frames.push(FrameSync {
                command_buffer: cb,
                image_available: unsafe {
                    gpu.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                }?,
                in_flight: unsafe {
                    gpu.device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                }?,
            });
        }

        let globe = GlobeRenderer::new(&gpu, render_pass)?;
        let egui = egui_ash_renderer::Renderer::with_gpu_allocator(
            gpu.allocator_arc(),
            gpu.device.clone(),
            RenderMode::RenderPass(render_pass),
            Options {
                in_flight_frames: FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
                // The swapchain is UNORM, so egui converts to sRGB itself.
                srgb_framebuffer: false,
            },
        )
        .map_err(|e| anyhow::anyhow!("egui renderer: {e}"))?;

        Ok(Renderer {
            frames,
            frame: 0,
            frame_counter: 0,
            render_pass,
            swapchain: Some(swapchain),
            globe: Some(globe),
            egui: Some(egui),
            pending_texture_frees: Vec::new(),
            needs_resize: false,
            surface_size: size,
            gpu,
        })
    }

    /// Name of the physical device in use.
    pub fn device_name(&self) -> &str {
        &self.gpu.device_name
    }

    /// Whether the Khronos validation layer is active.
    pub fn validation_enabled(&self) -> bool {
        self.gpu.validation_enabled
    }

    /// Culling statistics from the last recorded frame.
    pub fn stats(&self) -> DrawStats {
        self.globe.as_ref().map(|g| g.stats).unwrap_or_default()
    }

    /// Current swapchain extent aspect ratio.
    pub fn aspect(&self) -> f32 {
        self.swapchain.as_ref().map(|s| s.aspect()).unwrap_or(1.0)
    }

    /// Upload planet geometry. Safe to call again with a different mesh.
    pub fn upload_mesh(&mut self, mesh: &iw_mesh::Mesh) -> Result<()> {
        let globe = self.globe.as_mut().expect("globe alive");
        globe.upload_mesh(&self.gpu, mesh)
    }

    /// Replace per-cell elevation (metres) and colour (RGBA8).
    pub fn update_cells(&mut self, elevation_m: &[f32], color_rgba8: &[[u8; 4]]) -> Result<()> {
        self.globe
            .as_mut()
            .expect("globe alive")
            .update_cells(elevation_m, color_rgba8)
    }

    /// Note that the window changed size; the swapchain is rebuilt next frame.
    pub fn request_resize(&mut self, size: (u32, u32)) {
        self.surface_size = size;
        self.needs_resize = true;
    }

    /// Block until the GPU is idle.
    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.gpu.device.device_wait_idle();
        }
    }

    fn recreate_swapchain(&mut self) -> Result<bool> {
        unsafe { self.gpu.device.device_wait_idle() }?;
        let old = self.swapchain.as_ref().map(|s| s.handle);
        let new = match Swapchain::new(
            &self.gpu,
            self.render_pass,
            self.surface_size.0,
            self.surface_size.1,
            old,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Zero-sized (minimised) surfaces are normal; try again later.
                log::debug!("swapchain not ready: {e}");
                return Ok(false);
            }
        };
        if let Some(mut old) = self.swapchain.take() {
            old.destroy(&self.gpu);
        }
        self.swapchain = Some(new);
        self.needs_resize = false;
        Ok(true)
    }

    /// Draw one frame. Returns `Ok(false)` when the frame was skipped because
    /// the surface is not currently presentable (e.g. minimised).
    pub fn render(&mut self, params: &GlobeParams, ui: EguiFrame<'_>) -> Result<bool> {
        // Texture deltas must be consumed even if the frame is skipped, so
        // handle them first. They are rare (font atlas changes), so a full
        // idle wait here is cheaper than tracking per-texture hazards.
        if !ui.textures_delta.set.is_empty() {
            unsafe { self.gpu.device.device_wait_idle()? };
            if let Some(egui) = self.egui.as_mut() {
                for (id, deltas) in &ui.textures_delta.set {
                    for delta in deltas {
                        egui.set_texture(self.gpu.queue, self.gpu.command_pool, *id, delta)
                            .map_err(|e| anyhow::anyhow!("egui texture upload: {e}"))?;
                    }
                }
            }
            ui.textures_delta.set.clear();
        }
        if !ui.textures_delta.free.is_empty() {
            self.pending_texture_frees.push((
                self.frame_counter,
                ui.textures_delta.free.iter().copied().collect(),
            ));
            ui.textures_delta.free.clear();
        }
        let counter = self.frame_counter;
        if let Some(egui) = self.egui.as_mut() {
            self.pending_texture_frees.retain(|(when, ids)| {
                if counter.saturating_sub(*when) < FRAMES_IN_FLIGHT as u64 {
                    return true;
                }
                for id in ids {
                    if let Err(e) = egui.free_texture(*id) {
                        log::warn!("egui texture free: {e}");
                    }
                }
                false
            });
        }

        if (self.needs_resize || self.swapchain.is_none()) && !self.recreate_swapchain()? {
            return Ok(false);
        }

        let frame = self.frame;
        let fence = self.frames[frame].in_flight;
        unsafe {
            self.gpu.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        let (image_index, suboptimal) = {
            let sc = self.swapchain.as_ref().expect("swapchain");
            match unsafe {
                self.gpu.swapchain_fn.acquire_next_image(
                    sc.handle,
                    u64::MAX,
                    self.frames[frame].image_available,
                    vk::Fence::null(),
                )
            } {
                Ok(v) => v,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.needs_resize = true;
                    return Ok(false);
                }
                Err(e) => return Err(e).context("vkAcquireNextImageKHR"),
            }
        };

        unsafe { self.gpu.device.reset_fences(&[fence])? };

        let cb = self.frames[frame].command_buffer;
        let (extent, framebuffer, render_finished) = {
            let sc = self.swapchain.as_ref().expect("swapchain");
            (
                sc.extent,
                sc.framebuffers[image_index as usize],
                sc.render_finished[image_index as usize],
            )
        };

        unsafe {
            self.gpu
                .device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            self.gpu.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        if let Some(globe) = self.globe.as_mut() {
            globe.record_uploads(&self.gpu, cb, frame)?;
        }

        // Reverse-Z: depth clears to 0.0 (the far plane).
        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];
        unsafe {
            self.gpu.device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    })
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
            self.gpu.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.gpu.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
        }

        if let Some(globe) = self.globe.as_mut() {
            globe.record_starfield(&self.gpu, cb, params);
            globe.record_globe(&self.gpu, cb, frame, params);
        }

        if let Some(egui) = self.egui.as_mut() {
            egui.cmd_draw(cb, extent, ui.pixels_per_point, ui.primitives)
                .map_err(|e| anyhow::anyhow!("egui draw: {e}"))?;
        }

        unsafe {
            self.gpu.device.cmd_end_render_pass(cb);
            self.gpu.device.end_command_buffer(cb)?;

            let wait = [self.frames[frame].image_available];
            let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal = [render_finished];
            let cbs = [cb];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&cbs)
                .signal_semaphores(&signal);
            self.gpu
                .device
                .queue_submit(self.gpu.queue, &[submit], fence)?;

            let swapchains = [self.swapchain.as_ref().expect("swapchain").handle];
            let indices = [image_index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self
                .gpu
                .swapchain_fn
                .queue_present(self.gpu.queue, &present)
            {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_resize = true,
                Err(e) => return Err(e).context("vkQueuePresentKHR"),
            }
        }
        if suboptimal {
            self.needs_resize = true;
        }

        self.frame_counter += 1;
        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        Ok(true)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.gpu.device.device_wait_idle();
        }
        // Order matters: everything holding device memory must go before the
        // allocator, and the allocator before the device (see `Gpu::drop`).
        self.egui = None;
        if let Some(mut globe) = self.globe.take() {
            globe.destroy(&self.gpu);
        }
        if let Some(mut sc) = self.swapchain.take() {
            sc.destroy(&self.gpu);
        }
        unsafe {
            for f in self.frames.drain(..) {
                self.gpu.device.destroy_semaphore(f.image_available, None);
                self.gpu.device.destroy_fence(f.in_flight, None);
            }
            self.gpu.device.destroy_render_pass(self.render_pass, None);
        }
        self.gpu.release_allocator();
    }
}
