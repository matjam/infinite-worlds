//! FIFO swapchain plus the reverse-Z depth buffer and framebuffers.
//!
//! Recreated wholesale on resize; the render pass outlives it because the
//! formats never change.

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::gpu::Gpu;

/// Swapchain images, depth attachment and framebuffers for one size.
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    depth_image: vk::Image,
    depth_alloc: Option<Allocation>,
    pub depth_view: vk::ImageView,
    /// One per swapchain image, so a semaphore is never reused while the
    /// presentation engine may still be waiting on it.
    pub render_finished: Vec<vk::Semaphore>,
}

/// Pick a surface format. A UNORM target is preferred: all project colours are
/// already sRGB-encoded bytes, so no hardware conversion is wanted.
fn pick_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    const PREFERRED: [vk::Format; 2] = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM];
    for want in PREFERRED {
        if let Some(f) = formats
            .iter()
            .find(|f| f.format == want && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        {
            return *f;
        }
    }
    formats[0]
}

impl Swapchain {
    /// Create (or recreate) the swapchain at the window's current size.
    pub fn new(
        gpu: &Gpu,
        render_pass: vk::RenderPass,
        width: u32,
        height: u32,
        old: Option<vk::SwapchainKHR>,
    ) -> Result<Swapchain> {
        let caps = unsafe {
            gpu.surface_fn
                .get_physical_device_surface_capabilities(gpu.physical_device, gpu.surface)
        }?;
        let formats = unsafe {
            gpu.surface_fn
                .get_physical_device_surface_formats(gpu.physical_device, gpu.surface)
        }?;
        if formats.is_empty() {
            return Err(anyhow!("surface reports no formats"));
        }
        let surface_format = pick_format(&formats);

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(anyhow!("zero-sized surface"));
        }

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(gpu.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            // FIFO is always supported and is what the design asks for.
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old.unwrap_or(vk::SwapchainKHR::null()));
        let handle = unsafe { gpu.swapchain_fn.create_swapchain(&ci, None) }?;
        let images = unsafe { gpu.swapchain_fn.get_swapchain_images(handle) }?;

        let views = images
            .iter()
            .map(|img| unsafe {
                gpu.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(*img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Depth attachment.
        let depth_image = unsafe {
            gpu.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(gpu.depth_format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }?;
        let reqs = unsafe { gpu.device.get_image_memory_requirements(depth_image) };
        let depth_alloc = gpu
            .alloc()
            .allocate(&AllocationCreateDesc {
                name: "depth",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::DedicatedImage(depth_image),
            })
            .map_err(|e| anyhow!("depth allocation: {e}"))?;
        unsafe {
            gpu.device
                .bind_image_memory(depth_image, depth_alloc.memory(), depth_alloc.offset())
        }?;
        let depth_view = unsafe {
            gpu.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(depth_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(gpu.depth_format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }?;

        let framebuffers = views
            .iter()
            .map(|v| {
                let attachments = [*v, depth_view];
                unsafe {
                    gpu.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&attachments)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let render_finished = (0..images.len())
            .map(|_| unsafe {
                gpu.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Swapchain {
            handle,
            format: surface_format.format,
            extent,
            images,
            views,
            framebuffers,
            depth_image,
            depth_alloc: Some(depth_alloc),
            depth_view,
            render_finished,
        })
    }

    /// Aspect ratio of the current extent.
    pub fn aspect(&self) -> f32 {
        self.extent.width as f32 / self.extent.height.max(1) as f32
    }

    /// Destroy everything. The caller must have waited for the device to idle.
    pub fn destroy(&mut self, gpu: &Gpu) {
        unsafe {
            for s in self.render_finished.drain(..) {
                gpu.device.destroy_semaphore(s, None);
            }
            for fb in self.framebuffers.drain(..) {
                gpu.device.destroy_framebuffer(fb, None);
            }
            gpu.device.destroy_image_view(self.depth_view, None);
            if let Some(a) = self.depth_alloc.take() {
                let _ = gpu.alloc().free(a);
            }
            gpu.device.destroy_image(self.depth_image, None);
            for v in self.views.drain(..) {
                gpu.device.destroy_image_view(v, None);
            }
            gpu.swapchain_fn.destroy_swapchain(self.handle, None);
        }
    }
}

/// The single render pass: colour + reverse-Z depth, one subpass.
pub fn create_render_pass(gpu: &Gpu, color_format: vk::Format) -> Result<vk::RenderPass> {
    let attachments = [
        vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        vk::AttachmentDescription::default()
            .format(gpu.depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref)
        .depth_stencil_attachment(&depth_ref)];
    let dependency = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )];
    let ci = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpass)
        .dependencies(&dependency);
    Ok(unsafe { gpu.device.create_render_pass(&ci, None) }?)
}
