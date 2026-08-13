//! Vulkan instance, surface, device and allocator bring-up.
//!
//! Nothing here is platform specific beyond what `ash-window` resolves from the
//! winit raw handles, so the same path serves Wayland, X11, Win32 and Metal
//! (via MoltenVK, which is why the portability enumeration flag is set when the
//! extension is present).

use std::ffi::{c_char, CStr, CString};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::AllocatorDebugSettings;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Everything that lives for the whole run of the renderer.
pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_fn: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device_name: String,
    pub queue_family: u32,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub swapchain_fn: khr::swapchain::Device,
    /// `Option` only so it can be dropped before the device in `Drop`.
    allocator: Option<Arc<Mutex<Allocator>>>,
    pub command_pool: vk::CommandPool,
    /// Depth format chosen for the reverse-Z buffer.
    pub depth_format: vk::Format,
    /// True when the validation layer was actually enumerated and enabled.
    pub validation_enabled: bool,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe {
        let d = &*data;
        if d.p_message.is_null() {
            "<no message>".to_string()
        } else {
            CStr::from_ptr(d.p_message).to_string_lossy().into_owned()
        }
    };
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    match severity {
        s if s.contains(S::ERROR) => log::error!("vulkan[{kind:?}] {msg}"),
        s if s.contains(S::WARNING) => log::warn!("vulkan[{kind:?}] {msg}"),
        s if s.contains(S::INFO) => log::debug!("vulkan[{kind:?}] {msg}"),
        _ => log::trace!("vulkan[{kind:?}] {msg}"),
    }
    vk::FALSE
}

impl Gpu {
    /// Create the instance, surface and device for `window`.
    pub fn new<W>(window: &W, app_name: &str) -> Result<Gpu>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan loader")?;

        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        let mut extensions: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        let available_exts = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let has_ext = |name: &CStr| {
            available_exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };

        let mut flags = vk::InstanceCreateFlags::empty();
        if has_ext(khr::portability_enumeration::NAME) {
            extensions.push(khr::portability_enumeration::NAME.as_ptr());
            extensions.push(khr::get_physical_device_properties2::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let want_debug_utils = has_ext(ext::debug_utils::NAME);
        if want_debug_utils {
            extensions.push(ext::debug_utils::NAME.as_ptr());
        }

        // Validation layers are opt-in and optional: enable only if the loader
        // actually enumerates them, never require them.
        let layers_available = unsafe { entry.enumerate_instance_layer_properties() }?;
        let validation_present = layers_available
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        let want_validation = cfg!(debug_assertions)
            || std::env::var("IW_VALIDATION").is_ok_and(|v| v != "0" && !v.is_empty());
        let validation_enabled = validation_present && want_validation;
        let mut layers: Vec<*const c_char> = Vec::new();
        if validation_enabled {
            layers.push(VALIDATION_LAYER.as_ptr());
            log::info!("VK_LAYER_KHRONOS_validation enabled");
        } else if want_validation {
            log::info!("validation layer not installed; continuing without it");
        }

        let app_name_c = CString::new(app_name)?;
        let engine = c"infinite-worlds";
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name_c)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_2);

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers)
            .flags(flags);
        let instance =
            unsafe { entry.create_instance(&instance_ci, None) }.context("vkCreateInstance")?;

        let debug = if want_debug_utils {
            let dbg = ext::debug_utils::Instance::new(&entry, &instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { dbg.create_debug_utils_messenger(&ci, None) }?;
            log::info!("VK_EXT_debug_utils messenger active (errors and warnings)");
            Some((dbg, messenger))
        } else {
            log::warn!("VK_EXT_debug_utils unavailable: driver messages will not be logged");
            None
        };

        let surface_fn = khr::surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .context("creating the window surface")?;

        let (physical_device, queue_family) =
            pick_physical_device(&instance, &surface_fn, surface)?;
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<unknown>".into());
        log::info!(
            "using {device_name} ({:?}), queue family {queue_family}",
            props.device_type
        );

        let dev_exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
        let has_dev_ext = |name: &CStr| {
            dev_exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut device_extensions = vec![khr::swapchain::NAME.as_ptr()];
        // Required by the spec when present (MoltenVK).
        if has_dev_ext(khr::portability_subset::NAME) {
            device_extensions.push(khr::portability_subset::NAME.as_ptr());
        }

        let priorities = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ci)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features)
            .push_next(&mut features12);
        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .context("vkCreateDevice")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let swapchain_fn = khr::swapchain::Device::new(&instance, &device);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| anyhow!("gpu-allocator: {e}"))?;

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;

        let depth_format = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ]
        .into_iter()
        .find(|f| {
            let p = unsafe { instance.get_physical_device_format_properties(physical_device, *f) };
            p.optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .ok_or_else(|| anyhow!("no usable depth format"))?;

        Ok(Gpu {
            entry,
            instance,
            debug,
            surface_fn,
            surface,
            physical_device,
            device_name,
            queue_family,
            device,
            queue,
            swapchain_fn,
            allocator: Some(Arc::new(Mutex::new(allocator))),
            command_pool,
            depth_format,
            validation_enabled,
        })
    }

    /// Shared handle to the memory allocator (the egui backend wants one too).
    pub fn allocator_arc(&self) -> Arc<Mutex<Allocator>> {
        self.allocator.clone().expect("allocator alive")
    }

    /// Lock the memory allocator.
    pub fn alloc(&self) -> std::sync::MutexGuard<'_, Allocator> {
        self.allocator
            .as_ref()
            .expect("allocator alive")
            .lock()
            .expect("allocator mutex")
    }

    /// Release the allocator. Must be called after every allocation is freed
    /// and before the device is destroyed.
    pub fn release_allocator(&mut self) {
        self.allocator = None;
    }

    /// Record and submit a one-shot command buffer, waiting for it to finish.
    pub fn one_shot<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
        unsafe {
            let cb = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            f(cb);
            self.device.end_command_buffer(cb)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            self.device.queue_submit(self.queue, &[submit], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cbs);
        }
        Ok(())
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_fn: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let Some(family) = families.iter().enumerate().position(|(i, f)| {
            f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && unsafe {
                    surface_fn
                        .get_physical_device_surface_support(pd, i as u32, surface)
                        .unwrap_or(false)
                }
        }) else {
            continue;
        };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        if best.is_none_or(|(s, _, _)| score > s) {
            best = Some((score, pd, family as u32));
        }
    }
    best.map(|(_, pd, q)| (pd, q))
        .ok_or_else(|| anyhow!("no Vulkan device with graphics + present support"))
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            // The allocator frees device memory in its own Drop, so it must go
            // before the device does.
            self.allocator = None;
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            if let Some((dbg, messenger)) = &self.debug {
                dbg.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
