//! Thin buffer wrapper over `gpu-allocator`. Explicit `destroy`, no Drop magic:
//! freeing needs the device and the allocator, which a Drop impl can't reach.

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

use crate::gpu::Gpu;

/// A Vulkan buffer plus its allocation.
pub struct Buffer {
    pub handle: vk::Buffer,
    allocation: Option<Allocation>,
    pub size: u64,
}

impl Buffer {
    /// Allocate `size` bytes with the given usage and memory location.
    pub fn new(
        gpu: &Gpu,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Buffer> {
        let size = size.max(4);
        let handle = unsafe {
            gpu.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let requirements = unsafe { gpu.device.get_buffer_memory_requirements(handle) };
        let allocation = gpu
            .alloc()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| anyhow!("allocating {name} ({size} bytes): {e}"))?;
        unsafe {
            gpu.device
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
        }?;
        Ok(Buffer {
            handle,
            allocation: Some(allocation),
            size,
        })
    }

    /// Copy `data` into a host-visible buffer.
    pub fn write<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes as u64 > self.size {
            return Err(anyhow!(
                "write of {bytes} bytes into a {} byte buffer",
                self.size
            ));
        }
        let ptr = self
            .allocation
            .as_mut()
            .and_then(|a| a.mapped_ptr())
            .ok_or_else(|| anyhow!("buffer is not host visible"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr.as_ptr() as *mut u8,
                bytes,
            )
        };
        Ok(())
    }

    /// Free the buffer and its memory.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(a) = self.allocation.take() {
            let _ = allocator.free(a);
        }
        if self.handle != vk::Buffer::null() {
            unsafe { device.destroy_buffer(self.handle, None) };
            self.handle = vk::Buffer::null();
        }
    }
}

/// Create a device-local buffer holding `data`, staged through a temporary
/// host-visible buffer. Used for the static geometry, uploaded once.
pub fn upload_device_local<T: Copy>(
    gpu: &Gpu,
    data: &[T],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<Buffer> {
    let bytes = std::mem::size_of_val(data) as u64;
    let mut staging = Buffer::new(
        gpu,
        bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        "staging",
    )?;
    staging.write(data)?;
    let dst = Buffer::new(
        gpu,
        bytes,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        name,
    )?;
    gpu.one_shot(|cb| unsafe {
        gpu.device.cmd_copy_buffer(
            cb,
            staging.handle,
            dst.handle,
            &[vk::BufferCopy::default().size(bytes)],
        );
    })?;
    staging.destroy(&gpu.device, &mut gpu.alloc());
    Ok(dst)
}
