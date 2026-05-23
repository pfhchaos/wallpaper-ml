"""Vulkan compute wrapper for GPU training.

Provides a minimal but complete interface for dispatching compute shaders
on the Arc A770 via Vulkan 1.4 + Python cffi bindings.

Usage:
    gpu = VkCompute()
    buf = gpu.create_buffer(1024, usage='storage')
    gpu.upload(buf, np.zeros(256, dtype=np.float32))
    pipeline = gpu.create_pipeline('shader.comp', buffers=[buf])
    gpu.dispatch(pipeline, groups_x=4)
    result = gpu.download(buf, dtype=np.float32)
"""
from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import vulkan as vk


class VkBuffer:
    """Wraps a Vulkan buffer + device memory allocation."""
    __slots__ = ('buffer', 'memory', 'size', 'device', '_ctx')

    def __init__(self, buffer, memory, size, device, ctx=None):
        self.buffer = buffer
        self.memory = memory
        self.size = size
        self.device = device
        self._ctx = ctx

    def destroy(self):
        if self.buffer is None:
            return  # already destroyed
        vk.vkDestroyBuffer(self.device, self.buffer, None)
        vk.vkFreeMemory(self.device, self.memory, None)
        if self._ctx is not None:
            self._ctx._live_buffers.discard(id(self))
            self._ctx._buffer_bytes -= self.size
            self._ctx = None
        self.buffer = None
        self.memory = None


class VkPipeline:
    """Wraps a compute pipeline + descriptor set.

    If `_cached=True`, destroy() is a no-op — the pipeline is owned by the
    VkCompute pipeline cache and lives until the compute context is torn
    down. Callers can still call destroy() unconditionally; cached
    pipelines just don't release resources, which avoids the
    create/destroy churn that stresses the Mesa/Arc driver during
    training (millions of pipeline lifecycles → kernel-level GPU hang).
    """
    __slots__ = ('pipeline', 'pipeline_layout', 'descriptor_set',
                 'descriptor_set_layout', 'descriptor_pool', 'device',
                 '_ctx', '_cached')

    def __init__(self, pipeline, pipeline_layout, descriptor_set,
                 descriptor_set_layout, descriptor_pool, device, ctx=None,
                 cached=False):
        self.pipeline = pipeline
        self.pipeline_layout = pipeline_layout
        self.descriptor_set = descriptor_set
        self.descriptor_set_layout = descriptor_set_layout
        self.descriptor_pool = descriptor_pool
        self.device = device
        self._ctx = ctx
        self._cached = cached

    def destroy(self):
        if self.pipeline is None:
            return  # already destroyed
        if self._cached:
            return  # owned by the pipeline cache, freed at context teardown
        vk.vkDestroyPipeline(self.device, self.pipeline, None)
        vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        if self._ctx is not None:
            self._ctx._live_pipelines.discard(id(self))
            self._ctx = None
        self.pipeline = None


class VkCompute:
    """Minimal Vulkan compute context.

    Handles instance, device, queue, command buffer lifecycle.
    Provides buffer management and shader dispatch.
    """

    def __init__(self, device_index: int = 0):
        self._create_instance()
        self._select_device(device_index)
        self._create_device()
        self._create_command_pool()
        self._shader_cache: dict[str, int] = {}  # path → shader module
        # Pipeline cache keyed by (shader_path, buffer-handle tuple, push size).
        # Avoids destroy/recreate churn on Mesa/Arc — training runs millions
        # of dispatches, the driver doesn't reclaim resources fast enough.
        self._pipeline_cache: dict[tuple, VkPipeline] = {}
        # Leak-detection counters (incremented in create_*, decremented in destroy)
        self._live_buffers: set = set()
        self._live_pipelines: set = set()
        self._buffer_bytes: int = 0

    @property
    def buffer_count(self) -> int:
        """Number of buffers that have been created and not yet destroyed."""
        return len(self._live_buffers)

    @property
    def pipeline_count(self) -> int:
        """Number of pipelines that have been created and not yet destroyed."""
        return len(self._live_pipelines)

    @property
    def buffer_bytes(self) -> int:
        """Total bytes across all live buffers."""
        return self._buffer_bytes

    def _create_instance(self):
        app_info = vk.VkApplicationInfo(
            pApplicationName='cnn-trainer',
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName='flame-sheep',
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        self.instance = vk.vkCreateInstance(create_info, None)

    def _select_device(self, index: int):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError('No Vulkan devices found')
        self.physical_device = devices[index]
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        self.device_name = props.deviceName
        self.mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)

        # Find compute queue family
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        self.compute_queue_family = None
        for i, qf in enumerate(queue_families):
            if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.compute_queue_family = i
                break
        if self.compute_queue_family is None:
            raise RuntimeError('No compute queue family found')

    def _create_device(self):
        queue_create = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self.compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_create = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create],
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_create, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0)

    def _create_command_pool(self):
        pool_create = vk.VkCommandPoolCreateInfo(
            queueFamilyIndex=self.compute_queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_create, None)

        # Pre-allocate one command buffer for reuse
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.cmd_buf = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

        # Create fence for synchronization
        fence_create = vk.VkFenceCreateInfo(flags=0)
        self.fence = vk.vkCreateFence(self.device, fence_create, None)

    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        for i in range(self.mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (self.mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError(f'No suitable memory type for filter={type_filter:#x}, props={properties:#x}')

    # -----------------------------------------------------------------
    # Buffer management
    # -----------------------------------------------------------------

    def create_buffer(self, size: int, usage: str = 'storage') -> VkBuffer:
        """Create a host-visible, host-coherent storage buffer.

        For simplicity, all buffers are host-visible so we can upload/download
        without staging buffers. This is slightly slower than device-local
        for pure GPU work, but much simpler and fine for our scale.
        """
        usage_flags = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        if usage == 'uniform':
            usage_flags = vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT

        buf_create = vk.VkBufferCreateInfo(
            size=size,
            usage=usage_flags,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer = vk.vkCreateBuffer(self.device, buf_create, None)

        mem_req = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        mem_type = self._find_memory_type(
            mem_req.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type,
        )
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)

        buf = VkBuffer(buffer, memory, size, self.device, ctx=self)
        self._live_buffers.add(id(buf))
        self._buffer_bytes += size
        return buf

    def upload(self, buf: VkBuffer, data: np.ndarray | bytes):
        """Copy CPU data → GPU buffer."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        assert len(data) <= buf.size, f'Data {len(data)} > buffer {buf.size}'

        ptr = vk.vkMapMemory(self.device, buf.memory, 0, len(data), 0)
        # cffi ffi.memmove for the actual copy
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(ptr, data, len(data))
        vk.vkUnmapMemory(self.device, buf.memory)

    def download(self, buf: VkBuffer, dtype: np.dtype = np.float32,
                 count: int | None = None) -> np.ndarray:
        """Copy GPU buffer → CPU numpy array."""
        size = buf.size if count is None else count * np.dtype(dtype).itemsize
        ptr = vk.vkMapMemory(self.device, buf.memory, 0, size, 0)

        # vkMapMemory returns a cffi buffer object
        raw = bytes(ptr)[:size]
        vk.vkUnmapMemory(self.device, buf.memory)
        arr = np.frombuffer(raw, dtype=dtype)
        if count is not None:
            arr = arr[:count]
        return arr.copy()

    def zero_buffer(self, buf: VkBuffer):
        """Zero out a GPU buffer."""
        self.upload(buf, b'\x00' * buf.size)

    # -----------------------------------------------------------------
    # Shader compilation
    # -----------------------------------------------------------------

    def compile_shader(self, source_path: str | Path) -> int:
        """Compile GLSL compute shader → SPIR-V → Vulkan shader module.

        Returns shader module handle. Caches by path.
        """
        source_path = str(Path(source_path).resolve())
        if source_path in self._shader_cache:
            return self._shader_cache[source_path]

        # Compile GLSL → SPIR-V using glslc
        with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as spv_file:
            spv_path = spv_file.name

        result = subprocess.run(
            ['glslc', '-fshader-stage=compute', source_path, '-o', spv_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f'Shader compilation failed:\n{result.stderr}')

        spv_data = Path(spv_path).read_bytes()
        Path(spv_path).unlink()

        # Create shader module
        module_create = vk.VkShaderModuleCreateInfo(
            codeSize=len(spv_data),
            pCode=spv_data,
        )
        module = vk.vkCreateShaderModule(self.device, module_create, None)
        self._shader_cache[source_path] = module
        return module

    # -----------------------------------------------------------------
    # Pipeline creation
    # -----------------------------------------------------------------

    def create_pipeline(self, shader_path: str | Path,
                        buffers: list[VkBuffer],
                        push_constant_size: int = 0) -> VkPipeline:
        """Create a compute pipeline with descriptor set bound to buffers.

        Each buffer gets binding = its index in the list.
        Optional push constants for per-dispatch parameters.

        Cached by (shader_path, buffer handles, push_constant_size) so
        repeat calls during a training loop return the same pipeline
        instead of recreating Vulkan objects every dispatch. The Mesa/Arc
        driver doesn't reclaim resources fast enough under the churn
        from thousands of training steps and will eventually hang the
        whole GPU.
        """
        cache_key = (
            str(shader_path),
            tuple(buf.buffer for buf in buffers),
            push_constant_size,
        )
        cached = self._pipeline_cache.get(cache_key)
        if cached is not None:
            return cached

        shader_module = self.compile_shader(shader_path)

        # Descriptor set layout — one storage buffer per binding
        bindings = []
        for i in range(len(buffers)):
            bindings.append(vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ))

        ds_layout_create = vk.VkDescriptorSetLayoutCreateInfo(
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        ds_layout = vk.vkCreateDescriptorSetLayout(self.device, ds_layout_create, None)

        # Push constant range (if any)
        push_ranges = []
        if push_constant_size > 0:
            push_ranges.append(vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size,
            ))

        # Pipeline layout
        pl_layout_create = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[ds_layout],
            pushConstantRangeCount=len(push_ranges),
            pPushConstantRanges=push_ranges if push_ranges else None,
        )
        pipeline_layout = vk.vkCreatePipelineLayout(self.device, pl_layout_create, None)

        # Compute pipeline
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName='main',
        )
        pipeline_create = vk.VkComputePipelineCreateInfo(
            stage=stage,
            layout=pipeline_layout,
        )
        pipeline = vk.vkCreateComputePipelines(
            self.device, None, 1, [pipeline_create], None,
        )[0]

        # Descriptor pool + set
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=len(buffers),
        )
        pool_create = vk.VkDescriptorPoolCreateInfo(
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
        )
        descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_create, None)

        ds_alloc = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[ds_layout],
        )
        descriptor_set = vk.vkAllocateDescriptorSets(self.device, ds_alloc)[0]

        # Bind buffers to descriptor set
        writes = []
        for i, buf in enumerate(buffers):
            buf_info = vk.VkDescriptorBufferInfo(
                buffer=buf.buffer,
                offset=0,
                range=buf.size,
            )
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=descriptor_set,
                dstBinding=i,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            ))
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        pipe = VkPipeline(pipeline, pipeline_layout, descriptor_set,
                          ds_layout, descriptor_pool, self.device,
                          ctx=self, cached=True)
        self._live_pipelines.add(id(pipe))
        self._pipeline_cache[cache_key] = pipe
        return pipe

    # -----------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------

    def dispatch(self, pipeline: VkPipeline,
                 groups_x: int, groups_y: int = 1, groups_z: int = 1,
                 push_constants: bytes | None = None):
        """Record and submit a compute dispatch, wait for completion."""
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(self.cmd_buf, begin_info)

        vk.vkCmdBindPipeline(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline.pipeline_layout, 0, 1,
                                   [pipeline.descriptor_set], 0, None)

        if push_constants is not None:
            import cffi as _cffi
            _ffi = _cffi.FFI()
            pc_ptr = _ffi.new('char[]', push_constants)
            vk.vkCmdPushConstants(self.cmd_buf, pipeline.pipeline_layout,
                                 vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 len(push_constants), _ffi.cast('void*', pc_ptr))

        vk.vkCmdDispatch(self.cmd_buf, groups_x, groups_y, groups_z)

        vk.vkEndCommandBuffer(self.cmd_buf)

        submit_info = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[self.cmd_buf],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], self.fence)
        vk.vkWaitForFences(self.device, 1, [self.fence], vk.VK_TRUE, 2**63 - 1)
        vk.vkResetFences(self.device, 1, [self.fence])
        vk.vkResetCommandBuffer(self.cmd_buf, 0)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def destroy(self):
        vk.vkDeviceWaitIdle(self.device)
        # Tear down cached pipelines — they're flagged _cached so their
        # destroy() is a no-op, force the real teardown here.
        for pipe in self._pipeline_cache.values():
            pipe._cached = False
            pipe.destroy()
        self._pipeline_cache.clear()
        for module in self._shader_cache.values():
            vk.vkDestroyShaderModule(self.device, module, None)
        vk.vkDestroyFence(self.device, self.fence, None)
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)

    def __del__(self):
        # Don't auto-destroy — let caller manage lifecycle explicitly.
        # The cffi bindings don't handle teardown order well in __del__.
        pass
