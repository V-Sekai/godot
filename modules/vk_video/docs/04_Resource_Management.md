# Resource Management for Vulkan Video

## Brief Description
Comprehensive resource management system for Vulkan Video operations including DPB management, memory pools, command buffers, and synchronization primitives.

## Core Resource Manager

### VulkanVideoResourceManager Class

```cpp
// Main resource manager for Vulkan Video operations
class VulkanVideoResourceManager : public RefCounted {
    GDCLASS(VulkanVideoResourceManager, RefCounted);

private:
    // Core video objects
    RID video_session;
    RID dpb_image_array;

    // Memory pools
    Ref<VideoMemoryPool> dpb_pool;
    Ref<BitstreamBufferPool> bitstream_pool;
    Ref<CommandBufferPool> command_pool;

    // Synchronization
    Vector<RID> frame_semaphores;
    Vector<RID> frame_fences;
    uint32_t current_frame_index = 0;

    // Resource tracking
    HashMap<uint32_t, DPBSlotInfo> dpb_slots;
    Vector<BitstreamBufferInfo> bitstream_buffers;

    // Configuration
    uint32_t max_frames_in_flight = 3;
    uint32_t max_dpb_slots = 8;
    uint32_t max_bitstream_buffers = 16;

protected:
    static void _bind_methods();

public:
    VulkanVideoResourceManager();
    virtual ~VulkanVideoResourceManager();

    // Initialization
    Error initialize(RID p_video_session, RID p_dpb_image_array);
    void cleanup();

    // DPB management
    RID acquire_dpb_slot();
    void release_dpb_slot(RID p_slot);
    VideoPictureResourceInfo get_picture_resource(RID p_slot);
    RID get_output_texture(RID p_slot);

    // Bitstream buffer management
    RID acquire_bitstream_buffer(uint32_t p_size);
    void release_bitstream_buffer(RID p_buffer);

    // Command buffer management
    RDD::CommandBufferID begin_decode_commands();
    void submit_decode_commands(RDD::CommandBufferID p_cmd_buffer);
    void wait_for_decode_completion();

    // Synchronization
    void advance_frame();
    bool is_frame_ready(uint32_t p_frame_index);
};

// DPB slot information
struct DPBSlotInfo {
    RID texture_slice;          // Slice of DPB image array
    uint32_t slot_index;        // Slot index in DPB
    bool in_use = false;        // Currently allocated
    uint64_t frame_number = 0;  // Frame that owns this slot
    double timestamp = 0.0;     // Timestamp of frame
    bool is_reference = false;  // Used as reference frame
};

// Bitstream buffer information
struct BitstreamBufferInfo {
    RID buffer;                 // Vulkan buffer
    uint32_t size;              // Buffer size
    bool in_use = false;        // Currently allocated
    uint64_t last_used_frame;   // Last frame that used this buffer
};
```

## Memory Pool Implementation

### VideoMemoryPool Class

```cpp
// Memory pool for video-specific allocations
class VideoMemoryPool : public RefCounted {
    GDCLASS(VideoMemoryPool, RefCounted);

private:
    struct MemoryBlock {
        RID memory;
        uint32_t size;
        uint32_t offset;
        bool free = true;
        uint64_t last_used_frame = 0;
    };

    Vector<MemoryBlock> memory_blocks;
    uint32_t total_allocated = 0;
    uint32_t total_used = 0;
    uint32_t block_size = 64 * 1024 * 1024; // 64MB blocks

    // Memory type indices
    uint32_t device_local_memory_type = UINT32_MAX;
    uint32_t host_visible_memory_type = UINT32_MAX;

public:
    VideoMemoryPool();
    virtual ~VideoMemoryPool();

    Error initialize(uint32_t p_initial_size = 256 * 1024 * 1024); // 256MB initial
    void cleanup();

    // Memory allocation
    RID allocate_memory(uint32_t p_size, uint32_t p_alignment, bool p_device_local = true);
    void free_memory(RID p_memory);

    // Statistics
    uint32_t get_total_allocated() const { return total_allocated; }
    uint32_t get_total_used() const { return total_used; }
    float get_fragmentation_ratio() const;

    // Garbage collection
    void collect_unused_blocks(uint64_t p_current_frame, uint32_t p_max_age = 60);
};

// Memory allocation implementation
RID VideoMemoryPool::allocate_memory(uint32_t p_size, uint32_t p_alignment, bool p_device_local) {
    // Align size to required alignment
    uint32_t aligned_size = (p_size + p_alignment - 1) & ~(p_alignment - 1);

    // Find suitable free block
    for (int i = 0; i < memory_blocks.size(); i++) {
        MemoryBlock &block = memory_blocks.write[i];
        if (block.free && block.size >= aligned_size) {
            // Split block if necessary
            if (block.size > aligned_size + 1024) { // Leave at least 1KB for next allocation
                MemoryBlock new_block;
                new_block.memory = block.memory;
                new_block.size = block.size - aligned_size;
                new_block.offset = block.offset + aligned_size;
                new_block.free = true;
                memory_blocks.insert(i + 1, new_block);

                block.size = aligned_size;
            }

            block.free = false;
            block.last_used_frame = Engine::get_singleton()->get_process_frames();
            total_used += aligned_size;

            return block.memory;
        }
    }

    // No suitable block found, allocate new one
    return _allocate_new_block(aligned_size, p_device_local);
}
```

### BitstreamBufferPool Class

```cpp
// Pool for bitstream buffers with size-based allocation
class BitstreamBufferPool : public RefCounted {
    GDCLASS(BitstreamBufferPool, RefCounted);

private:
    struct BufferBucket {
        uint32_t size_class;        // Power of 2 size class
        Vector<RID> free_buffers;   // Available buffers
        Vector<RID> used_buffers;   // Currently allocated buffers
    };

    Vector<BufferBucket> size_buckets;
    uint32_t min_buffer_size = 64 * 1024;      // 64KB
    uint32_t max_buffer_size = 16 * 1024 * 1024; // 16MB
    uint32_t max_buffers_per_bucket = 8;

public:
    BitstreamBufferPool();
    virtual ~BitstreamBufferPool();

    Error initialize();
    void cleanup();

    // Buffer allocation
    RID acquire_buffer(uint32_t p_required_size);
    void release_buffer(RID p_buffer);

    // Pool management
    void trim_unused_buffers();
    uint32_t get_total_buffer_count() const;
    uint32_t get_used_buffer_count() const;

private:
    uint32_t _get_size_class(uint32_t p_size);
    RID _create_buffer(uint32_t p_size);
};

// Buffer acquisition with size-based pooling
RID BitstreamBufferPool::acquire_buffer(uint32_t p_required_size) {
    uint32_t size_class = _get_size_class(p_required_size);
    uint32_t actual_size = 1 << size_class;

    // Find or create bucket for this size class
    BufferBucket *bucket = nullptr;
    for (int i = 0; i < size_buckets.size(); i++) {
        if (size_buckets[i].size_class == size_class) {
            bucket = &size_buckets.write[i];
            break;
        }
    }

    if (!bucket) {
        // Create new bucket
        BufferBucket new_bucket;
        new_bucket.size_class = size_class;
        size_buckets.push_back(new_bucket);
        bucket = &size_buckets.write[size_buckets.size() - 1];
    }

    // Get buffer from bucket or create new one
    RID buffer;
    if (!bucket->free_buffers.is_empty()) {
        buffer = bucket->free_buffers[bucket->free_buffers.size() - 1];
        bucket->free_buffers.remove_at(bucket->free_buffers.size() - 1);
    } else {
        buffer = _create_buffer(actual_size);
    }

    bucket->used_buffers.push_back(buffer);
    return buffer;
}
```

## Command Buffer Management

### VideoCommandManager Class

```cpp
// Command buffer management for video operations
class VideoCommandManager : public RefCounted {
    GDCLASS(VideoCommandManager, RefCounted);

private:
    struct CommandBufferInfo {
        RDD::CommandBufferID cmd_buffer;
        RDD::CommandPoolID cmd_pool;
        bool recording = false;
        bool submitted = false;
        uint64_t submission_frame = 0;
        RID fence;
    };

    Vector<CommandBufferInfo> command_buffers;
    uint32_t current_buffer_index = 0;
    uint32_t max_command_buffers = 8;

    // Queue family indices
    uint32_t video_queue_family = UINT32_MAX;
    uint32_t graphics_queue_family = UINT32_MAX;

public:
    VideoCommandManager();
    virtual ~VideoCommandManager();

    Error initialize(uint32_t p_video_queue_family, uint32_t p_graphics_queue_family);
    void cleanup();

    // Command buffer lifecycle
    RDD::CommandBufferID begin_recording();
    void end_recording(RDD::CommandBufferID p_cmd_buffer);
    void submit_commands(RDD::CommandBufferID p_cmd_buffer,
                        const Vector<RID> &p_wait_semaphores = Vector<RID>(),
                        const Vector<RID> &p_signal_semaphores = Vector<RID>());

    // Synchronization
    bool is_command_buffer_complete(RDD::CommandBufferID p_cmd_buffer);
    void wait_for_completion(RDD::CommandBufferID p_cmd_buffer);
    void wait_for_all_commands();

    // Resource management
    void reset_command_buffer(RDD::CommandBufferID p_cmd_buffer);
    void advance_frame();

private:
    CommandBufferInfo* _find_command_buffer_info(RDD::CommandBufferID p_cmd_buffer);
    RDD::CommandBufferID _allocate_command_buffer();
};

// Command buffer recording
RDD::CommandBufferID VideoCommandManager::begin_recording() {
    // Find available command buffer
    CommandBufferInfo *info = nullptr;
    for (int i = 0; i < command_buffers.size(); i++) {
        CommandBufferInfo &cmd_info = command_buffers.write[i];
        if (!cmd_info.recording && !cmd_info.submitted) {
            info = &cmd_info;
            break;
        }

        // Check if submitted command buffer is complete
        if (cmd_info.submitted && is_command_buffer_complete(cmd_info.cmd_buffer)) {
            reset_command_buffer(cmd_info.cmd_buffer);
            info = &cmd_info;
            break;
        }
    }

    // Allocate new command buffer if needed
    if (!info && command_buffers.size() < max_command_buffers) {
        RDD::CommandBufferID new_cmd_buffer = _allocate_command_buffer();
        if (new_cmd_buffer != RDD::CommandBufferID()) {
            CommandBufferInfo new_info;
            new_info.cmd_buffer = new_cmd_buffer;
            command_buffers.push_back(new_info);
            info = &command_buffers.write[command_buffers.size() - 1];
        }
    }

    ERR_FAIL_NULL_V(info, RDD::CommandBufferID());

    // Begin recording
    RenderingDevice *rd = RenderingDevice::get_singleton();
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult result = vkBeginCommandBuffer(info->cmd_buffer, &begin_info);
    ERR_FAIL_COND_V(result != VK_SUCCESS, RDD::CommandBufferID());

    info->recording = true;
    info->submitted = false;

    return info->cmd_buffer;
}
```

## Synchronization Management

### FrameSynchronizer Class

```cpp
// Frame-based synchronization for video operations
class FrameSynchronizer : public RefCounted {
    GDCLASS(FrameSynchronizer, RefCounted);

private:
    struct FrameSync {
        RID semaphore;              // Frame completion semaphore
        RID fence;                  // Frame completion fence
        uint64_t frame_number = 0;  // Frame number
        bool signaled = false;      // Fence signaled
        Vector<RID> dependencies;   // Resources that depend on this frame
    };

    Vector<FrameSync> frame_syncs;
    uint32_t current_frame = 0;
    uint32_t max_frames_in_flight = 3;

    // Timeline semaphore for advanced synchronization
    RID timeline_semaphore;
    uint64_t timeline_value = 0;

public:
    FrameSynchronizer();
    virtual ~FrameSynchronizer();

    Error initialize(uint32_t p_max_frames_in_flight = 3);
    void cleanup();

    // Frame synchronization
    void begin_frame();
    void end_frame();
    RID get_current_frame_semaphore();
    RID get_current_frame_fence();

    // Dependency tracking
    void add_frame_dependency(RID p_resource);
    void wait_for_frame_dependencies();

    // Timeline synchronization
    uint64_t get_timeline_value() const { return timeline_value; }
    void signal_timeline(uint64_t p_value);
    void wait_for_timeline(uint64_t p_value);

    // Utility
    bool is_frame_complete(uint32_t p_frame_index);
    void wait_for_frame(uint32_t p_frame_index);
    void wait_for_all_frames();

private:
    void _advance_frame_index();
    FrameSync* _get_frame_sync(uint32_t p_frame_index);
};

// Frame synchronization implementation
void FrameSynchronizer::begin_frame() {
    FrameSync *frame_sync = _get_frame_sync(current_frame);
    ERR_FAIL_NULL(frame_sync);

    // Wait for frame to be available (if it was used before)
    if (frame_sync->signaled) {
        RenderingDevice *rd = RenderingDevice::get_singleton();
        rd->fence_wait(frame_sync->fence);
        rd->fence_reset(frame_sync->fence);
        frame_sync->signaled = false;
    }

    // Clear dependencies from previous use
    frame_sync->dependencies.clear();
    frame_sync->frame_number = Engine::get_singleton()->get_process_frames();
}

void FrameSynchronizer::end_frame() {
    FrameSync *frame_sync = _get_frame_sync(current_frame);
    ERR_FAIL_NULL(frame_sync);

    // Signal frame completion
    frame_sync->signaled = true;

    // Advance to next frame
    _advance_frame_index();

    // Update timeline
    timeline_value++;
    signal_timeline(timeline_value);
}
```

## Resource Cleanup and Garbage Collection

### ResourceGarbageCollector Class

```cpp
// Garbage collector for video resources
class ResourceGarbageCollector : public RefCounted {
    GDCLASS(ResourceGarbageCollector, RefCounted);

private:
    struct PendingResource {
        RID resource;
        uint64_t frame_released;
        ResourceType type;
    };

    Vector<PendingResource> pending_resources;
    uint32_t max_frame_age = 60; // Keep resources for 60 frames

public:
    enum ResourceType {
        RESOURCE_TEXTURE,
        RESOURCE_BUFFER,
        RESOURCE_COMMAND_BUFFER,
        RESOURCE_SEMAPHORE,
        RESOURCE_FENCE
    };

    ResourceGarbageCollector();
    virtual ~ResourceGarbageCollector();

    // Resource management
    void schedule_cleanup(RID p_resource, ResourceType p_type);
    void collect_garbage(uint64_t p_current_frame);
    void force_cleanup_all();

    // Configuration
    void set_max_frame_age(uint32_t p_age) { max_frame_age = p_age; }
    uint32_t get_max_frame_age() const { return max_frame_age; }

private:
    void _cleanup_resource(const PendingResource &p_resource);
};

// Garbage collection implementation
void ResourceGarbageCollector::collect_garbage(uint64_t p_current_frame) {
    Vector<PendingResource> remaining_resources;

    for (const PendingResource &resource : pending_resources) {
        if (p_current_frame - resource.frame_released >= max_frame_age) {
            _cleanup_resource(resource);
        } else {
            remaining_resources.push_back(resource);
        }
    }

    pending_resources = remaining_resources;
}

void ResourceGarbageCollector::_cleanup_resource(const PendingResource &p_resource) {
    RenderingDevice *rd = RenderingDevice::get_singleton();
    ERR_FAIL_NULL(rd);

    switch (p_resource.type) {
        case RESOURCE_TEXTURE:
            rd->free(p_resource.resource);
            break;
        case RESOURCE_BUFFER:
            rd->free(p_resource.resource);
            break;
        case RESOURCE_COMMAND_BUFFER:
            // Command buffers are freed with their pools
            break;
        case RESOURCE_SEMAPHORE:
        case RESOURCE_FENCE:
            rd->free(p_resource.resource);
            break;
    }
}
```

## Usage Examples

### Basic Resource Management
```cpp
// Initialize resource manager
Ref<VulkanVideoResourceManager> resource_manager = memnew(VulkanVideoResourceManager);
Error err = resource_manager->initialize(video_session, dpb_image_array);
ERR_FAIL_COND(err != OK);

// Acquire resources for decode
RID dpb_slot = resource_manager->acquire_dpb_slot();
RID bitstream_buffer = resource_manager->acquire_bitstream_buffer(frame_size);

// Use resources for decode operation
RDD::CommandBufferID cmd_buffer = resource_manager->begin_decode_commands();
// ... record decode commands ...
resource_manager->submit_decode_commands(cmd_buffer);

// Release resources when done
resource_manager->release_dpb_slot(dpb_slot);
resource_manager->release_bitstream_buffer(bitstream_buffer);
```

### Advanced Memory Management
```cpp
// Configure memory pools
Ref<VideoMemoryPool> memory_pool = memnew(VideoMemoryPool);
memory_pool->initialize(512 * 1024 * 1024); // 512MB initial allocation

Ref<BitstreamBufferPool> bitstream_pool = memnew(BitstreamBufferPool);
bitstream_pool->initialize();

// Monitor memory usage
print("Memory usage: ", memory_pool->get_total_used(), "/", memory_pool->get_total_allocated());
print("Fragmentation: ", memory_pool->get_fragmentation_ratio() * 100.0, "%");

// Periodic cleanup
memory_pool->collect_unused_blocks(current_frame);
bitstream_pool->trim_unused_buffers();
```

This resource management system provides efficient, thread-safe handling of all Vulkan Video resources with automatic cleanup and memory optimization.
