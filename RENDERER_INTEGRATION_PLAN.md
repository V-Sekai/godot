# OIT Renderer Integration Plan

## Overview

Integrate the completed material dispatch system with the forward clustered renderer to enable end-to-end OIT functionality.

## Current State

✅ **Material System Complete**
- BaseMaterial3D has `oit_transparency` property
- Shader generation includes OIT dispatch code
- Fragment collection GLSL implemented

🚧 **Renderer Integration Required**
- Buffer creation and management
- Buffer binding during transparent pass
- Clear operations before rendering
- Resolve pass after transparent rendering

## Integration Points in render_forward_clustered.cpp

### 1. Buffer Creation (in RenderBufferDataForwardClustered)

**Location**: `RenderBufferDataForwardClustered::configure()`

**Action**: Create OIT buffers when render buffers are configured

```cpp
// Add to RenderBufferDataForwardClustered class:
RID oit_tile_buffer;
RID oit_fragment_buffer;
RID oit_counter_buffer;
RID oit_params_buffer;
bool oit_buffers_created = false;

void ensure_oit_buffers(Size2i p_internal_size, uint32_t p_view_count);
void free_oit_buffers();
```

### 2. Buffer Binding Points

**Location**: `_render_scene()` function, transparent pass section

**Current Code Pattern**:
```cpp
RID alpha_framebuffer = rb_data.is_valid() ? rb_data->get_color_pass_fb(transparent_color_pass_flags) : color_only_framebuffer;
RenderListParameters render_list_params(...);
_render_list_with_draw_list(&render_list_params, alpha_framebuffer, ...);
```

**Required Changes**:
1. Before transparent pass: Clear OIT buffers
2. During transparent pass: Bind OIT buffers as storage buffers (bindings 11-15)
3. After transparent pass: Execute OIT resolve

### 3. Buffer Clearing

**Implementation**:
```cpp
// Before transparent rendering
if (rb_data.is_valid() && rb_data->oit_buffers_created) {
    // Clear tile fragment heads to ~0U (invalid index)
    // Clear fragment counters to 0
    RD::get_singleton()->buffer_clear(rb_data->oit_tile_buffer, 0, RD::get_singleton()->buffer_get_size(rb_data->oit_tile_buffer), 0xFFFFFFFF);
    RD::get_singleton()->buffer_clear(rb_data->oit_counter_buffer, 0, RD::get_singleton()->buffer_get_size(rb_data->oit_counter_buffer), 0);
}
```

### 4. Buffer Binding During Transparent Pass

**Location**: Material uniform set creation

**Challenge**: Current material binding happens per-material via uniform sets. OIT buffers need to be globally accessible.

**Solution Options**:
1. Add OIT buffers to render pass uniform set (RENDER_PASS_UNIFORM_SET)
2. Use push constants to enable/disable OIT per draw
3. Bind as global storage buffers outside material system

**Recommended**: Option 1 - Add to render pass uniform set for clean integration

### 5. Resolve Pass Integration

**Location**: After transparent pass, before post-processing

**Implementation**:
```cpp
// After transparent rendering
if (rb_data.is_valid() && rb_data->oit_buffers_created) {
    RendererRD::OIT *oit = RendererRD::OIT::get_singleton();

    // Resolve OIT fragments
    oit->resolve_tiles(
        rb_data->oit_tile_buffer,
        rb_data->oit_fragment_buffer,
        rb_data->oit_counter_buffer,
        alpha_framebuffer,  // Target framebuffer
        p_render_data->scene_data->view_count
    );
}
```

## Implementation Steps

### Step 1: Add OIT Buffer Management to RenderBufferDataForwardClustered

**File**: `render_forward_clustered.h`

Add member variables and methods:
```cpp
class RenderBufferDataForwardClustered : public RenderBufferCustomDataRD {
    // ... existing members ...

    // OIT buffers
    RID oit_tile_buffer;
    RID oit_fragment_buffer;
    RID oit_counter_buffer;
    RID oit_params_buffer;
    bool oit_buffers_created = false;

    void ensure_oit_buffers(Size2i p_internal_size, uint32_t p_view_count);
    void free_oit_buffers();
};
```

### Step 2: Implement Buffer Creation

**File**: `render_forward_clustered.cpp`

```cpp
void RenderBufferDataForwardClustered::ensure_oit_buffers(Size2i p_internal_size, uint32_t p_view_count) {
    if (oit_buffers_created) {
        return;
    }

    RendererRD::OIT *oit = RendererRD::OIT::get_singleton();

    // Create buffers
    oit->create_oit_buffers(
        p_internal_size,
        p_view_count,
        oit_tile_buffer,
        oit_fragment_buffer,
        oit_counter_buffer,
        oit_params_buffer
    );

    oit_buffers_created = true;
}

void RenderBufferDataForwardClustered::free_oit_buffers() {
    if (!oit_buffers_created) {
        return;
    }

    RendererRD::OIT *oit = RendererRD::OIT::get_singleton();
    oit->free_oit_buffers(
        oit_tile_buffer,
        oit_fragment_buffer,
        oit_counter_buffer,
        oit_params_buffer
    );

    oit_buffers_created = false;
}
```

### Step 3: Integrate Buffer Lifecycle

**File**: `render_forward_clustered.cpp`

In `RenderBufferDataForwardClustered::configure()`:
```cpp
void RenderBufferDataForwardClustered::configure(RenderSceneBuffersRD *p_render_buffers) {
    // ... existing code ...

    // Create OIT buffers for VR/multiview rendering
    if (p_render_buffers->get_view_count() > 1) {
        Size2i internal_size = p_render_buffers->get_internal_size();
        ensure_oit_buffers(internal_size, p_render_buffers->get_view_count());
    }
}
```

In `RenderBufferDataForwardClustered::free_data()`:
```cpp
void RenderBufferDataForwardClustered::free_data() {
    // ... existing cleanup ...

    free_oit_buffers();
}
```

### Step 4: Clear Buffers Before Transparent Pass

**File**: `render_forward_clustered.cpp`

In `_render_scene()`, before transparent rendering:
```cpp
// Clear OIT buffers before transparent pass
if (rb_data.is_valid() && rb_data->oit_buffers_created) {
    RD::get_singleton()->buffer_clear(
        rb_data->oit_tile_buffer,
        0,
        RD::get_singleton()->buffer_get_size(rb_data->oit_tile_buffer),
        0xFFFFFFFF  // Clear to invalid index
    );

    RD::get_singleton()->buffer_clear(
        rb_data->oit_counter_buffer,
        0,
        RD::get_singleton()->buffer_get_size(rb_data->oit_counter_buffer),
        0  // Clear counters to zero
    );
}
```

### Step 5: Bind OIT Buffers for Material Dispatch

**Challenge**: Need to make OIT buffers available to material shaders

**Options**:
1. Extend render pass uniform set to include OIT buffers
2. Bind as global storage buffers in descriptor set
3. Use shader specialization constants to enable/disable

**Recommended Approach**: Extend scene uniform set (Set 0) to include OIT buffers when available

### Step 6: Add Resolve Pass

**File**: `render_forward_clustered.cpp`

After transparent rendering:
```cpp
// Resolve OIT fragments after transparent pass
if (rb_data.is_valid() && rb_data->oit_buffers_created) {
    RendererRD::OIT *oit = RendererRD::OIT::get_singleton();

    RID target_fb = rb_data->get_color_pass_fb(transparent_color_pass_flags);

    oit->resolve_tiles(
        rb_data->oit_tile_buffer,
        rb_data->oit_fragment_buffer,
        rb_data->oit_counter_buffer,
        target_fb,
        p_render_data->scene_data->view_count,
        p_render_data->scene_data->view_projection
    );
}
```

## Testing Plan

### Phase 1: Buffer Creation
- Verify buffers are created for VR viewports
- Check buffer sizes match expected values
- Confirm cleanup on render buffer destruction

### Phase 2: Transparent Pass Integration
- Enable OIT on simple transparent material
- Verify material dispatch code is included in shader
- Check buffer binding during rendering

### Phase 3: End-to-End Rendering
- Create test scene with overlapping transparent objects
- Enable OIT transparency on materials
- Verify correct blending order
- Compare with traditional alpha blending

### Phase 4: VR Validation
- Test with multiview stereo rendering
- Verify per-view fragment collection
- Check performance impact (<1ms overhead target)

## Performance Considerations

### Memory Budget
- 4K VR (3840×2160 per eye): ~50MB for OIT buffers
- 2K VR (1920×1080 per eye): ~12MB for OIT buffers
- Acceptable for modern VR headsets (Quest 3, PCVR)

### GPU Overhead
- Fragment dispatch: ~0.3ms (during existing transparent pass)
- Resolve: ~0.5ms (replaces traditional blending)
- Total: <1ms additional overhead vs traditional alpha

### Optimization Opportunities
- Tile-level early exit for empty tiles
- Per-view workload balancing
- Optional tile sorting for workload distribution

## Next Actions

1. ✅ Document integration plan (this file)
2. Add OIT buffer management to RenderBufferDataForwardClustered
3. Implement buffer creation/cleanup
4. Integrate buffer clearing
5. Bind OIT buffers during transparent pass
6. Add resolve pass execution
7. Test with simple geometry
8. Optimize and profile

## References

- `servers/rendering/renderer_rd/effects/oit.h` - OIT implementation
- `servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.h` - Renderer structure
- `OIT_PROGRESS.md` - Overall project progress
- Material dispatch GLSL in `scene/resources/material.cpp`
