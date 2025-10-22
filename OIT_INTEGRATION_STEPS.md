# OIT Integration Steps for QA Testing (Mobile Renderer Only)

## Current Status
- ✅ All C++ infrastructure complete
- ✅ All shaders implemented
- ❌ Not integrated into mobile renderer pipeline (cannot test yet)

## Required Integration in render_forward_mobile.cpp

### Step 1: Buffer Creation (in _allocate_buffers or similar)
```cpp
// Already exists in render_forward_mobile.h:
// RID oit_tile_buffer;
// RID oit_fragment_buffer;
// RID oit_counter_buffer;

// Create OIT buffers when creating other render buffers
OIT::get_singleton()->create_oit_buffers(
    oit_tile_buffer,
    oit_fragment_buffer,
    oit_counter_buffer,
    rb_size.x, rb_size.y,
    rb->view_count > 1, // multiview
    rb->view_count
);
```

### Step 2: Clear Buffers Before Transparent Pass
```cpp
// In render_scene() before transparent rendering:
if (oit_tile_buffer.is_valid()) {
    OIT::get_singleton()->clear_oit_buffers(
        oit_tile_buffer,
        oit_fragment_buffer,
        oit_counter_buffer,
        rb_size.x, rb_size.y,
        rb->view_count
    );
}
```

### Step 3: Bind OIT Buffers During Transparent Pass
```cpp
// When setting up uniforms for transparent materials:
Vector<RD::Uniform> oit_uniforms;

// Binding 11: Tile buffer
RD::Uniform u_tile;
u_tile.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
u_tile.binding = 11;
u_tile.append_id(oit_tile_buffer);
oit_uniforms.push_back(u_tile);

// Binding 12: Fragment buffer
RD::Uniform u_frag;
u_frag.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
u_frag.binding = 12;
u_frag.append_id(oit_fragment_buffer);
oit_uniforms.push_back(u_frag);

// Binding 13: Counter buffer
RD::Uniform u_counter;
u_counter.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
u_counter.binding = 13;
u_counter.append_id(oit_counter_buffer);
oit_uniforms.push_back(u_counter);

// Binding 14: Params UBO (create params buffer with tile info)
// Binding 15: Additional buffers if needed

// Bind uniform set to draw list
```

### Step 4: Resolve After Transparent Pass
```cpp
// After transparent rendering completes:
if (oit_tile_buffer.is_valid()) {
    OIT::get_singleton()->resolve_oit(
        oit_tile_buffer,
        oit_fragment_buffer,
        color_buffer, // destination framebuffer texture
        depth_buffer, // optional depth texture
        rb_size.x, rb_size.y,
        0 // view index (0 for mono, loop for stereo)
    );
}
```

### Step 5: Buffer Cleanup (in _free_buffers or destructor)
```cpp
if (oit_tile_buffer.is_valid()) {
    OIT::get_singleton()->free_oit_buffers(
        oit_tile_buffer,
        oit_fragment_buffer,
        oit_counter_buffer
    );
}
```

## QA Testing Plan

### Test 1: Basic Transparent Object
1. Create a 3D scene with opaque ground plane
2. Add transparent cube with BaseMaterial3D
3. Enable "OIT Transparency" checkbox in material
4. Render scene - should see transparent cube with OIT

### Test 2: Overlapping Transparency
1. Create multiple overlapping transparent planes at different depths
2. Enable OIT on all materials
3. Verify correct depth sorting (near planes occlude far planes properly)

### Test 3: Performance Test
1. Create scene with 10-20 transparent objects
2. Enable profiling
3. Verify OIT overhead is <1ms as designed

### Test 4: VR Multiview (if XR enabled)
1. Enable VR mode
2. Render transparent objects
3. Verify correct stereo rendering with OIT

## Expected Results When Working

### Visual
- Transparent objects render correctly sorted by depth
- No popping artifacts
- Smooth blending regardless of render order

### Performance
- Collection: ~0.3ms
- Resolve: ~0.5ms
- Total OIT overhead: <1ms @ 2K resolution

### Console Output
```
OIT: Created buffers for X tiles, Y max fragments (views: Z, multiview: true/false)
OIT: Cleared buffers for X tile entries
OIT: Resolve complete for view 0, AxB tiles
```

## Current Blocker

**Cannot QA test until mobile renderer integration is complete.**

The OIT system is fully implemented but not connected to the mobile rendering pipeline.
Once integrated (Steps 1-5 above), QA testing can begin immediately.

*Note: This integration is currently scoped to the mobile renderer only. Clustered renderer integration can be addressed separately.*
