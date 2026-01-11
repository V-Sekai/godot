# SceneMerge Testing Plan

## Overview

This document outlines the comprehensive testing plan for the SceneMerge module, a Godot extension that provides mesh merging, texture atlas generation, and scene optimization capabilities.

## Module Architecture

-   **SceneMerge Class**: RefCounted wrapper for mesh merging operations
-   **MeshTextureAtlas**: Core implementation with xAtlas integration
-   **Tools Integration**: Editor plugin for visual workflow

## Build Environment

### Build Alias Configuration

```bash
# Development build with full debugging and testing support
alias godot-build-editor='scons platform=macos arch=arm64 target=editor dev_build=yes debug_symbols=yes compiledb=yes tests=yes generate_bundle=yes cache_path=/Users/ernest.lee/.scons_cache use_asan=no'

# Release build for performance testing
alias godot-build-release='scons platform=macos arch=arm64 target=editor dev_build=no debug_symbols=no compiledb=no tests=no generate_bundle=yes cache_path=/Users/ernest.lee/.scons_cache use_asan=no'

# Clean rebuild (use when switching configurations)
alias godot-clean-build='scons platform=macos arch=arm64 target=editor dev_build=yes debug_symbols=yes compiledb=yes tests=yes generate_bundle=yes cache_path=/Users/ernest.lee/.scons_cache use_asan=no --clean'
```

### Prerequisites

-   Godot 4.6+ with scene_merge module enabled
-   xAtlas library for UV unwrapping
-   Compiler with C++17 support

## Testing Framework

### 1. Unit Tests (C++/doctest)

**Files**:

-   `modules/scene_merge/tests/test_scene_merge.h` (main include file)
-   `modules/scene_merge/tests/test_scene_merge_basic.h` (basic functionality)
-   `modules/scene_merge/tests/test_scene_merge_triangle.h` (triangle operations)
-   `modules/scene_merge/tests/test_scene_merge_atlas.h` (texture atlas operations)
-   `modules/scene_merge/tests/test_scene_merge_unwrap_basic.h` (basic UV unwrapping)
-   `modules/scene_merge/tests/test_scene_merge_unwrap_density.h` (texel density UV unwrapping)
-   `modules/scene_merge/tests/test_scene_merge_unwrap_barycentric.h` (barycentric coordinate testing)
-   `modules/scene_merge/tests/test_scene_merge_unwrap_trim.h` (trim sheet handling)

-   Basic class instantiation tests
-   MeshMergeTriangle geometric operations
-   Texture atlas texel manipulation
-   Memory management validation

### 2. Integration Tests (GDScript SceneTree)

**Files**: `modules/scene_merge/tests/test_scene_merge_integration.gd`

-   Scene extending SceneTree for full engine integration
-   Headless testing environment
-   Performance benchmarking
-   Error handling scenarios

### 3. Editor Plugin Tests

**Files**: `modules/scene_merge/merge_plugin.*`

-   GUI functionality validation
-   Workflow integration testing
-   Export/import compatibility

## Test Categories

### Core Functionality Tests

1. **SceneMerge Instantiation**

    - Verify RefCounted creation
    - Check method bindings
    - Validate runtime accessibility

2. **Mesh Merging Operations**

    - Single mesh processing
    - Multiple mesh batch processing
    - Hierarchical scene traversal
    - Material preservation

3. **Texture Atlas Generation**

    - UV unwrapping with xAtlas
    - Texture packing optimization
    - Mipmap generation
    - Atlas resolution management

4. **UV Unwrapping Tests**

    - ✅ **Basic UV Unwrapping** (`test_scene_merge_unwrap_basic.h`): Basic mesh unwrapping with default parameters
    - ✅ **Texel Density** (`test_scene_merge_unwrap_density.h`): Custom texel density settings and validation
    - ✅ **Barycentric Coordinates** (`test_scene_merge_unwrap_barycentric.h`): Barycentric coordinate interpolation and atlas texel setting accuracy
    - ✅ **Trim Sheet Handling** (`test_scene_merge_unwrap_trim.h`): Documented trim sheet limitations (currently unsupported)
    - ✅ Resolution and padding parameter validation
    - ✅ Chart generation options (max area, boundary length)
    - ✅ Output mesh validation (UV coordinates, vertex mapping)
    - ✅ Multi-surface mesh handling (ImporterMesh compatibility)
    - ✅ Material preservation across surfaces
    - ✅ ImporterMesh input and output type validation

    **⚠️ Critical Limitation**: Trim sheet textures cause mesh merge operations to fail. This significantly impacts production workflows relying on trim sheets.

5. **Error Handling**
    - Empty scene scenarios
    - Invalid mesh data
    - Memory allocation failures
    - Cyclic reference detection

### Performance Tests

1. **Scalability Testing**

    - Large scene processing (1000+ meshes)
    - High-resolution texture atlases
    - Memory usage profiling

2. **Optimization Validation**
    - Draw call reduction metrics
    - Texture memory savings
    - Processing time benchmarks

### Compatibility Tests

1. **Platform Validation**

    - macOS (arm64/x86_64)
    - Linux (x86_64)
    - Windows (x86_64)

2. **Godot Version Compatibility**
    - Godot 4.6 LTS
    - Development branch compatibility

## SceneMerge Feature Testing Checklist

This comprehensive checklist covers all SceneMerge functionality that must be tested for production readiness.

### 🔧 Core Foundation (Test First - Dependencies)

-   [x] **Module Loading** - SceneMerge loads with Godot without conflicts (register_types.cpp)
-   [x] **Module Registration** - SceneMerge class properly registers with Godot runtime (MODULE_INITIALIZATION_LEVEL_SCENE)
-   [x] **Basic Instantiation** - Can create SceneMerge RefCounted instances (SceneMerge::SceneMerge())

### 🏗️ Basic Operations (Essential Functionality)

-   [x] **merge() Method** - Core mesh merging functionality executes without crashes (MeshTextureAtlas::merge_meshes implementation)
-   [x] **Empty Scene Handling** - Gracefully handles scenes with no mesh instances (returns root unchanged with logging)
-   [x] **Single Mesh Handling** - Processes scenes with only one mesh instance (rejects gracefully, returns unchanged + unit test added)
-   [x] **Multiple Mesh Merging** - Correctly combines 2+ mesh instances into single geometry (core functionality implemented)

### 📐 Mesh Processing (Geometric Foundations)

-   [x] **Vertex Data Merging** - All vertex positions properly combined (simplified unit test implemented)
-   [x] **Normal Vector Preservation** - Surface normals accurately maintained (unit test implemented)
-   [x] **Index Buffer Optimization** - Triangle indices efficiently combined (unit test implemented)
-   [x] **Primitive Type Compatibility** - Handles triangles, quads, ngons correctly. Handles all godot primitive types. (unit test implemented)
-   [ ] **Mesh Surface Preservation** - Multiple material surfaces maintained
-   [ ] **Hierarchy Preservation** - Maintains scene node structure after merging

### 🛡️ Error Handling & Input Validation (Critical for Stability)

-   [x] **Null Pointer Handling** - Graceful handling of null inputs (implemented: validate_merge_input() returns early + unit test added)
-   [x] **Invalid Mesh Data** - Corrupted or malformed geometry (unit test implemented)
-   [ ] **Memory Allocation Failures** - Low memory condition handling
-   [ ] **File System Permission Issues** - Read/write access problems
-   [ ] **Texture Loading Failures** - Missing or corrupted textures

### 🎨 Basic Material Support (Rendering Fundamentals)

-   [x] **BaseMaterial3D Compatibility** - SceneMerge works with all BaseMaterial3D-derived materials (unit test implemented)
-   [ ] **Material Reference Preservation** - Original materials accessible after merging
-   [ ] **Material Index Mapping** - Correct surface-to-material assignments
-   [ ] **Albedo Properties** - Color and texture preservation through merge operations

### 🗺️ UV Unwrapping Basics (Essential for Texturing)

-   [x] **Basic UV Unwrapping** - Default parameter unwrapping without errors (SceneMerge::unwrap_mesh implementation)
-   [x] **Chart Generation** - Automatic UV chart creation from mesh geometry (xAtlas integration)
-   [x] **Island Packing** - UV islands packed efficiently without overlap (xAtlas PackOptions)
-   [ ] **UV Coordinate Mapping** - Texture coordinates preserved or remapped as needed

### 📏 Boundary & Precision Testing (Prevent Visual Bugs)

-   [ ] **MeshMergeTriangle Off-by-One Testing** - Pixel coordinate boundaries in triangle rasterization (x,y edge cases)
-   [ ] **xAtlas Off-by-One Errors** - UV coordinate boundary testing at atlas edges (0,0) and (1,1)
-   [ ] **Texel Coordinate Precision** - Ensure no rounding errors in pixel-perfect coordinate mapping

### 🎨 Texture Atlas System (Core Optimization Feature)

-   [ ] **Basic Texture Atlasing** - Single texture atlas creation
-   [x] **Multiple Texture Support** - Different material textures combined (implemented in merge_meshes)
-   [ ] **Multiple Atlases for 4K Texel Sheets** - Automatically create separate atlases when textures exceed 4K resolution (4096x4096)
-   [ ] **Atlas Resolution Control** - Configurable output texture sizes
-   [ ] **Packing Efficiency** - Minimize wasted texture space (>90% utilization)

### ⚡ Performance & Optimization (Quality Assurance)

-   [ ] **Small Scenes** - Fast processing (< 100ms) for simple scenes
-   [ ] **Medium Scenes** - Acceptable performance (1-5s) for game levels
-   [ ] **Draw Call Reduction** - Quantify improvement (10:1, 50:1, etc.)
-   [ ] **Memory Usage Optimization** - Reduce total VRAM footprint

### 🧪 Advanced UV Features (Building on Basics)

-   [x] **Texel Density Control** - Custom texels-per-unit parameter support (float p_texel_density parameter)
-   [x] **Barycentric Coordinate Validation** - UV interpolation accuracy testing (implemented in MeshMergeTriangle)
-   [ ] **Seam Optimization** - Minimize visible seams in final UV layout
-   [ ] **Padding Control** - Adjustable spacing between UV islands

### 🖼️ Advanced Materials (Complete Material Ecosystem)

-   [ ] **PBR Parameters** - Metallic, roughness, specular properties maintained
-   [ ] **Normal Maps** - Normal texture and scale settings preserved
-   [ ] **Emission** - Emission color, energy and texture compatibility
-   [ ] **Transparency** - Alpha blending, scissor threshold, hash mode support
-   [ ] **Rendering Modes** - Cull mode, depth draw, two-sided rendering
-   [ ] **Multi-Surface Materials** - Material arrays on merged meshes functional

### 🔧 Advanced Atlas Features (Optimization)

-   [ ] **Mipmap Generation** - Proper mipmap chains for atlas textures
-   [ ] **Border Artifact Prevention** - Adequate padding to prevent bleeding
-   [ ] **Material Processing** - Shader compatibility, texture coordinate remapping

### 🔗 Integration & Compatibility (External Dependencies)

-   [x] **GDScript API Access** - Scriptable interface functions properly (GDScript bindings added)
-   [ ] **ImporterMesh Compatibility** - Works with imported .gltf/.fbx files
-   [ ] **ArrayMesh Support** - Programmatically created meshes
-   [ ] **MultiMesh Support** - Instancing system compatibility

### 🔍 Edge Cases & Special Scenarios (Coverage Gaps)

-   [ ] **Thin Geometry Handling** - Very thin or elongated mesh features
-   [ ] **High Polygon Count** - Performance with 100K+ polygons
-   [ ] **Degenerate Geometry** - Zero-area triangles, co-linear vertices, invalid normals
-   [ ] **Extreme Scale Objects** - Very large/small objects (scale ratios >1000:1)
-   [ ] **Self-Intersecting Meshes** - Topologically complex models with holes/intersections
-   [ ] **Circular Dependencies** - Scenes with circular node hierarchies
-   [ ] **Mixed Units/Scales** - Importing meshes with different unit conventions
-   [ ] **UTF-8/Unicode Names** - Non-ASCII node names, texture paths, material names
-   [ ] **Memory Exhaustion** - Very large scenes approaching system memory limits
-   [ ] **Texture Format Compatibility** - Various compression formats (DXT, ASTC, ETC)
-   [ ] **Animation Data Preservation** - How merging affects skeleton/transform animations
-   [ ] **LOD Level Compatibility** - Interaction with level-of-detail systems
-   [ ] **Multiple Material Support** - Scenes with different materials per mesh
-   [ ] **Partial Failure Handling** - Continue processing after individual mesh failures

### 🌐 Platform-Specific Validation (Deployment Ready)

-   [ ] **macOS Support** - Native Intel/Apple Silicon compatibility
-   [x] **Cross-Platform Building** - macOS/Linux/Windows build compatibility (SCsub configured)
-   [ ] **Windows Support** - DirectX/OpenGL rendering pipelines
-   [ ] **Linux Support** - X11/Wayland display server compatibility

### 🐛 Trim Sheet Limitations (Known Issues)

-   [x] **Trim Sheet Detection** - Identify when trim sheets are present
-   [x] **Failure Mode Documentation** - Document merge failure with trim sheets
-   [ ] **Trim Sheet Bypass** - Workaround testing for trim sheet scenarios
-   [ ] **Fallback Error Messages** - Clear user feedback on trim sheet issues

---

## Implementation Status Summary

**Total Test Categories:** 5 major areas (Core Merging, UV Unwrapping, Texture Atlas, Performance, Integration)
**Total Test Items:** 87 individual test scenarios
**Completed Items:** ~15 (implementation infrastructure)
**Remaining Items:** ~72 (feature-specific test implementation)

**Priority Order:**

1. **High Priority** (Foundation) - Core merging, basic UV, error handling
2. **Medium Priority** (Optimization) - Performance, scaling, atlas optimization
3. **Low Priority** (Polish) - Edge cases, platform-specific, advanced features

## Test Execution

### Development Workflow

**Prerequisites:**

```bash
# Set the build alias in your shell (run this in your terminal first)
alias godot-build-editor='scons platform=macos arch=arm64 target=editor dev_build=yes debug_symbols=yes compiledb=yes tests=yes generate_bundle=yes cache_path=/Users/ernest.lee/.scons_cache use_asan=no'
```

**Testing Workflow:**

```bash
# 1. Build editor with debug symbols AND run C++ unit tests automatically
./modules/scene_merge/scripts/test_scene_merge_build.sh

# Expected output:
# ✅ Build completed successfully!
# 🧪 Running SceneMerge C++ Unit Tests (doctest framework)...
# 🎉 ALL TESTS PASSED! (1276 test cases, 416K+ assertions)

# 2. Run GDScript integration tests (if C++ tests pass)
godot-editor --headless --test res://modules/scene_merge/tests/test_scene_merge_integration.gd

# 3. Editor plugin testing
godot-editor --test plugins/scene_merge
```

### Continuous Integration

```bash
# Automated testing pipeline
./ci/test_scene_merge.sh

# IMPORTANT: CI/CD doctests with Godot Engine C++ modules must use importer mesh and importer mesh instance data for accurate testing
# Programmatically created meshes may behave differently from imported .gltf/.fbx/.obj assets
# Ensure test assets are imported through Godot's asset pipeline (not generated at runtime)
```

## Test Data Preparation

### Sample Scenes for Mesh Merging Tests

**⚠️ CRITICAL: All test scenes must contain MULTIPLE mesh instances to properly test mesh merging functionality. Single mesh scenes cannot validate merging behavior.**

1. **Basic Multi-Cube Grid (Validation Scene)**

    - 10-20 cube MeshInstance3D objects in a 3D grid
    - All using same base material with texture
    - Expected result: Single merged mesh with texture atlas
    - Validates: Basic merging, UV remapping, material reduction

2. **Complex Architectural Scene (Performance Test)**

    - 50+ architectural meshes (walls, doors, windows, furniture)
    - Multiple different materials and textures
    - Hierarchical scene structure with rooms/buildings
    - Expected result: Optimized scene with merged geometry batches
    - Validates: Large-scale merging, material atlas complexity, hierarchy preservation

3. **Game Environment Scene (Optimization Test)**

    - Terrain chunks, buildings, vegetation instances
    - Environmental objects with varied scale and orientation
    - Performance-critical scene (target 60fps)
    - Expected result: Draw call reduction from hundreds to dozens
    - Validates: Real-world optimization impact, batching efficiency

4. **Edge Case Scenes**

    - **Empty scene**: No meshes (error handling)
    - **Single mesh only**: Cannot test merging (baseline reference)
    - **Identical transform meshes**: Overlapping geometry edge case
    - **Maximum vertex count**: Stress test large mesh merging
    - **Mixed primitive types**: Triangles, quads, ngon compatibility

5. **UV Unwrapping Test Meshes**
    - **Simple Cube**: Basic unwrapping validation
    - **Complex Model**: High-poly mesh with multiple charts
    - **Thin Geometry**: Edge case for chart generation
    - **Large Scale**: Test texel density calculations
    - **Custom Density**: Validate texels per unit parameter

## Validation Metrics

### Success Criteria

-   ✅ All C++ unit tests pass (1276 test cases, 416619 assertions)
-   ✅ Integration tests complete without exceptions
-   ✅ Memory leaks < 1% of allocated memory
-   ✅ Texture atlas quality > 90% packing efficiency
-   ✅ UV unwrapping accuracy > 95% (chart overlap < 5%)
-   ✅ Texel density consistency within 10% of specified value
-   ✅ Performance regression < 5% vs baseline

### Quality Assurance

-   Code coverage > 85% (excluding generated code)
-   Static analysis clean (no critical warnings)
-   Documentation completion for public APIs
-   Example project testing

## Future Enhancements

### Additional Test Coverage

1. **GPU Rendering Validation**

    - Render output comparison before/after merging
    - Visual regression testing

2. **UV Unwrapping Advanced Tests**

    - Chart options parameter testing (weights, iterations)
    - Pack options validation (brute force, block align)
    - Multi-surface mesh handling
    - Normal/tangent preservation in unwrapped meshes

3. **Network/Asset Pipeline Testing**

    - Remote texture loading
    - Asynchronous processing validation

4. **Physics Integration Testing**
    - Collision shape generation from merged meshes
    - Physics performance comparison

## Contributing

When adding new features to SceneMerge:

1. Add corresponding unit tests to the appropriate split test file:
    - `test_scene_merge_basic.h` for basic functionality
    - `test_scene_merge_triangle.h` for triangle/drawing operations
    - `test_scene_merge_atlas.h` for texture atlas operations
    - `test_scene_merge_unwrap_basic.h` for basic UV unwrapping functionality
    - `test_scene_merge_unwrap_density.h` for texel density UV unwrapping
    - `test_scene_merge_unwrap_barycentric.h` for barycentric coordinate testing
    - `test_scene_merge_unwrap_trim.h` for trim sheet handling (document limitations)
2. Update integration tests if API changes
3. Add UV unwrapping tests for new parameters in `unwrap_mesh`
4. Document test data and expected results
5. Update this plan document

## Troubleshooting

-   **Build failures**: Check xAtlas dependency installation
-   **Test failures**: Verify Godot module registration in `SCSub`
-   **Performance issues**: Profile with `--profiler` flag
-   **Memory issues**: Use `--asan` build for debugging
-   **Trim Sheet Issues**: ⚠️ **CRITICAL LIMITATION** - Texture atlas generation with trim sheets causes mesh merge operations to fail. Trim sheets are currently not supported and must be disabled for successful merging. This significantly limits the module's usefulness for production workflows that rely on trim sheet textures.

---

_Updated: January 11, 2026 | Version: 4.6 | Authors: V-Sekai Team_

**Recent Changes:**

-   Split monolithic `test_scene_merge_unwrap.h` into four focused test files for better organization and coverage
-   Added barycentric coordinate testing for UV unwrapping accuracy validation
-   Documented critical trim sheet limitation preventing production workflow usage
