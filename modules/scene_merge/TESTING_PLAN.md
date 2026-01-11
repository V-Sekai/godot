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

**Files**: `modules/scene_merge/tests/test_scene_merge.h`

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

    - ✅ Basic mesh unwrapping with default parameters (resolution mode)
    - ✅ Custom texel density settings (texels per unit, >= 0.0)
    - ✅ Resolution and padding parameter validation
    - ✅ Chart generation options (max area, boundary length)
    - ✅ Output mesh validation (UV coordinates, vertex mapping)
    - ✅ Multi-surface mesh handling (ImporterMesh compatibility)
    - ✅ Material preservation across surfaces
    - ✅ ImporterMesh input and output type validation

4. **Error Handling**
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

## Test Execution

### Development Workflow

```bash
# 1. Build editor with debug symbols
godot-build-editor

# 2. Run C++ unit tests
godot-editor --test tests/scene_merge

# 3. Run GDScript integration tests
godot-editor --headless --test res://modules/scene_merge/tests/test_scene_merge_integration.gd

# 4. Editor plugin testing
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

3. **Physics Integration Testing**
    - Collision shape generation from merged meshes
    - Physics performance comparison

## Contributing

When adding new features to SceneMerge:

1. Add corresponding unit tests to `test_scene_merge.h`
2. Update integration tests if API changes
3. Add UV unwrapping tests for new parameters in `unwrap_mesh`
4. Document test data and expected results
5. Update this plan document

## Troubleshooting

-   **Build failures**: Check xAtlas dependency installation
-   **Test failures**: Verify Godot module registration in `SCSub`
-   **Performance issues**: Profile with `--profiler` flag
-   **Memory issues**: Use `--asan` build for debugging

---

_Updated: January 11, 2026 | Version: 4.6 | Authors: V-Sekai Team_
