# Sculpted Primitives for CSG

This document provides guidelines for contributing to the sculpted primitives in Godot's CSG (Constructive Solid Geometry) module. These primitives allow for advanced sculpting of 3D shapes through profile and path curve transformations.

## Overview

Sculpted primitives extend Godot's CSG system with parametric shapes that can be deformed using various sculpting parameters:

-   Profile curves (circle, square, triangle shapes)
-   Path curves (line, circle variations)
-   Transformations (twist, taper, shear, radius offset, skew)
-   Hollow shapes with customizable inner geometry

## Ensuring Manifold Geometry

**Critical Requirement**: All sculpted primitives MUST generate manifold meshes. A manifold mesh has the property that every edge is shared by exactly two faces, ensuring valid solid geometry for CSG operations.

### What is Manifold Geometry?

A manifold mesh is defined according to Godot's CSG implementation, adapted from the manifold library:

> Every edge of every triangle must contain the same two vertices (by index) as exactly one other triangle edge, and the start and end vertices must switch places between these two edges. The triangle vertices must appear in clockwise order when viewed from the outside of the Godot Engine manifold mesh.

In practical terms for CSG sculpted primitives:

- **Edge pairing**: Every triangle edge must be shared with exactly one other triangle
- **Vertex ordering**: Triangle vertices must be specified in clockwise order when viewed from outside
- **No boundary edges**: All edges must connect exactly two faces (no holes or gaps)
- **Orientable surface**: The mesh forms a continuous, orientable boundary

Non-manifold geometry breaks CSG boolean operations and can cause rendering artifacts, physics simulation failures, and export issues.

### Validation Checks

The code includes several validation mechanisms:

1. **Debug Output**: Verbose logging shows vertex counts, face counts, and manifold status
2. **Manifold Library Validation**: Uses the manifold library's built-in error checking with descriptive error messages
3. **Profile Closure**: Ensures profile curves are properly closed loops without gaps
4. **Face Winding**: Maintains consistent counter-clockwise winding for outward-facing normals
5. **Edge Connectivity**: Verifies all edges are shared by exactly two faces
6. **Self-Intersection Detection**: Checks for geometry that crosses itself

### Testing Manifoldness

Before committing sculpted primitive changes:

1. **Enable verbose logging** to see detailed mesh statistics
2. **Test with extreme parameters** (zero radius, maximum twist, etc.)
3. **Verify with manifold library** - check that `manifold::Manifold` construction succeeds
4. **Visual inspection** - ensure no holes, gaps, or overlapping faces
5. **CSG operations** - test boolean operations with other shapes

### Common Issues and Solutions

#### 1. Non-Manifold Edges

**Symptom**: Manifold library reports "Not Manifold" error or edge connectivity warnings
**Causes**:

-   Incorrect vertex indexing in face generation
-   Missing or duplicate edges in triangle strips
-   Improper loop closure (gaps between first and last vertices)
-   Inconsistent edge sharing between adjacent faces

**Solutions**:

-   Verify all triangles share edges correctly using consistent vertex ordering
-   Ensure profile loops are closed without duplicate vertices at seam
-   Check face winding order (clockwise from outside)
-   Use edge adjacency checks in debug builds

#### 2. Degenerate Faces

**Symptom**: Zero-area triangles, invalid geometry, or manifold library errors
**Causes**:

-   Collinear or coincident vertices creating zero-area triangles
-   Incorrect parameter combinations causing geometric collapse
-   Floating-point precision issues with near-zero values
-   Division by zero in transformation calculations

**Solutions**:

-   Add epsilon checks (`Math::is_zero_approx()`) for near-zero values
-   Validate parameter ranges with `CLAMP()` and minimum thresholds
-   Use robust geometric computations avoiding exact zero divisions
-   Skip degenerate triangles during mesh generation

#### 3. Path/Profile Mismatches

**Symptom**: Inconsistent vertex counts between path segments or profile discontinuities
**Causes**:

-   Profile point count changes along path extrusion
-   Incorrect handling of closed vs open curves
-   Path segmentation that doesn't align with profile resolution
-   Hollow shape inner/outer profile count mismatches

**Solutions**:

-   Pre-calculate and fix profile sizes before vertex generation
-   Ensure consistent segmentation across the entire path
-   Handle closed loops correctly with proper vertex indexing
-   Validate hollow shape profile compatibility

#### 4. Self-Intersections

**Symptom**: Geometry crosses itself, causing invalid topology
**Causes**:

-   Extreme twist or shear parameters
-   Profile curves that intersect when extruded
-   Path curves that create overlapping sections
-   Insufficient resolution for complex curves

**Solutions**:

-   Limit parameter ranges to prevent extreme deformations
-   Increase curve resolution for complex shapes
-   Add intersection detection in validation
-   Use manifold library's intersection removal features

### Best Practices for Manifold Generation

1. **Clockwise vertex ordering**: Always generate triangles with clockwise vertex order when viewed from outside
2. **Closed Loops**: Ensure profile curves form complete loops without gaps
3. **Edge Sharing**: Every edge must be shared by exactly two triangles
4. **Parameter Validation**: Clamp inputs to prevent degenerate cases
5. **Precision Handling**: Use epsilon comparisons for floating-point operations
6. **Debug Verification**: Always test with verbose logging enabled
7. **Incremental Testing**: Validate after each mesh generation step
-   Handle closed loops correctly

## Contributing Guidelines

### Code Structure

Each sculpted primitive follows this pattern:

```cpp
class CSGSculptedX3D : public CSGSculptedPrimitive3D {
    // Parameters
    real_t parameter_name = default_value;

    // Methods
    void _bind_methods();
    CSGSculptedX3D();
    CSGBrush *_build_brush() override;
};
```

### Parameter Validation

-   Validate input ranges to prevent degenerate geometry
-   Use `CLAMP()` for bounded parameters
-   Check for division by zero
-   Ensure positive values where required

### Mesh Generation Steps

1. **Profile Generation**: Create 2D profile points based on curve type
2. **Path Application**: Transform profile along path with sculpting parameters
3. **Face Creation**: Generate triangles with proper winding
4. **Manifold Validation**: Verify geometry is manifold
5. **CSGBrush Creation**: Convert to Godot's brush format

### Testing

-   Include unit tests for basic shape generation
-   Test edge cases (zero radius, extreme parameters)
-   Verify manifoldness with debug output
-   Check UV coordinate generation

### Documentation

-   Document all public parameters with PropertyInfo
-   Add class and method documentation
-   Include usage examples
-   Explain sculpting parameter effects

## Debug Output

When `verbose` logging is enabled, the primitives output detailed information:

```
CSGSculptedX3D::_build_brush() debug:
  Profile size: X, effective_profile_count: Y
  Hollow profile size: A, effective_hollow_count: B
  Total profile: C, path_segments: D
  Vertices generated: E
  Indices generated: F (face_count: G)
  Faces array size: H
  Brush faces after build_from_faces: I
```

Use this output to diagnose manifold issues and optimize performance.

## Performance Considerations

-   Minimize vertex count for real-time editing
-   Cache expensive calculations when possible
-   Use efficient algorithms for curve generation
-   Profile performance with different parameter combinations

## No Future Enhancements

-   Do not add additional curve types (Bézier, spline-based)
-   We already have Texture-based deformation in sculpted texture
-   Godot Engine can animate all properties in Variant
-   Sculpted uses CPU generation
-   Do not add advanced hollow shape options

## Commit Messages

Follow root [CONTRIBUTING.md](../../../CONTRIBUTING.md) guidelines strictly.

**Avoid conventional commits** (`feat:`, `fix:`, `docs:`, etc.).

Use Godot style:
- Imperative mood, capitalized title ≤72 chars
- Optional area prefix (`CSG:`, `Tests:`)
- Wrapped body at 80 chars

Example:
```
Fix CSG cap_open_ends winding for manifold validation

Reversed bottom cap triangle order for correct outward normals (CW from above).
```
