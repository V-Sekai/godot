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

### Validation Checks

The code includes several validation mechanisms:

1. **Debug Output**: Verbose logging shows vertex counts, face counts, and manifold status
2. **Manifold Library Validation**: Uses the manifold library's built-in error checking
3. **Profile Closure**: Ensures profile curves are properly closed loops
4. **Face Winding**: Maintains consistent counter-clockwise winding for manifoldness

### Common Issues and Solutions

#### 1. Non-Manifold Edges

**Symptom**: Manifold library reports "Not Manifold" error
**Causes**:

-   Incorrect vertex indexing in face generation
-   Missing or duplicate edges
-   Improper loop closure

**Solutions**:

-   Verify all triangles share edges correctly
-   Ensure profile loops are closed without duplicate vertices
-   Check face winding order (counter-clockwise from outside)

#### 2. Degenerate Faces

**Symptom**: Zero-area triangles or invalid geometry
**Causes**:

-   Collinear vertices
-   Incorrect parameter combinations
-   Floating-point precision issues

**Solutions**:

-   Add epsilon checks for near-zero values
-   Validate parameter ranges
-   Use robust geometric computations

#### 3. Path/Profile Mismatches

**Symptom**: Inconsistent vertex counts between path segments
**Causes**:

-   Profile point count changes along path
-   Incorrect handling of closed vs open curves

**Solutions**:

-   Pre-calculate profile sizes before vertex generation
-   Ensure consistent segmentation
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
-   Texture-based deformation
-   Animation support for parameters
-   GPU-accelerated generation
-   Advanced hollow shape options
