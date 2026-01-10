# CASSIE Module - Developer Guide

## Overview
Complete implementation of the CASSIE (Curve and Surface Sketching in Immersive Environments) module for Godot Engine, based on the CHI 2021 research paper.

**Implementation Status**: ✅ COMPLETE

## Core Classes

### CassieSurface (High-level API) - START HERE
**Recommended for most applications** - orchestrates the entire pipeline.

```gdscript
var surface = CassieSurface.new()

# Add boundary paths
var path = CassiePath3D.new()
path.add_point(Vector3(0, 0, 0))
path.add_point(Vector3(1, 0, 0))
# ... more points
surface.add_boundary_path(path)

# Generate surface
var mesh = surface.generate_surface()
```

**Key Properties**:
- `auto_beautify: bool` - Enable Taubin smoothing (default: true)
- `auto_resample: bool` - Enable uniform resampling (default: true)
- `use_intrinsic_remeshing: bool` - Enable Delaunay refinement (default: true)
- `target_boundary_points: int` - Points after resampling (default: 30)
- `beautify_lambda: float` - Smoothing strength 0-1 (default: 0.5)
- `beautify_mu: float` - Anti-shrinkage -1-0 (default: -0.53)
- `max_flip_iterations: int` - Edge flip iterations (default: 100)

**Pipeline Stages**:
1. Curve beautification (Taubin smoothing)
2. Uniform arc-length resampling
3. DMWT polygon triangulation
4. Intrinsic Delaunay remeshing
5. Position smoothing

### CassiePath3D (Curve Processing)
**Use for curve manipulation and beautification** before triangulation.

```gdscript
var path = CassiePath3D.new()

# Add points with normals
path.add_point(Vector3(x, y, z), Vector3(nx, ny, nz))

# Beautify curve
path.beautify_taubin(0.5, -0.53, 5)  # lambda, mu, iterations

# Resample to uniform spacing
path.resample_uniform(50)  # target point count

# Get processed data
var points = path.get_points()
var normals = path.get_normals()
```

**Key Methods**:
- `add_point(pos, normal)` - Add point to path
- `beautify_laplacian(lambda, iterations)` - Simple smoothing
- `beautify_taubin(lambda, mu, iterations)` - Better smoothing (preserves volume)
- `resample_uniform(count)` - Uniform arc-length spacing
- `smooth_normals()` - Average neighboring normals
- `get_total_length()` - Path length
- `set_closed(bool)` - Mark as closed loop

**Taubin Smoothing Parameters**:
- **λ (lambda)**: Smoothing strength, typically 0.5-0.6
- **μ (mu)**: Anti-shrinkage compensation, typically -λ/(1+λ) ≈ -0.53

### PolygonTriangulationGodot (Triangulation)
**Use for low-level control** over polygon triangulation.

```gdscript
# Create from boundary points
var triangulator = PolygonTriangulationGodot.create(boundary_points)

# Configure (optional)
triangulator.set_cost_weights(1.0, 0.5, 0.0, 1.0, 0.0)
triangulator.set_optimization_rounds(3)

# Preprocess and triangulate
triangulator.preprocess()
triangulator.triangulate()

# Get results
var mesh = triangulator.get_mesh()
var vertices = triangulator.get_vertices()
var indices = triangulator.get_indices()
```

**Cost Weights** (tune triangulation quality):
- `triangle_cost` - Area-based triangle cost (default: 1.0)
- `edge_cost` - Edge length cost (default: 0.5)
- `bi_triangle_cost` - Two-triangle configuration cost (default: 0.0)
- `triangle_boundary_cost` - Boundary triangle cost (default: 1.0)
- `worst_dihedral_cost` - Dihedral angle penalty (default: 0.0)

**Algorithm**: DMWT (Dynamic Programming Multi-Way Tiling)
- Dynamic programming on Delaunay tetrahedralization
- Optimizes triangle quality according to cost weights
- Complexity: O(n³) preprocessing, O(n²) triangulation

### IntrinsicTriangulation (Mesh Refinement)
**Use for mesh quality improvement** after initial triangulation.

```gdscript
var intrinsic = IntrinsicTriangulation.new()

# Initialize from mesh
intrinsic.set_mesh(initial_mesh)

# Or from raw data
intrinsic.set_mesh_data(vertices, indices, normals)

# Flip to Delaunay
intrinsic.set_max_flip_iterations(100)
intrinsic.flip_to_delaunay()

# Refine mesh
intrinsic.refine_intrinsic_triangulation(0.1)  # target edge length

# Smooth positions
intrinsic.smooth_intrinsic_positions(5)  # iterations

# Get result
var refined_mesh = intrinsic.get_mesh()
```

**Key Methods**:
- `flip_to_delaunay()` - Optimize triangle quality via edge flipping
- `refine_intrinsic_triangulation(length)` - Uniform edge lengths
- `smooth_intrinsic_positions(iterations)` - Smooth vertex positions
- `get_statistics()` - Edge/triangle counts and lengths

**Delaunay Criterion**: Sum of opposite angles ≤ π
- Maximizes minimum angle
- Avoids thin/slivered triangles
- Implemented via iterative edge flipping

## Common Workflows

### Workflow 1: Simple Surface from Sketch
```gdscript
# User draws a stroke in 3D
var sketch_points = []  # Fill with user input

# Create path
var path = CassiePath3D.new()
for point in sketch_points:
    path.add_point(point)

# Generate surface (automatic pipeline)
var surface = CassieSurface.new()
surface.add_boundary_path(path)
var mesh = surface.generate_surface()

# Display
$MeshInstance3D.mesh = mesh
```

### Workflow 2: Custom Curve Processing
```gdscript
var path = CassiePath3D.new()
# ... add points

# Manual beautification (more aggressive than default)
path.beautify_taubin(0.6, -0.55, 10)
path.resample_uniform(40)

# Manual triangulation
var tri = PolygonTriangulationGodot.create(path.get_points())
tri.preprocess()
tri.triangulate()
var mesh = tri.get_mesh()
```

### Workflow 3: High-quality Mesh Refinement
```gdscript
# Start with basic triangulation
var surface = CassieSurface.new()
surface.use_intrinsic_remeshing = false  # Skip auto-remeshing
# ... add paths
var base_mesh = surface.generate_surface()

# Apply custom intrinsic refinement (multiple passes)
var intrinsic = IntrinsicTriangulation.new()
intrinsic.set_mesh(base_mesh)

for i in range(3):
    intrinsic.flip_to_delaunay()
    intrinsic.refine_intrinsic_triangulation(-1.0)  # Auto edge length
    intrinsic.smooth_intrinsic_positions(10)

var refined = intrinsic.get_mesh()
```

### Workflow 4: Closed Loop Surface
```gdscript
var path = CassiePath3D.new()
# Add points in loop
for angle in range(0, 360, 30):
    var rad = deg_to_rad(angle)
    path.add_point(Vector3(cos(rad), sin(rad), 0))

path.set_closed(true)  # Mark as closed loop

var surface = CassieSurface.new()
surface.add_boundary_path(path)
var mesh = surface.generate_surface()
```

## Performance Guidelines

- **Point Count**: 20-50 points per boundary is typically sufficient
- **Smoothing Iterations**: 3-5 iterations for most cases
- **Edge Flipping**: 50-100 iterations typically converges
- **Mesh Refinement**: Skip if input is already good quality

## Troubleshooting

### Mesh Not Generated
- Check `mesh.is_valid()` and `mesh.get_surface_count() > 0`
- Ensure path has at least 3 points
- Check `triangulator.preprocess()` return value

### Poor Triangle Quality
- Increase `max_flip_iterations`
- Enable intrinsic remeshing
- Try different cost weights

### Curve Too Rough/Smooth
- Adjust `beautify_lambda` (higher = smoother)
- Change `beautify_iterations` (more = smoother)
- Use Taubin instead of Laplacian to preserve volume

### Performance Issues
- Reduce `target_boundary_points`
- Disable `use_intrinsic_remeshing` for draft mode
- Lower `max_flip_iterations`

## Debugging

Enable DOT output for triangulation visualization:
```gdscript
var tri = PolygonTriangulationGodot.create(points)
tri.enable_dot_output(true)  # Generates .dot files
tri.triangulate()
# Check console for .dot file paths
```

Get statistics:
```gdscript
var stats = triangulator.get_statistics()
print("Triangles: ", stats["triangle_count"])
print("Vertices: ", stats["vertex_count"])
print("Optimal cost: ", stats["optimal_cost"])

var intrinsic_stats = intrinsic.get_statistics()
print("Edges: ", intrinsic_stats["edge_count"])
print("Avg edge length: ", intrinsic_stats["avg_edge_length"])
```

## Algorithm References

### Taubin Smoothing
- **Paper**: "A Signal Processing Approach To Fair Surface Design" (Taubin, 1995)
- **Method**: Two-pass filter with λ (smoothing) and μ (anti-shrinkage) passes
- **Benefits**: Removes noise while preserving volume

### Delaunay Triangulation
- **Criterion**: Sum of opposite angles ≤ π
- **Benefits**: Maximizes minimum angle, avoids thin triangles
- **Implementation**: Iterative edge flipping until convergence

### DMWT (Dynamic Multi-Way Tiling)
- **Paper**: "Curve and Surface Sketching in Immersive Environments" (CHI 2021)
- **Authors**: Yotam Gingold, et al.
- **Method**: Dynamic programming on Delaunay tetrahedralization
- **Optimizes**: Triangle quality according to custom cost weights
- **URL**: http://www-sop.inria.fr/reves/Basilic/2021/YASBS21

## Class Hierarchy
```
RefCounted
├── CassieSurface (orchestrator)
├── PolygonTriangulationGodot (wrapper)
└── IntrinsicTriangulation (refinement)

Resource
└── CassiePath3D (curve data)

RefCounted
└── PolygonTriangulation (C++ backend - DMWT algorithm)
```

## Implementation Details

### Core Components

1. **PolygonTriangulationGodot** (`src/polygon_triangulation_godot.h/.cpp`)
   - Godot-friendly wrapper around C++ DMWT triangulation
   - Handles PackedVector3Array ↔ C array conversion
   - Factory methods: `create()`, `create_planar()`
   - Configuration: cost weights, optimization rounds, point limits
   - Result extraction: vertices, indices, normals, ArrayMesh

2. **CassiePath3D** (`src/cassie_path_3d.h/.cpp`)
   - 3D curve representation and beautification
   - Point/normal management (add, insert, remove, modify)
   - Laplacian and Taubin smoothing algorithms
   - Uniform arc-length resampling
   - Path analysis (length, average segment, closed loops)

3. **IntrinsicTriangulation** (`src/intrinsic_triangulation.h/.cpp`)
   - Intrinsic geometry-based mesh refinement
   - Edge/triangle structures with intrinsic lengths
   - Delaunay criterion checking via circumcircle tests
   - Edge flipping algorithm with configurable iterations
   - Mesh refinement (edge split/collapse)
   - Laplacian position smoothing
   - Statistics collection

4. **CassieSurface** (`src/cassie_surface.h/.cpp`)
   - High-level pipeline orchestrator
   - Multi-boundary path management
   - Automated: beautify → resample → triangulate → remesh
   - Configurable stages (enable/disable each step)
   - Parameter tuning for all algorithms
   - Result caching

### Testing

9 comprehensive test cases covering:
- Basic triangulation (monkey's saddle)
- Simple square triangulation
- Point add/retrieve operations
- Laplacian smoothing
- Uniform resampling
- Mesh initialization
- Delaunay flipping
- Complete pipeline (circular boundary)
- Multiple boundaries
- Planar projection

### Build Configuration

- **register_types.cpp**: Registers all classes (PolygonTriangulation, PolygonTriangulationGodot, CassiePath3D, IntrinsicTriangulation, CassieSurface)
- **config.py**: Defines doc_classes list
- **SCsub**: Build script includes src/*.cpp files
- **DMWT.h**: Cleaned up unused includes

## Future Enhancements

1. **Multi-boundary triangulation**: Proper hole handling in CassieSurface
2. **Performance optimization**: Parallel edge flipping, SIMD smoothing
3. **Advanced features**:
   - Adaptive mesh refinement based on curvature
   - Feature-preserving smoothing
   - Texture coordinate generation
   - Material assignment
4. **Godot Editor integration**:
   - Custom EditorPlugin for sketch-based modeling
   - Gizmo for interactive path editing
   - Inspector property editors

## Testing

Build and test with:
```bash
# From godot root
scons platform=macos module_cassie_enabled=yes target=editor dev_build=yes

# Run tests
./bin/godot.macos.editor.dev.arm64 --test --test-case="[Modules][Cassie]*"
```

## License
All code follows Godot Engine's MIT license and copyright header format.
