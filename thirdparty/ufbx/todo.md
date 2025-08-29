# ufbx Export Implementation TODO

## Project Overview

Implement FBX export functionality in ufbx library and integrate with Godot's FBXDocument class to match GLTFDocument export capabilities.

**Total Estimated Time:** 4-6 weeks  
**Approach:** Leverage ufbx's existing import knowledge to implement export by "reversing" the import process.

---

## Phase 1: ufbx Export API Implementation (2-3 weeks)

### 1.1 Scene Construction API (1-1.5 weeks)

-   [ ] **Study ufbx data structures** (2 days)

    -   Analyze `ufbx_scene`, `ufbx_node`, `ufbx_mesh`, `ufbx_material` structures
    -   Document memory layout and relationships
    -   Understand existing allocation patterns

-   [ ] **Create scene building functions** (3-4 days)

    -   `ufbx_create_scene()` - Initialize empty scene
    -   `ufbx_add_node()` - Add nodes to scene hierarchy
    -   `ufbx_add_mesh()` - Add mesh data to scene
    -   `ufbx_add_material()` - Add material definitions
    -   `ufbx_add_animation()` - Add animation tracks

-   [ ] **Implement hierarchy management** (2-3 days)
    -   Parent-child node relationships
    -   Transform inheritance
    -   Scene graph validation
    -   Memory management for constructed scenes

### 1.2 FBX Writing Implementation (1-1.5 weeks)

-   [ ] **Analyze ufbx import format handling** (2 days)

    -   Study how ufbx parses FBX binary format
    -   Identify reusable compression/encoding logic
    -   Document FBX version handling

-   [ ] **Implement FBX binary writer** (4-5 days)

    -   Serialize ufbx_scene to FBX binary format
    -   Handle different FBX versions (2020, 2019, etc.)
    -   Implement proper chunk/node writing
    -   Add compression support

-   [ ] **Add export validation** (1-2 days)
    -   Validate scene structure before export
    -   Error handling for malformed data
    -   Memory leak prevention

---

## Phase 2: Godot Integration (1.5-2 weeks)

### 2.1 FBXDocument Export Methods (1-1.5 weeks)

-   [ ] **Implement `append_from_scene()`** (3-4 days)

    -   Convert Godot `Node` hierarchy to `ufbx_scene`
    -   Handle coordinate system conversion (reuse import logic)
    -   Map Godot node types to FBX equivalents
    -   Process scene transforms and inheritance

-   [ ] **Asset conversion pipeline** (2-3 days)

    -   Convert `ImporterMesh` to `ufbx_mesh`
    -   Map Godot materials to FBX material properties
    -   Handle texture references and embedding
    -   Convert animation tracks to FBX format

-   [ ] **Handle special cases** (1-2 days)
    -   Skinned meshes and skeleton export
    -   Camera and light parameter mapping
    -   Custom properties and metadata

### 2.2 File Output Implementation (0.5 weeks)

-   [ ] **Implement `write_to_filesystem()`** (1-2 days)

    -   Call ufbx export functions
    -   Handle file path and naming
    -   Error reporting and validation

-   [ ] **Implement `generate_buffer()`** (1 day)

    -   Export to memory buffer
    -   Return PackedByteArray for in-memory operations

-   [ ] **Integration testing** (1 day)
    -   Test with existing FBXDocument import pipeline
    -   Verify method signatures match GLTFDocument

---

## Phase 3: Testing & Polish (0.5-1 week)

### 3.1 Testing & Validation (0.5-1 week)

-   [ ] **Round-trip testing** (2 days)

    -   Export scene → Import back → Compare
    -   Test with various scene complexities
    -   Validate data integrity

-   [ ] **Compatibility testing** (1-2 days)

    -   Test exported FBX files in external tools (Blender, Maya)
    -   Verify different FBX version outputs
    -   Cross-platform file compatibility

-   [ ] **Performance optimization** (1 day)
    -   Profile export performance
    -   Memory usage optimization
    -   Large scene handling

---

## Technical Implementation Notes

### Key Files to Modify

-   `modules/fbx/fbx_document.cpp` - Main export implementation
-   `modules/fbx/fbx_document.h` - Export method declarations
-   `thirdparty/ufbx/` - Add export functionality to ufbx library

### Data Flow

```
Godot Scene → FBXDocument::append_from_scene() → ufbx_scene → ufbx_export() → FBX File
```

### Coordinate System Handling

-   Reuse existing import coordinate conversion logic in reverse
-   Handle right-handed Y-up (ufbx) ↔ right-handed Y-up (Godot) conversions
-   Apply proper transform matrices

### Memory Management

-   Follow ufbx's existing allocation patterns
-   Ensure proper cleanup of constructed scenes
-   Handle large scene memory requirements

---

## Risk Assessment

### Low Risk ✅

-   **Format Knowledge**: ufbx already understands FBX format
-   **Data Structures**: All ufbx structures are defined and tested
-   **Godot Integration**: Well-established patterns from GLTFDocument

### Medium Risk ⚠️

-   **Performance**: Large scene export optimization
-   **Memory Usage**: Efficient handling of complex scenes
-   **Edge Cases**: Unusual scene configurations

### Mitigation Strategies

-   Start with simple scenes and gradually add complexity
-   Implement comprehensive testing at each phase
-   Profile memory usage early and often
-   Follow existing ufbx patterns closely

---

## Dependencies

### Prerequisites

-   Understanding of ufbx library internals
-   Familiarity with FBX format structure
-   Knowledge of Godot's scene system

### External Dependencies

-   ufbx library (already integrated)
-   Godot's GLTFDocument patterns (for reference)

---

## Success Criteria

### Minimum Viable Product

-   [ ] Export simple static meshes with materials
-   [ ] Basic scene hierarchy preservation
-   [ ] Compatible with standard FBX importers

### Full Feature Set

-   [ ] Complete parity with GLTFDocument export capabilities
-   [ ] Animation export support
-   [ ] Skinned mesh and skeleton export
-   [ ] Camera and light export
-   [ ] Texture embedding and referencing

### Quality Metrics

-   [ ] Round-trip fidelity > 95%
-   [ ] Export performance comparable to GLTF export
-   [ ] Memory usage within acceptable limits
-   [ ] Cross-platform compatibility

---

## Timeline

| Week | Focus             | Deliverables                       |
| ---- | ----------------- | ---------------------------------- |
| 1-2  | ufbx Export API   | Scene construction and FBX writing |
| 3-4  | Godot Integration | FBXDocument export methods         |
| 4-5  | Testing & Polish  | Validation and optimization        |
| 5-6  | Buffer/Refinement | Edge cases and performance         |

---

## Notes

-   This approach leverages ufbx's existing format knowledge
-   Much more feasible than implementing FBX export from scratch
-   Can reuse and reverse existing import logic
-   Maintains consistency with current ufbx architecture

---

## Current Testing Plan: FBX Export with Godot Primitives

### Immediate Testing Tasks

-   [ ] Create standalone GDScript test for FBX export
-   [ ] Set up test scene with Godot primitives (cube, sphere, cylinder)
-   [ ] Test ASCII FBX export functionality
-   [ ] Verify export API integration with existing ufbx test cases
-   [ ] Create test cases based on ufbx test patterns
-   [ ] Validate exported FBX files can be imported back
-   [ ] Document test results and findings

### Test Scene Components

-   [ ] **Basic Primitives**

    -   Cube mesh with material
    -   Sphere mesh with PBR material
    -   Cylinder mesh with textures
    -   Plane mesh with UV mapping

-   [ ] **Scene Hierarchy**

    -   Root node with multiple children
    -   Nested transform hierarchies
    -   Empty nodes for organization

-   [ ] **Materials & Textures**

    -   StandardMaterial3D with albedo
    -   PBR materials with metallic/roughness
    -   Normal maps and emission
    -   Multiple materials per scene

-   [ ] **Export Formats**
    -   ASCII FBX export (primary focus)
    -   Binary FBX export (secondary)
    -   Different FBX versions (7400, 7500)

### Test Script Structure

-   [x] Create `test_fbx_export.h` with ufbx test framework
-   [x] Implement comprehensive test cases using UFBXT_TEST() macros
-   [x] Set up material and mesh validation tests
-   [x] Integrate with ufbx test suite via all_tests.h
-   [x] Move GDScript test to thirdparty/ufbx/test/ directory
-   [ ] Remove standalone validation test (test_export_simple.c)

### Validation Criteria

-   [ ] Export API compiles without errors ✓ (Basic compilation fixed)
-   [ ] ufbx test framework integration works
-   [ ] Exported FBX files are valid format (when implementation is complete)
-   [ ] ASCII format is human-readable (when implementation is complete)
-   [ ] Mesh data integrity preserved (when implementation is complete)
-   [ ] Material properties correctly exported (when implementation is complete)
-   [ ] Scene hierarchy maintained (when implementation is complete)
-   [ ] Files can be imported by external tools (when implementation is complete)

---

## Implementation Checklist

### Completed Tasks

-   [x] Move ufbx_new export functionality to thirdparty/ufbx
-   [x] Update export API header with missing declarations
-   [x] Implement basic export functions in ufbx_export.c
-   [x] Fix function signatures and error handling
-   [x] Fix compilation errors in export implementation
-   [x] Update error constants to use valid ufbx error types
-   [x] Fix error handling to use ufbx_error.info array format
-   [x] Fix allocator access patterns
-   [x] Fix material property names (metallic -> metalness)
-   [x] Fix mesh vertex attribute access patterns
-   [x] Update test files to match fixed implementation
-   [x] Consolidate test files into ufbx test framework
-   [x] Move test files to thirdparty/ufbx/test/ directory
-   [x] Implement scene construction API (ufbx_create_scene, ufbx_add_node, etc.)
-   [x] Implement mesh data setting functions (vertices, indices, normals, UVs)
-   [x] Implement material property setting functions
-   [x] Create comprehensive test suite with ufbx test framework
-   [x] Implement basic error handling and validation
-   [x] Set up memory management for export scenes

### Current Sprint Tasks - Core FBX Writer Implementation

-   [ ] **CRITICAL: Implement FBX binary writer** (ufbx_export_writer.c)

    -   [ ] Study ufbx.c import format parsing for reverse engineering
    -   [ ] Implement FBX header writing (version, timestamp, creator)
    -   [ ] Implement FBX node/property tree serialization
    -   [ ] Add compression support (zlib/deflate for large data blocks)
    -   [ ] Handle different FBX versions (7400, 7500, etc.)

-   [ ] **Implement ASCII FBX writer** (alternative format)

    -   [ ] Create human-readable FBX format writer
    -   [ ] Implement proper indentation and formatting
    -   [ ] Handle special characters and escaping
    -   [ ] Add ASCII-specific validation

-   [ ] **Complete export pipeline integration**

    -   [ ] Connect scene construction API to FBX writer
    -   [ ] Implement ufbx_export_to_file() with actual writing
    -   [ ] Implement ufbx_export_to_memory() with buffer management
    -   [ ] Add proper size calculation in ufbx_get_export_size()

-   [ ] **Advanced scene features**
    -   [ ] Implement animation export (ufbx_add_animation)
    -   [ ] Add camera and light export support
    -   [ ] Handle texture embedding vs. referencing
    -   [ ] Implement proper parent-child relationships in nodes
    -   [ ] Add support for multiple materials per mesh

### Next Phase Tasks

-   [ ] **Godot Integration** (modules/fbx/fbx_document.cpp)

    -   [ ] Implement FBXDocument::append_from_scene()
    -   [ ] Convert Godot Node hierarchy to ufbx_export_scene
    -   [ ] Map Godot materials to ufbx material properties
    -   [ ] Handle Godot-specific coordinate system conversion
    -   [ ] Implement FBXDocument::write_to_filesystem()
    -   [ ] Implement FBXDocument::generate_buffer()

-   [ ] **Round-trip testing and validation**
    -   [ ] Test export → import → compare workflow
    -   [ ] Validate with external FBX tools (Blender, Maya)
    -   [ ] Performance testing with large scenes
    -   [ ] Memory leak detection and optimization

### Code Review Checkpoints

-   [ ] Export API implementation review
-   [ ] Test script functionality review
-   [ ] ASCII export validation review
-   [ ] Integration with existing ufbx tests review
-   [ ] Final testing and documentation review

---

## Development Environment Setup

### Required Tools

-   [ ] Godot development build environment
-   [ ] FBX SDK documentation access
-   [ ] External FBX viewers (FBX Review, Blender, Maya)
-   [ ] Memory profiling tools (Valgrind, AddressSanitizer)

---

## Detailed Implementation Guide

### Critical Next Steps (Priority Order)

#### 1. FBX Binary Writer Implementation (HIGHEST PRIORITY)

The core blocker is implementing the actual FBX file writing functionality. Currently, `ufbx_export_to_file()` and `ufbx_export_to_memory()` return "not implemented" errors.

**Key Implementation Areas:**

-   [ ] **FBX File Header Structure**

    ```c
    // FBX binary header format (first 27 bytes)
    // "Kaydara FBX Binary  \x00\x1a\x00" + version (4 bytes)
    ```

    -   [ ] Write magic header bytes
    -   [ ] Set FBX version (7400 = FBX 2014/2015)
    -   [ ] Add timestamp and creator info

-   [ ] **FBX Node Tree Serialization**

    -   [ ] Study ufbx.c parsing logic for node structure
    -   [ ] Implement property writing (P: properties, C: connections)
    -   [ ] Handle nested node hierarchies
    -   [ ] Write null terminator for node lists

-   [ ] **Data Block Compression**
    -   [ ] Implement zlib compression for large vertex arrays
    -   [ ] Handle uncompressed vs compressed data blocks
    -   [ ] Add proper size headers for compressed blocks

#### 2. ASCII FBX Writer (MEDIUM PRIORITY)

Easier to implement and debug than binary format.

-   [ ] **ASCII Format Structure**

    ```
    ; FBX 7.4.0 project file
    ; Created by ufbx_export

    FBXHeaderExtension:  {
        FBXHeaderVersion: 1003
        FBXVersion: 7400
        Creator: "ufbx_export"
    }
    ```

    -   [ ] Implement proper indentation (tabs vs spaces)
    -   [ ] Handle property value formatting (strings, numbers, arrays)
    -   [ ] Add comment generation for debugging

#### 3. Scene Data Serialization

-   [ ] **Geometry Export**

    -   [ ] Convert ufbx_mesh to FBX Geometry nodes
    -   [ ] Write vertex arrays with proper indexing
    -   [ ] Handle normal and UV coordinate export
    -   [ ] Implement face index mapping

-   [ ] **Material Export**

    -   [ ] Convert ufbx_material to FBX Material nodes
    -   [ ] Map PBR properties to FBX material properties
    -   [ ] Handle texture connections and references
    -   [ ] Support multiple material slots per mesh

-   [ ] **Transform Export**
    -   [ ] Convert ufbx_transform to FBX transform properties
    -   [ ] Handle local vs world space transforms
    -   [ ] Implement proper matrix decomposition
    -   [ ] Support transform inheritance chains

### Implementation Strategy

#### Phase A: Minimal ASCII Export (1-2 weeks)

Focus on getting basic ASCII export working first, as it's easier to debug and validate.

-   [ ] **Week 1: Basic ASCII Structure**

    -   [ ] Implement FBX header writing
    -   [ ] Create basic node tree structure
    -   [ ] Export simple geometry (single mesh)
    -   [ ] Add basic material support

-   [ ] **Week 2: ASCII Feature Completion**
    -   [ ] Add hierarchy support
    -   [ ] Implement texture references
    -   [ ] Add transform export
    -   [ ] Create validation tests

#### Phase B: Binary Export Implementation (2-3 weeks)

Once ASCII export is working, implement binary format for production use.

-   [ ] **Week 3-4: Binary Writer Core**

    -   [ ] Study ufbx binary parsing code
    -   [ ] Implement binary header and node writing
    -   [ ] Add compression support
    -   [ ] Port ASCII logic to binary format

-   [ ] **Week 5: Binary Feature Completion**
    -   [ ] Optimize binary output size
    -   [ ] Add version compatibility
    -   [ ] Implement proper error handling
    -   [ ] Performance testing and optimization

### Testing and Validation Strategy

#### Unit Testing (Ongoing)

-   [x] Basic API functionality tests (completed)
-   [x] Memory management tests (completed)
-   [x] Error handling validation (completed)
-   [ ] **FBX Format Validation Tests**
    -   [ ] Test exported files with FBX format validators
    -   [ ] Verify header structure correctness
    -   [ ] Validate node tree integrity

#### Integration Testing

-   [ ] **Round-trip Testing**

    -   [ ] Export scene → Import with ufbx → Compare data
    -   [ ] Test with various scene complexities
    -   [ ] Validate transform preservation
    -   [ ] Check material property accuracy

-   [ ] **External Tool Compatibility**
    -   [ ] Test imports in Blender 3.x+
    -   [ ] Test imports in Autodesk Maya
    -   [ ] Test imports in 3ds Max
    -   [ ] Verify FBX Review compatibility

#### Performance Testing

-   [ ] **Memory Usage Analysis**
    -   [ ] Profile memory allocation patterns
    -   [ ] Test with large scenes (10k+ vertices)
    -   [ ] Validate cleanup and leak prevention
    -   [ ] Benchmark against GLTF export performance

### Known Technical Challenges

#### 1. FBX Format Complexity

-   **Challenge**: FBX binary format has complex nested structure
-   **Solution**: Reverse-engineer from ufbx import parsing logic
-   **Timeline**: 1-2 weeks of careful analysis

#### 2. Coordinate System Conversion

-   **Challenge**: Ensuring proper transform handling between Godot and FBX
-   **Solution**: Reuse existing import conversion matrices in reverse
-   **Timeline**: 2-3 days of testing and validation

#### 3. Material Property Mapping

-   **Challenge**: Converting Godot PBR materials to FBX material system
-   **Solution**: Study existing GLTFDocument material conversion patterns
-   **Timeline**: 3-4 days of implementation and testing

### Debug and Development Tools

#### Recommended Development Workflow

1. **Start with ASCII export** - easier to debug and validate
2. **Use external FBX viewers** - validate output immediately
3. **Implement incremental features** - test each component separately
4. **Profile early and often** - catch performance issues early

#### Debug Output Recommendations

-   [ ] Add verbose logging for export process
-   [ ] Implement scene dumping for debugging
-   [ ] Create ASCII preview mode for binary exports
-   [ ] Add validation checkpoints throughout export pipeline

---

## Future Enhancement Opportunities

### Advanced Features (Post-MVP)

-   [ ] **Animation Export**

    -   [ ] Keyframe animation support
    -   [ ] Skeletal animation export
    -   [ ] Morph target animations
    -   [ ] Custom animation curves

-   [ ] **Advanced Geometry**

    -   [ ] NURBS surface export
    -   [ ] Subdivision surface support
    -   [ ] Instanced geometry optimization
    -   [ ] Level-of-detail (LOD) support

-   [ ] **Lighting and Cameras**
    -   [ ] Directional/point/spot light export
    -   [ ] Camera parameter export
    -   [ ] Environment lighting support
    -   [ ] Shadow map references

### Performance Optimizations

-   [ ] **Memory Optimization**

    -   [ ] Streaming export for large scenes
    -   [ ] Incremental scene building
    -   [ ] Memory pool allocation
    -   [ ] Lazy data loading

-   [ ] **Export Speed**
    -   [ ] Multi-threaded export processing
    -   [ ] Parallel mesh processing
    -   [ ] Optimized compression algorithms
    -   [ ] Cached export data structures

### Integration Enhancements

-   [ ] **Godot Editor Integration**

    -   [ ] Export progress indicators
    -   [ ] Export option dialogs
    -   [ ] Batch export functionality
    -   [ ] Export preset management

-   [ ] **Pipeline Integration**
    -   [ ] Asset pipeline integration
    -   [ ] Build system integration
    -   [ ] Continuous integration testing
    -   [ ] Automated validation workflows
