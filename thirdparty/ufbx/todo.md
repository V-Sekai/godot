# ufbx Export Implementation TODO

## Project Overview
Implement FBX export functionality in ufbx library and integrate with Godot's FBXDocument class to match GLTFDocument export capabilities.

**Total Estimated Time:** 4-6 weeks  
**Approach:** Leverage ufbx's existing import knowledge to implement export by "reversing" the import process.

---

## Phase 1: ufbx Export API Implementation (2-3 weeks)

### 1.1 Scene Construction API (1-1.5 weeks)
- [ ] **Study ufbx data structures** (2 days)
  - Analyze `ufbx_scene`, `ufbx_node`, `ufbx_mesh`, `ufbx_material` structures
  - Document memory layout and relationships
  - Understand existing allocation patterns

- [ ] **Create scene building functions** (3-4 days)
  - `ufbx_create_scene()` - Initialize empty scene
  - `ufbx_add_node()` - Add nodes to scene hierarchy
  - `ufbx_add_mesh()` - Add mesh data to scene
  - `ufbx_add_material()` - Add material definitions
  - `ufbx_add_animation()` - Add animation tracks

- [ ] **Implement hierarchy management** (2-3 days)
  - Parent-child node relationships
  - Transform inheritance
  - Scene graph validation
  - Memory management for constructed scenes

### 1.2 FBX Writing Implementation (1-1.5 weeks)
- [ ] **Analyze ufbx import format handling** (2 days)
  - Study how ufbx parses FBX binary format
  - Identify reusable compression/encoding logic
  - Document FBX version handling

- [ ] **Implement FBX binary writer** (4-5 days)
  - Serialize ufbx_scene to FBX binary format
  - Handle different FBX versions (2020, 2019, etc.)
  - Implement proper chunk/node writing
  - Add compression support

- [ ] **Add export validation** (1-2 days)
  - Validate scene structure before export
  - Error handling for malformed data
  - Memory leak prevention

---

## Phase 2: Godot Integration (1.5-2 weeks)

### 2.1 FBXDocument Export Methods (1-1.5 weeks)
- [ ] **Implement `append_from_scene()`** (3-4 days)
  - Convert Godot `Node` hierarchy to `ufbx_scene`
  - Handle coordinate system conversion (reuse import logic)
  - Map Godot node types to FBX equivalents
  - Process scene transforms and inheritance

- [ ] **Asset conversion pipeline** (2-3 days)
  - Convert `ImporterMesh` to `ufbx_mesh`
  - Map Godot materials to FBX material properties
  - Handle texture references and embedding
  - Convert animation tracks to FBX format

- [ ] **Handle special cases** (1-2 days)
  - Skinned meshes and skeleton export
  - Camera and light parameter mapping
  - Custom properties and metadata

### 2.2 File Output Implementation (0.5 weeks)
- [ ] **Implement `write_to_filesystem()`** (1-2 days)
  - Call ufbx export functions
  - Handle file path and naming
  - Error reporting and validation

- [ ] **Implement `generate_buffer()`** (1 day)
  - Export to memory buffer
  - Return PackedByteArray for in-memory operations

- [ ] **Integration testing** (1 day)
  - Test with existing FBXDocument import pipeline
  - Verify method signatures match GLTFDocument

---

## Phase 3: Testing & Polish (0.5-1 week)

### 3.1 Testing & Validation (0.5-1 week)
- [ ] **Round-trip testing** (2 days)
  - Export scene → Import back → Compare
  - Test with various scene complexities
  - Validate data integrity

- [ ] **Compatibility testing** (1-2 days)
  - Test exported FBX files in external tools (Blender, Maya)
  - Verify different FBX version outputs
  - Cross-platform file compatibility

- [ ] **Performance optimization** (1 day)
  - Profile export performance
  - Memory usage optimization
  - Large scene handling

---

## Technical Implementation Notes

### Key Files to Modify
- `modules/fbx/fbx_document.cpp` - Main export implementation
- `modules/fbx/fbx_document.h` - Export method declarations
- `thirdparty/ufbx/` - Add export functionality to ufbx library

### Data Flow
```
Godot Scene → FBXDocument::append_from_scene() → ufbx_scene → ufbx_export() → FBX File
```

### Coordinate System Handling
- Reuse existing import coordinate conversion logic in reverse
- Handle right-handed Y-up (ufbx) ↔ right-handed Y-up (Godot) conversions
- Apply proper transform matrices

### Memory Management
- Follow ufbx's existing allocation patterns
- Ensure proper cleanup of constructed scenes
- Handle large scene memory requirements

---

## Risk Assessment

### Low Risk ✅
- **Format Knowledge**: ufbx already understands FBX format
- **Data Structures**: All ufbx structures are defined and tested
- **Godot Integration**: Well-established patterns from GLTFDocument

### Medium Risk ⚠️
- **Performance**: Large scene export optimization
- **Memory Usage**: Efficient handling of complex scenes
- **Edge Cases**: Unusual scene configurations

### Mitigation Strategies
- Start with simple scenes and gradually add complexity
- Implement comprehensive testing at each phase
- Profile memory usage early and often
- Follow existing ufbx patterns closely

---

## Dependencies

### Prerequisites
- Understanding of ufbx library internals
- Familiarity with FBX format structure
- Knowledge of Godot's scene system

### External Dependencies
- ufbx library (already integrated)
- Godot's GLTFDocument patterns (for reference)

---

## Success Criteria

### Minimum Viable Product
- [ ] Export simple static meshes with materials
- [ ] Basic scene hierarchy preservation
- [ ] Compatible with standard FBX importers

### Full Feature Set
- [ ] Complete parity with GLTFDocument export capabilities
- [ ] Animation export support
- [ ] Skinned mesh and skeleton export
- [ ] Camera and light export
- [ ] Texture embedding and referencing

### Quality Metrics
- [ ] Round-trip fidelity > 95%
- [ ] Export performance comparable to GLTF export
- [ ] Memory usage within acceptable limits
- [ ] Cross-platform compatibility

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1-2 | ufbx Export API | Scene construction and FBX writing |
| 3-4 | Godot Integration | FBXDocument export methods |
| 4-5 | Testing & Polish | Validation and optimization |
| 5-6 | Buffer/Refinement | Edge cases and performance |

---

## Notes
- This approach leverages ufbx's existing format knowledge
- Much more feasible than implementing FBX export from scratch
- Can reuse and reverse existing import logic
- Maintains consistency with current ufbx architecture
---

## Current Testing Plan: FBX Export with Godot Primitives

### Immediate Testing Tasks
- [ ] Create standalone GDScript test for FBX export
- [ ] Set up test scene with Godot primitives (cube, sphere, cylinder)
- [ ] Test ASCII FBX export functionality
- [ ] Verify export API integration with existing ufbx test cases
- [ ] Create test cases based on ufbx test patterns
- [ ] Validate exported FBX files can be imported back
- [ ] Document test results and findings

### Test Scene Components
- [ ] **Basic Primitives**
  - Cube mesh with material
  - Sphere mesh with PBR material
  - Cylinder mesh with textures
  - Plane mesh with UV mapping

- [ ] **Scene Hierarchy**
  - Root node with multiple children
  - Nested transform hierarchies
  - Empty nodes for organization

- [ ] **Materials & Textures**
  - StandardMaterial3D with albedo
  - PBR materials with metallic/roughness
  - Normal maps and emission
  - Multiple materials per scene

- [ ] **Export Formats**
  - ASCII FBX export (primary focus)
  - Binary FBX export (secondary)
  - Different FBX versions (7400, 7500)

### Test Script Structure
- [x] Create `test_fbx_export.h` with ufbx test framework
- [x] Implement comprehensive test cases using UFBXT_TEST() macros
- [x] Set up material and mesh validation tests
- [x] Integrate with ufbx test suite via all_tests.h
- [x] Move GDScript test to thirdparty/ufbx/test/ directory
- [ ] Remove standalone validation test (test_export_simple.c)

### Validation Criteria
- [ ] Export API compiles without errors ✓ (Basic compilation fixed)
- [ ] ufbx test framework integration works
- [ ] Exported FBX files are valid format (when implementation is complete)
- [ ] ASCII format is human-readable (when implementation is complete)
- [ ] Mesh data integrity preserved (when implementation is complete)
- [ ] Material properties correctly exported (when implementation is complete)
- [ ] Scene hierarchy maintained (when implementation is complete)
- [ ] Files can be imported by external tools (when implementation is complete)

---

## Implementation Checklist

### Completed Tasks
- [x] Move ufbx_new export functionality to thirdparty/ufbx
- [x] Update export API header with missing declarations
- [x] Implement basic export functions in ufbx_export.c
- [x] Fix function signatures and error handling
- [x] Fix compilation errors in export implementation
- [x] Update error constants to use valid ufbx error types
- [x] Fix error handling to use ufbx_error.info array format
- [x] Fix allocator access patterns
- [x] Fix material property names (metallic -> metalness)
- [x] Fix mesh vertex attribute access patterns
- [x] Update test files to match fixed implementation
- [x] Consolidate test files into ufbx test framework
- [x] Move test files to thirdparty/ufbx/test/ directory

### Current Sprint Tasks
- [ ] Remove unnecessary standalone validation test
- [ ] Implement actual FBX export functionality (currently stubbed)
- [ ] Test export functionality with ufbx test framework
- [ ] Validate exported FBX files can be imported back
- [ ] Document test results and API usage

### Code Review Checkpoints
- [ ] Export API implementation review
- [ ] Test script functionality review
- [ ] ASCII export validation review
- [ ] Integration with existing ufbx tests review
- [ ] Final testing and documentation review

---

## Development Environment Setup

### Required Tools
- [ ] Godot development build environment
- [ ] FBX SDK documentation access
- [ ] External FBX viewers (FBX Review, Blender, Maya)
- [ ] Memory profiling tools (Valgrind, AddressSanitizer)
