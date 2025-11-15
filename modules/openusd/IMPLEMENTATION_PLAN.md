# USD Import Module Implementation Plan

## Overview

This document outlines the implementation plan for the `usd_import` module, which enables importing Universal Scene Description (USD) files (`.usd`, `.usda`, `.usdc`) into Godot as PackedScenes. The module follows the architectural pattern established by the FBX module, leveraging `GLTFDocument` and `GLTFState` to convert USD data directly into Godot scenes.

## Architecture

The module follows the FBX module pattern:
- **USDDocument**: Inherits from `GLTFDocument`, handles USD parsing and populates `GLTFState`
- **USDState**: Inherits from `GLTFState`, stores USD-specific data (stage reference)
- **EditorSceneFormatImporterUSD**: Editor importer that registers USD file extensions
- Uses `SkinTool` for skeleton/skin processing (same as FBX module)
- Populates `GLTFState` directly (no intermediate GLTF file conversion)

## Implementation Status

### ✅ Completed

1. **Module Structure**
   - ✅ Created `modules/usd_import/` directory structure
   - ✅ `SCsub` - Build configuration linking USD libraries from `openusd` module
   - ✅ `config.py` - Module configuration with dependencies (`gltf`, `openusd`)
   - ✅ `register_types.h/cpp` - Module registration with Godot engine

2. **Core Classes**
   - ✅ `USDState` (`usd_state.h/cpp`)
     - Inherits from `GLTFState`
     - Stores `UsdStageRefPtr` reference
     - Provides accessors for USD stage
   
   - ✅ `USDDocument` (`usd_document.h/cpp`)
     - Inherits from `GLTFDocument`
     - Implements `append_from_file()` to load USD files
     - Implements parsing methods to populate `GLTFState`
     - Uses `GLTFDocument::generate_scene()` for scene generation

3. **Editor Integration**
   - ✅ `EditorSceneFormatImporterUSD` (`editor/editor_scene_importer_usd.h/cpp`)
     - Registers file extensions: `.usd`, `.usda`, `.usdc`
     - Implements `import_scene()` method
     - Provides import options (naming version, embedded image handling)
     - Handles compatibility options

4. **Parsing Implementation**
   - ✅ `_parse_scenes()` - Parses USD scene structure, identifies default/root prims
   - ✅ `_parse_nodes()` - Parses USD prims into GLTF nodes
     - Handles transforms from `UsdGeomXformable`
     - Builds node hierarchy
     - Stores prim paths in node `additional_data` for mesh matching
   - ✅ `_parse_meshes()` - Parses USD meshes
     - Extracts vertices from `points` attribute
     - Handles face vertex indices and counts
     - Triangulates polygons
     - Extracts normals and UVs (primvar "st")
     - Creates `ImporterMesh` and `GLTFMesh` objects
     - Matches meshes to nodes via stored prim paths

5. **Integration with GLTF System**
   - ✅ Uses `SkinTool::_determine_skeletons()` for skeleton processing
   - ✅ Uses `SkinTool::_create_skeletons()` for skeleton creation
   - ✅ Uses `SkinTool::_create_skins()` for skin creation
   - ✅ Calls `GLTFDocument::generate_scene()` for final scene generation

### ✅ Recently Completed

1. **Image/Texture Parsing** (`_parse_images()`)
   - ✅ Extract texture file paths from USD materials
   - ✅ Traverse `UsdShadeMaterial` and `UsdUVTexture` shaders
   - ✅ Handle relative and absolute paths
   - ✅ Load and convert images (PNG, JPG, TGA, WEBP)
   - ✅ Create `GLTFTexture` and `GLTFImage` objects
   - ✅ Support texture extraction and embedding modes

2. **Material Parsing** (`_parse_materials()`)
   - ✅ Parse `UsdPreviewSurface` materials
   - ✅ Convert USD material properties to Godot `StandardMaterial3D`
   - ✅ Handle texture references (albedo, normal, emissive)
   - ✅ Support baseColor/diffuseColor, metallic, roughness
   - ✅ Parse opacity/transparency
   - ✅ Support material binding (materials stored in state)

3. **Camera Parsing** (`_parse_cameras()`)
   - ✅ Parse `UsdGeomCamera` prims
   - ✅ Extract camera properties (FOV from focal length/aperture, near/far planes)
   - ✅ Support perspective and orthographic projections
   - ✅ Convert to `GLTFCamera` objects
   - ✅ Match cameras to nodes via prim paths

4. **Light Parsing** (`_parse_lights()`)
   - ✅ Parse `UsdLux` light prims
   - ✅ Support different light types (directional, point, spot, area)
   - ✅ Extract light properties (intensity, color, cone angles)
   - ✅ Convert to `GLTFLight` objects
   - ✅ Match lights to nodes via prim paths

### ✅ Recently Completed (Continued)

5. **Skeleton/Skin Parsing** (`_parse_skins()`)
   - ✅ Parse `UsdSkel` skeletons
   - ✅ Extract joint hierarchies from `joints` attribute
   - ✅ Extract bind transforms and rest transforms
   - ✅ Parse skin bindings via `UsdSkelBindingAPI`
   - ✅ Extract joint influences from primvars (`jointIndices`, `jointWeights`)
   - ✅ Convert to GLTF skin format
   - ✅ Integrate with `SkinTool` processing
   - ✅ Match joints to nodes via prim paths

### ✅ Recently Completed (Continued)

6. **Animation Parsing** (`_parse_animations()`)
   - ✅ Extract time samples from USD attributes
   - ✅ Parse `UsdSkelAnimation` for skeletal animations
   - ✅ Extract joint translations, rotations, and scales
   - ✅ Parse transform animations from `UsdGeomXformable` prims
   - ✅ Convert to GLTF animation tracks
   - ✅ Handle animation time ranges from stage
   - ✅ Match animations to nodes via prim paths

### ✅ Recently Completed (Continued)

7. **Mesh Enhancements**
   - ✅ Support multiple UV sets (tries "st", "uv", "map1" primvars)
   - ✅ Handle different UV interpolation modes (vertex, varying, faceVarying, uniform, constant)
   - ✅ Handle vertex colors (displayColor primvar)
   - ✅ Support different color interpolation modes
   - ✅ Material binding via UsdShadeMaterialBindingAPI
   - ✅ Legacy material binding support
   - ✅ Better error handling and validation
   - ✅ Point cloud support (meshes without faces)

8. **Error Handling & Validation**
   - ✅ Better error messages for failed USD file opens
   - ✅ Validation of USD stage validity
   - ✅ Handle missing or invalid prims gracefully
   - ✅ Warning messages for problematic meshes
   - ✅ File existence checks

### ✅ Recently Completed (Continued)

9. **Blend Shapes/Morph Targets**
   - ✅ Parse `UsdSkelBlendShape` prims
   - ✅ Extract blend shape targets via `UsdSkelBindingAPI`
   - ✅ Extract vertex and normal offsets
   - ✅ Handle point indices (sparse blend shapes)
   - ✅ Convert USD offsets to Godot absolute positions
   - ✅ Support normalized blend shape mode

10. **Subdivision Surfaces**
   - ✅ Detect subdivision schemes (catmullClark, loop, bilinear)
   - ✅ Warn when subdivision surfaces are encountered
   - ✅ Import control cage as fallback
   - ⚠️ Full subdivision tessellation not yet implemented

11. **Performance Optimizations**
   - ✅ Cache prim path to node index mapping
   - ✅ Build material name to index map for faster lookups
   - ✅ Reorder parsing to minimize redundant traversals
   - ✅ Use cached lookups instead of linear searches where possible

### 🚧 In Progress / TODO

1. **Additional Mesh Features**
   - [ ] Support instancing
   - [ ] Multiple UV sets per surface (currently only first UV set)
   - [ ] Full subdivision surface tessellation (currently imports control cage)

2. **Further Performance Optimizations**
   - [ ] Single-pass prim traversal for multiple data types
   - [ ] Attribute value caching
   - [ ] Memory management optimizations for large USD files

## File Extensions Supported

- ✅ `.usd` - Binary USD format
- ✅ `.usda` - ASCII USD format
- ✅ `.usdc` - Cached USD format
- ❌ `.usdz` - **Not supported** (archive format explicitly excluded)

## Dependencies

- **gltf** module - For `GLTFDocument`, `GLTFState`, and `SkinTool`
- **openusd** module - For USD library linking and headers

## Build Configuration

The module links against USD libraries from the `openusd` module:
- Core USD libraries (`libusd_usd.a`, `libusd_usdGeom.a`, etc.)
- USD shading libraries (`libusd_usdShade.a`)
- USD skeleton libraries (`libusd_usdSkel.a`)
- USD lighting libraries (`libusd_usdLux.a`)
- Base USD libraries (`libusd_gf.a`, `libusd_sdf.a`, `libusd_tf.a`, etc.)
- TBB libraries for threading

## Testing Checklist

- [ ] Simple mesh import (vertices, faces)
- [ ] Mesh with normals
- [ ] Mesh with UVs
- [ ] Mesh with materials
- [ ] Scene hierarchy (nested transforms)
- [ ] Multiple meshes
- [ ] Cameras
- [ ] Lights
- [ ] Skeletons and skins
- [ ] Animations
- [ ] Complex USD files from production

## Notes

- The module does **not** convert USD to GLTF format files. Instead, it directly populates `GLTFState` in-memory, which is then used by `GLTFDocument` to generate Godot scenes.
- USDZ archive support was explicitly removed per requirements.
- The implementation follows the FBX module pattern closely for consistency and maintainability.

