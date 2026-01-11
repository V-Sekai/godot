# Scene Merge Test Data

This directory contains GLB model files used for testing advanced mesh features in the SceneMerge integration tests.

## Test Assets

### MorphPrimitivesTest.glb
- **Source**: KhronosGroup/glTF-Sample-Assets/Models/MorphPrimitivesTest
- **Purpose**: Tests blend shape (morph target) functionality
- **Features**:
  - Complex primitives with morph targets for comprehensive testing
  - Tests that blend shapes are preserved through merge operations
  - Verifies SceneMerge doesn't crash on morph target data

### InterpolationTest.glb
- **Source**: KhronosGroup/glTF-Sample-Assets/Models/InterpolationTest
- **Purpose**: Tests animation interpolation and rigging features
- **Features**:
  - Various animation interpolation types
  - Skeletal animation structures
  - Validates handling of keyframe data

## Usage

These assets are automatically downloaded during test setup and loaded into the integration tests to verify:

1. **Blend Shape Support**: SceneMerge preserves morph target data in merged meshes
2. **Rigging Compatibility**: Merge operations handle animated meshes without corruption
3. **Advanced Mesh Compatibility**: Complex glTF features work with SceneMerge

## Download

The assets are downloaded automatically from the official Khronos GLTF Sample Assets repository.

## License

These sample assets are provided under the appropriate open-source licenses by the Khronos Group.
