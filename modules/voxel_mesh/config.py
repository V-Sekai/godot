def can_build(env, platform):
    return env.editor_build


def configure(env):
    pass


def get_icons_path():
    return "editor/icons"


def get_doc_classes():
    return [
        "VoxelServer",
        "Voxel",
        "VoxelLibrary",
        "VoxelColorPalette",
        "VoxelInstanceLibrary",
        "VoxelInstanceLibraryItem",
        "VoxelInstanceGenerator",
        "VoxelBuffer",
        "VoxelNode",
        "VoxelTerrain",
        "VoxelLodTerrain",
        "VoxelViewer",
        "VoxelInstancer",
        "VoxelStream",
        "VoxelStreamFile",
        "VoxelStreamBlockFiles",
        "VoxelStreamRegionFiles",
        "VoxelStreamSQLite",
        "VoxelStreamScript",
        "VoxelGenerator",
        "VoxelGeneratorFlat",
        "VoxelGeneratorWaves",
        "VoxelGeneratorHeightmap",
        "VoxelGeneratorImage",
        "VoxelGeneratorNoise2D",
        "VoxelGeneratorNoise",
        "VoxelGeneratorGraph",
        "VoxelGeneratorScript",
        "VoxelBoxMover",
        "VoxelRaycastResult",
        "VoxelTool",
        "VoxelToolBuffer",
        "VoxelBlockSerializer",
        "VoxelVoxLoader",
        "VoxelMesher",
        "VoxelMesherBlocky",
        "VoxelMesherDMC",
        "VoxelMesherCubes",
        "VoxelVoxImporter",
    ]


def get_doc_path():
    return "doc/classes"
