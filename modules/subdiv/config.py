def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "TriangleSubdivider",
        "TopologyDataMesh",
        "TopologyDataImporter",
        "SubdivisionMesh",
        "SubdivisionBaker",
        "Subdivider",
        "QuadSubdivider",
        "BakedSubdivMesh",
        "QuadSubdivider",
    ]


def get_doc_path():
    return "doc_classes"
