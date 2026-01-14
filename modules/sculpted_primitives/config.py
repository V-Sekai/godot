def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "CSGSculptedPrimitive3D",
        "CSGSculptedBox3D",
        "CSGSculptedCylinder3D",
        "CSGSculptedSphere3D",
        "CSGSculptedTorus3D",
        "CSGSculptedPrism3D",
        "CSGSculptedTube3D",
        "CSGSculptedRing3D",
        "CSGSculptedTexture3D",
    ]


def get_doc_path():
    return "doc_classes"
