def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "BlendShapeBake",
        "DemBonesProcessor",
    ]


def get_doc_path():
    return "."
