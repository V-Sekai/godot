def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "FBXAnimation",
        "FBXDocument",
        "FBXDocumentExtension",
        "FBXMesh",
        "FBXNode",
        "FBXSkeleton",
        "FBXSkin",
        "FBXState",
        "FBXTexture",
        "EditorSceneFormatImporterUFBX",
    ]


def get_doc_path():
    return "doc_classes"
