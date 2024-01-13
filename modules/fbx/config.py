def can_build(env, platform):
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "EditorSceneFormatImporterUFBX",
        "FBXCamera",
        "FBXDocument",
        "FBXDocumentExtension",
        "FBXLight",
        "FBXMesh",
        "FBXState",
        "FBXTexture",
    ]


def get_doc_path():
    return "doc_classes"
