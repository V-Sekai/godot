def can_build(env, platform):
    return platform not in ["web", "android", "ios"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "WasGoRuntime",
        "WasGoState",
        "WasmResource",
        "ResourceFormatLoaderWasm",
    ]


def get_doc_path():
    return "doc_classes"
