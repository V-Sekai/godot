def can_build(env, platform):
    return platform == "windows" or platform == "macos" or platform == "linux"


def configure(env):
    pass


def get_doc_classes():
    return [
        "RiscvEmulator",
    ]


def get_doc_path():
    return "doc_classes"
