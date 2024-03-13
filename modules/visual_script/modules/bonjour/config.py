def can_build(env, platform):
    return platform == "windows" or platform == "osx" or platform == "x11"

def configure(env):
    pass

def get_doc_classes():
    return [
        "Bonjour",
    ]

def get_doc_path():
    return "doc_classes"
