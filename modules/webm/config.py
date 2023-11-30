def can_build(env, platform):
    if platform in ["iphone", "web"]:
        return False

    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamWebm",
    ]


def get_doc_path():
    return "doc_classes"
