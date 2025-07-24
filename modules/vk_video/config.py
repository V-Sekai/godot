def can_build(env, platform):
    # Vulkan Video requires Vulkan support
    return env.get("vulkan", True)


def configure(env):
    pass


def get_doc_classes():
    return [
        "VideoStreamMKV",
    ]


def get_doc_path():
    return "doc_classes"
