# config.py


def can_build(env, platform):
    if platform in ["web", "android", "macos"]:
        return False
    return True


def configure(env):
    pass
