# config.py


def can_build(env, platform):
    if platform in ["web", "android", "linux"]:
        return False
    return True


def configure(env):
    pass
