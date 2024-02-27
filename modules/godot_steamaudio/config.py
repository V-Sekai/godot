# config.py


def can_build(env, platform):
    if platform in ["web", "android"]:
        return False
    return True


def configure(env):
    pass
