# config.py


def can_build(env, platform):
    if platform not in ["web", "android", "linux"]:
        return True
    return False


def configure(env):
    pass
