def can_build(env, platform):
    # OS keychain backend required: macOS/iOS/visionOS Keychain Services,
    # Windows Credential Vault, Linux libsecret, Android KeyStore.
    # Only the web platform lacks a secure storage primitive.
    if platform in ["macos", "windows", "android", "ios", "visionos"]:
        return True
    if platform == "linuxbsd":
        import os

        return os.system("pkg-config --exists libsecret-1") == 0
    return False


def configure(env):
    pass
