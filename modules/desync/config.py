import os
import subprocess


def can_build(env, platform):
    if platform == "windows" and os.name != "nt" and env["use_mingw"]:
        return False
    if platform == "windows":
        if not env["use_mingw"]:
            return False
        try:
            import subprocess

            mingw_version = subprocess.check_output(["gcc", "--version"])
            print("MinGW is installed: ", mingw_version)
        except Exception as e:
            print("MinGW is not installed or not found in PATH")
            return False
        return True
    return platform in ["macos", "linux", "windows"]


def get_doc_classes():
    return [
        "Desync",
    ]


def configure(env):
    try:
        go_version = subprocess.check_output(["go", "version"])
        print("Golang is installed: ", go_version)
    except Exception as e:
        print("Golang is not installed or not found in PATH")
        return False
    return True


def get_doc_path():
    return "doc_classes"
