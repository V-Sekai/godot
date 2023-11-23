# config.py

import subprocess

def can_build(env, platform):
    try:
        subprocess.check_output(["cargo", "-V"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("Cargo not found. mvsqlite build skipped.")
        if platform == "windows":
            print("Use `scoop install rustup-gnu` to install.")
        return False
    if platform == "ios" or platform == "web" or platform == "android":
        return False
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "MVSQLite",
        "MVSQLiteQuery",
    ]


def get_doc_path():
    return "doc_classes"
