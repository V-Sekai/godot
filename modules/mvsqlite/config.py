# config.py

import subprocess


def can_build(env, platform):
    if platform == "ios" or platform == "web" or platform == "android":
        return False
    if platform == "windows" and not env["use_mingw"]:
        return False
    try:
        rust_version_output = subprocess.check_output(["rustup", "show"], stderr=subprocess.STDOUT)
        if "stable-x86_64-pc-windows-gnu" not in rust_version_output.decode('utf-8'):
            print("Default Rust toolchain is not GNU. mvsqlite build skipped.")
            if platform == "windows":
                print("Use `rustup default stable-x86_64-pc-windows-gnu` to set the default Rust toolchain to GNU.")
            return False
        subprocess.check_output(["cargo", "--version"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("Cargo or Rustup not found. mvsqlite build skipped.")
        if platform == "windows":
            print("Use `scoop install rustup-gnu` and `rustup target add x86_64-pc-windows-gnu` to install rust.")
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
