#!/usr/bin/env python3
"""
Godot CNode Builder
Builds the Godot CNode executable for Erlang/Elixir integration.

This script builds the CNode standalone without requiring mix/elixir.
It uses SCons directly to build the CNode executable.

Usage:
    python3 modules/cnode/build_cnode.py

Or build directly with SCons:
    scons library_type=executable extra_suffix=cnode dev_build=yes debug_symbols=yes
"""

import subprocess
import os
import sys
import platform
import shutil

# Platform-specific settings
is_windows = platform.system() == "Windows"
is_macos = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"

if is_windows:
    godot_exe = "bin/godot.windows.editor.dev.x86_64.executable.exe"
    godot_lib_pattern = "bin/godot.windows.*.static_library"
    cnode_pattern = "bin/godot.windows.*.cnode.exe"
    cnode_exe = "priv/godot_cnode.exe"
else:
    if is_macos:
        godot_exe = "./bin/godot.macos.editor.dev.x86_64.executable"
        godot_lib_pattern = "bin/godot.macos.*.static_library"
        cnode_pattern = "bin/godot.macos.*.cnode"
    else:
        godot_exe = "./bin/godot.linuxbsd.editor.dev.x86_64.executable"
        godot_lib_pattern = "bin/godot.linuxbsd.*.static_library"
        cnode_pattern = "bin/godot.linuxbsd.*.cnode"
    cnode_exe = "./priv/godot_cnode"

print("=" * 60)
print("Building Godot CNode")
print("=" * 60)

# Step 1: Check if Godot needs to be built
godot_needs_build = True
if os.path.exists(godot_exe):
    print(f"Godot executable found: {godot_exe}")
    # Check if we need to rebuild based on library existence
    import glob
    libs = glob.glob(godot_lib_pattern)
    if libs:
        print(f"Godot library found: {libs[0]}")
        godot_needs_build = False
    else:
        print("Godot library not found, will build Godot")

if godot_needs_build:
    print("\n[1/4] Building Godot executable...")
    # Build Godot executable first (needed for API generation)
    subprocess.run([
        "scons", 
        "extra_suffix=executable", 
        "dev_build=yes", 
        "debug_symbols=yes",
        "scu_build=yes"
    ], check=True)
    print("✓ Godot executable built")

    print("\n[2/4] Generating extension API and interface files...")
    # Generate extension API and interface files
    subprocess.run([
        godot_exe,
        "--dump-extension-api",
        "--dump-gdextension-interface",
        "--headless"
    ], check=True)
    print("✓ Extension API generated")

    print("\n[3/4] Building Godot static library...")
    # Build Godot static library for linking
    subprocess.run([
        "scons",
        "library_type=static_library",
        "extra_suffix=static_library",
        "dev_build=yes",
        "debug_symbols=yes",
        "scu_build=yes"
    ], check=True)
    print("✓ Godot static library built")
else:
    print("\n[1-3/4] Skipping Godot build (already built)")

# Step 2: Check if CNode needs to be built
cnode_needs_build = True
import glob
cnode_libs = glob.glob(cnode_pattern)
if cnode_libs:
    print(f"CNode found: {cnode_libs[0]}")
    cnode_needs_build = False
else:
    print("CNode not found, will build CNode")

if cnode_needs_build:
    print("\n[4/4] Building CNode...")
    # Build CNode using SCons with library_type=executable and extra_suffix=cnode
    result = subprocess.run([
        "scons",
        "library_type=executable",
        "extra_suffix=cnode",
        "dev_build=yes",
        "debug_symbols=yes",
        "scu_build=yes"
    ], check=True)
    if result.returncode != 0:
        print("✗ CNode build failed")
        sys.exit(1)

    # Verify CNode was built using pattern
    cnode_libs = glob.glob(cnode_pattern)
    if not cnode_libs:
        print(f"✗ CNode executable not found with pattern {cnode_pattern}")
        # Check if it was built elsewhere
        found = glob.glob("**/godot_cnode", recursive=True) or glob.glob("**/godot_cnode.exe", recursive=True)
        if found:
            print(f"Found CNode at: {found[0]}")
        else:
            print("CNode not found anywhere")
            sys.exit(1)
    else:
        print(f"✓ CNode built successfully: {cnode_libs[0]}")
else:
    print("\n[4/4] Skipping CNode build (already built)")

print("\n" + "=" * 60)
print("Build complete!")
print("=" * 60)

# Get the actual cnode executable path
cnode_libs = glob.glob(cnode_pattern)
if cnode_libs:
    cnode_actual = cnode_libs[0]
    print(f"\nCNode executable: {os.path.abspath(cnode_actual)}")
    print("\nTo run the CNode:")
    if is_windows:
        print(f"  {cnode_actual} -name godot@127.0.0.1 -setcookie godotcookie")
    else:
        print(f"  ./{cnode_actual} -name godot@127.0.0.1 -setcookie godotcookie")
else:
    print(f"\nCNode executable: {os.path.abspath(cnode_exe)}")
    print("\nTo run the CNode:")
    print(f"  {cnode_exe} -name godot@127.0.0.1 -setcookie godotcookie")

