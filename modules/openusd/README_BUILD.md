# Building USD for Godot

The `openusd` module requires USD (Universal Scene Description) libraries to be built before compiling Godot.

## Quick Start

1. Build USD using the provided script:
   ```bash
   cd modules/openusd
   ./build_usd.sh
   ```

2. Build Godot:
   ```bash
   cd ../..
   scons
   ```

## Prerequisites

- **CMake** (3.5 or later)
  - macOS: `brew install cmake`
  - Linux: Usually available via package manager
  - Windows: Download from cmake.org

- **Ninja** (optional, but recommended for faster builds)
  - macOS: `brew install ninja`
  - Linux: Usually available via package manager
  - Windows: Download from ninja-build.org

- **C++ Compiler** with C++17 support
  - macOS: Xcode Command Line Tools
  - Linux: GCC 7+ or Clang 5+
  - Windows: Visual Studio 2019 or later

## Build Process

The `build_usd.sh` script will:

1. Build **oneTBB** (Threading Building Blocks)
2. Build **OpenSubdiv** (subdivision surface library)
3. Build **USD** (Universal Scene Description)

All libraries will be installed to `modules/openusd/thirdparty/install/`.

## Build Time

- **First build**: 10-30 minutes (depending on your system)
- **Subsequent builds**: Only rebuilds if source changes

## Troubleshooting

### USD headers not found

If you see errors like `'pxr/usd/usd/stage.h' file not found`, make sure you've run `./build_usd.sh` first.

### CMake not found

Install CMake:
- macOS: `brew install cmake`
- Linux: `sudo apt-get install cmake` (Ubuntu/Debian) or equivalent
- Windows: Download from cmake.org

### Build fails

1. Check that all prerequisites are installed
2. Ensure you have enough disk space (USD build requires several GB)
3. Check the error messages for specific issues
4. Try building with fewer parallel jobs: `./build_usd.sh` (modify NUM_JOBS in script)

## Manual Build

If you prefer to build manually or need custom options:

```bash
cd modules/openusd/thirdparty

# Build oneTBB
cd onetbb
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install

# Build OpenSubdiv
cd ../../OpenSubdiv
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../install -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install

# Build USD
cd ../../usd
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=../../install \
    -DPXR_ENABLE_PYTHON_SUPPORT=OFF \
    -DPXR_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DPXR_BUILD_MONOLITHIC=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
```

## Verification

After building, verify USD is installed:

```bash
ls modules/openusd/thirdparty/install/include/pxr/pxr.h
```

If this file exists, USD is ready and you can build Godot.

