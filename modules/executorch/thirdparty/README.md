# ExecuTorch Thirdparty Dependencies

This directory contains the ExecuTorch source code as managed by git subrepo for header access.
Prebuilt libraries should be available in system paths (e.g., /opt/executorch/lib for macOS/Linux).
SCsub will search system-wide ExecuTorch installation, avoiding git storage of binaries.

## Installation

Install ExecuTorch from prebuilt packages or build from source following:
https://pytorch.org/executorch/stable/using-executorch-building-from-source.html

For macOS arm64, build with:
```bash
cmake -B build -S . -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_FLAT_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DCMAKE_INSTALL_PREFIX=/opt/executorch
cmake --build build --parallel
cmake --install build
```

This installs headers to /opt/executorch/include and libs to /opt/executorch/lib.
