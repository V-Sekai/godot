# CMake toolchain file for building TensorFlow Lite with libc++ compatibility
# This file is used by build_tflite.sh to configure the compiler

# Don't set CMAKE_SYSTEM_NAME to avoid cross-compilation detection  
set(CMAKE_C_COMPILER /home/linuxbrew/.linuxbrew/bin/clang)
set(CMAKE_CXX_COMPILER /home/linuxbrew/.linuxbrew/bin/clang++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(LIBCXX_LIB_DIR "/home/linuxbrew/.linuxbrew/Cellar/llvm/21.1.5/lib")
set(CMAKE_EXE_LINKER_FLAGS_CXX "${CMAKE_EXE_LINKER_FLAGS_CXX} -stdlib=libc++ -L${LIBCXX_LIB_DIR} -Wl,-rpath,${LIBCXX_LIB_DIR} ${LIBCXX_LIB_DIR}/libc++.a ${LIBCXX_LIB_DIR}/libc++abi.a")
set(CMAKE_SHARED_LINKER_FLAGS_CXX "${CMAKE_SHARED_LINKER_FLAGS_CXX} -stdlib=libc++ -L${LIBCXX_LIB_DIR} -Wl,-rpath,${LIBCXX_LIB_DIR} ${LIBCXX_LIB_DIR}/libc++.a ${LIBCXX_LIB_DIR}/libc++abi.a")
set(CMAKE_MODULE_LINKER_FLAGS_CXX "${CMAKE_MODULE_LINKER_FLAGS_CXX} -stdlib=libc++ -L${LIBCXX_LIB_DIR} -Wl,-rpath,${LIBCXX_LIB_DIR} ${LIBCXX_LIB_DIR}/libc++.a ${LIBCXX_LIB_DIR}/libc++abi.a")
set(CMAKE_CXX_STANDARD_LIBRARIES "${LIBCXX_LIB_DIR}/libc++.a ${LIBCXX_LIB_DIR}/libc++abi.a")

