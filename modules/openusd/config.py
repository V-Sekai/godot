def can_build(env, platform):
    return platform == "macos"


def configure(env):
    import os

    usd_deps_dir = "#modules/openusd/thirdparty"
    os.makedirs(f"{usd_deps_dir}/install", exist_ok=True)
    os.system(f"""
        cd {usd_deps_dir} && \
        curl -L https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.9.0.zip --output oneTBB-2021.9.0.zip && \
        unzip oneTBB-2021.9.0.zip && mv oneTBB-2021.9.0 oneTBB && \
        cd oneTBB && mkdir -p build && cd build && \
        cmake .. -DTBB_TEST=OFF -DTBB_STRICT=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX={usd_deps_dir}/install && \
        cmake --build . --config Release && cmake --install .
    """)


def get_doc_classes():
    return []


def get_doc_path():
    return "doc_classes"
