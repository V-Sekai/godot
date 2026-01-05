add_executable(deflate_benchmark)

file(GLOB SRC_DEFLATE_BENCHMARK *.c *.cpp *.h)

target_link_libraries(deflate_benchmark PRIVATE ufbx ufbxw_extra ufbxwt_util)
target_sources(deflate_benchmark PRIVATE ${SRC_DEFLATE_BENCHMARK})
