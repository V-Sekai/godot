project(TimekeeperExample)

if (NOT TARGET Timekeeper::libTimekeeper)
    find_package(Timekeeper REQUIRED)
endif()

add_executable(example example.cc)
target_link_libraries(example PRIVATE Timekeeper::libTimekeeper)

