cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18.0)

project (libTimekeeper
    VERSION 0.0.1
    LANGUAGES C CXX
    )

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/libTimekeeper/Config/Version.hh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/libTimekeeper/Config/Version.hh"
)

add_library(libTimekeeper INTERFACE)
add_library(Timekeeper::libTimekeeper ALIAS libTimekeeper)

target_compile_features(libTimekeeper INTERFACE cxx_std_17)

target_include_directories(libTimekeeper
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)


set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Timekeeper)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/.cmake/TimekeeperConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TimekeeperConfig.cmake"
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
    )

write_basic_package_version_file(
    TimekeeperConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
    )

install(TARGETS libTimekeeper
    EXPORT libTimekeeperTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT libTimekeeperTargets
    FILE libTimekeeperTargets.cmake
    NAMESPACE Timekeeper::
    DESTINATION ${INSTALL_CONFIGDIR}
)

export(EXPORT libTimekeeperTargets
    NAMESPACE Timekeeper::)

# Install Header Files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libTimekeeper
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hh"
    )

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TimekeeperConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TimekeeperConfig.cmake"
    DESTINATION ${INSTALL_CONFIGDIR})


install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/libTimekeeper/Config"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libTimekeeper)

