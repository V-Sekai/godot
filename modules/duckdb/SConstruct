#!/usr/bin/env python
import os
import sys

target_path = ARGUMENTS.pop("target_path", "demo/addons/godot-duckdb/bin/")
target_name = ARGUMENTS.pop("target_name", "libgdduckdb")
duckdb_lib_path = ARGUMENTS.pop("duckdb_lib_path", "src/duckdb")
duckdb_lib_name = ARGUMENTS.pop("duckdb_lib_name", "libduckdb.dylib")

target = "{}{}".format(
    target_path, target_name
)

env = SConscript("godot-cpp/SConstruct")
#env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME']=1

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags

# tweak this if you want to use different folders, or more folders, to store your source code in.
env.Append(CPPPATH=["src/"])
sources = [Glob("src/*.cpp")]

# Add duckdb dylib as dependency
env.Append(LIBPATH=[duckdb_lib_path])
env.Append(LIBS=[duckdb_lib_name])

# Allow exceptions in code, this is required becausse DuckDB throws errors
env.Append(CCFLAGS=['-fexceptions'])

if env["platform"] == "macos":
    target = "{}.{}.{}.framework/{}.{}.{}".format(
        target,
        env["platform"], 
        env["target"],
        target_name,
        env["platform"],
        env["target"]
    )
else:
    target = "{}{}{}".format(
        target,
        env["suffix"],
        env["SHLIBSUFFIX"]
    )

library = env.SharedLibrary(target=target, source=sources)
Default(library)

