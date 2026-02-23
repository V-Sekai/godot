#!/bin/sh
# Build Godot using all CPU cores and a local SCons cache. Pass any scons arguments, e.g.:
#   ./build_all_cores.sh target=editor
#   ./build_all_cores.sh target=template_release
# Override cache: ./build_all_cores.sh target=editor cache_path=/tmp/my_cache cache_limit=10
CACHE_PATH="${CACHE_PATH:-$HOME/.scons_cache}"
CACHE_LIMIT="${CACHE_LIMIT:-5}"
exec scons debug_symbols=yes tests=yes cache_path="$CACHE_PATH" cache_limit="$CACHE_LIMIT" "$@"
