#!/usr/bin/env bash
go build -o desync_c_interface.a -buildmode=c-archive .
gcc untar/cgo_untar.c desync_c_interface.a -o cgo_untar -framework CoreFoundation -framework Security -lresolv
chmod +x cgo_untar