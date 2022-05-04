
.PHONY: bin/libgodot.linuxbsd.opt.tools.64.llvm.so

bin/libgodot.linuxbsd.opt.tools.64.llvm.so: 
	scons p=linuxbsd -j`nproc` target=release_debug use_llvm=yes use_lld=yes optimize=size debug_symbols=yes
	mkdir -p ${MIX_APP_PATH}/dev/lib/godot/priv/bundlex/nif/
	cp -f bin/libgodot.linuxbsd.opt.tools.64.llvm.so ${MIX_APP_PATH}/dev/lib/godot/priv/bundlex/nif/