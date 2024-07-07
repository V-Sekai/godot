TARGET_EXEC ?= thrust.wasm

BUILD_DIR ?= .

# COPY_DEST := "C\:\\Users\\Michael\\Documents\\_Active\\V-Sekai\\groups"
# COPY_DEST := C:/Users/Michael/Documents/_Active/V-Sekai/groups/

CXX := "../../thirdparty/zig-linux-x86_64-0.7.1+8078d8cd3/zig"
# CXX := "C:/Users/Michael/Documents/_Active/godot-wasm/godot/modules/WasGo/thirdparty/wasm-micro-runtime/wamr-sdk/opt/wasi-sdk/bin/clang.exe"

SRCS := $(wildcard *.cpp ../../api/Custom/*.cpp ../../api/Custom/wasgo/*.cpp ../../api/Generated/*.cpp)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := 
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= build-lib $(INC_FLAGS) --strip -fPIC -fno-compiler-rt -static -Bsymbolic

CPPFLAGS += -target wasm32-wasi -O ReleaseSafe
CPPFLAGS += -I../../thirdparty/wasi-sysroot/include
CPPFLAGS += -I../../thirdparty/wasi-sysroot/include/c++/v1
CPPFLAGS += -cflags -nostdlib --
# CPPFLAGS += -O3
# CPPFLAGS += -z
CPPFLAGS += --stack 409600
# CPPFLAGS += -Wl,--initial-memory=65536000
# CPPFLAGS += -std=c++11
# CPPFLAGS += --sysroot=C:\\wasi-sdk-11.0\\wasi-sysroot
# CPPFLAGS += --sysroot=C:\\Users\\Michael\\Documents\\_Active\\godot-wasm\\godot\\modules\\WasGo\\thirdparty\\wasm-micro-runtime\\wamr-sdk\\app\\libc-builtin-sysroot
# CPPFLAGS += -Wl,--allow-undefined-file=C:\\Users\\Michael\\Documents\\_Active\\godot-wasm\\godot\\modules\\WasGo\\thirdparty\\wasm-micro-runtime\\wamr-sdk\\app\\libc-builtin-sysroot\\share\\defined-symbols.txt
# CPPFLAGS += -Wl,--strip-all
# CPPFLAGS += -Wl,--no-entry
# CPPFLAGS += -nostdlib
# CPPFLAGS += -Wl,--no-threads
# CPPFLAGS += -Wl,--export-all

# CPPFLAGS += -cflags -Wl,--export=_notification --
# CPPFLAGS += -cflags -Wl,--export=_ready --
# CPPFLAGS += -cflags -Wl,--export=_process --
# CPPFLAGS += -cflags -Wl,--export=_physics_process --
# CPPFLAGS += -cflags -Wl,--export=_input --
# CPPFLAGS += -cflags -Wl,--export=_unhandled_input --
# CPPFLAGS += -cflags -Wl,--export=_unhandled_key_input --
# CPPFLAGS += -cflags -Wl,--export=_wasgo_input --
# CPPFLAGS += -cflags -Wl,--export=_wasgo_unhandled_input --
# CPPFLAGS += -cflags -Wl,--export=_wasgo_unhandled_key_input --

# CPPFLAGS += -Wl,--allow-undefined
# CPPFLAGS += --include-directory=C:\\Users\\Michael\\Documents\\_Active\\godot-wasm\\godot\\modules\\WasGo\\wasgo_headers
# CPPFLAGS += --include-directory=C:\\Users\\Michael\\Documents\\_Active\\godot-wasm\\godot\\godot\modules\WasGo\thirdparty\wasm-micro-runtime\wamr-sdk\opt\wasi-sdk\lib\clang\10.0.0\include
CPPFLAGS += -I../../api/Custom
CPPFLAGS += -I../../api/Generated
# CPPFLAGS += --include-directory=..\\wasgo_headers
# CPPFLAGS += --include-directory=C:\\wasi-sdk-11.0\\wasi-sysroot\\include
# CPPFLAGS += --include-directory=C:\\wasi-sdk-11.0\\wasi-sysroot\\include\\c++\\v1
CPPFLAGS += -DNO_THREADS
# CPPFLAGS += -oC:\\Users\\Michael\\Documents\\_Active\\V-Sekai\\groups\\test.wasm
# CPPFLAGS += -v

# $(COPY_DEST)/%.wasm: test.wasm
# 	$(info $@)
# 	copy /Y $< $@

$(BUILD_DIR)/$(TARGET_EXEC):
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRCS)

$(OBJS): Makefile

.PHONY: clean

clean:
	$(RM) $(TARGET_EXEC)

-include $(DEPS)

# MKDIR_P ?= mkdir -p

# $(info $(SRCS) )