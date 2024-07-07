TARGET_EXEC ?= thrust.wasm

BUILD_DIR ?= .

CXX := "../../thirdparty/zig-linux-x86_64-0.7.1+8078d8cd3/zig"

SRCS := $(wildcard *.zig ../../api/Custom/*.cpp ../../api/Custom/wasgo/*.cpp ../../api/Generated/*.cpp)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := 
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= build-lib $(INC_FLAGS)

CPPFLAGS += -target wasm32-wasi -O ReleaseSafe
CPPFLAGS += -fPIC
CPPFLAGS += -fno-compiler-rt
CPPFLAGS += --strip
CPPFLAGS += -static
CPPFLAGS += --single-threaded
CPPFLAGS += -I../../thirdparty/wasi-sysroot/include
CPPFLAGS += -I../../thirdparty/wasi-sysroot/include/c++/v1
CPPFLAGS += -cflags -nostdlib --
CPPFLAGS += --stack 409600
CPPFLAGS += -I../../api/Custom
CPPFLAGS += -I../../api/Generated
CPPFLAGS += -DNO_THREADS
# CPPFLAGS += -Wl,--initial-memory=65536000
# CPPFLAGS += -Wl,--allow-undefined-file=C:\\Users\\Michael\\Documents\\_Active\\godot-wasm\\godot\\modules\\WasGo\\thirdparty\\wasm-micro-runtime\\wamr-sdk\\app\\libc-builtin-sysroot\\share\\defined-symbols.txt


$(BUILD_DIR)/$(TARGET_EXEC):
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRCS)

$(OBJS): Makefile

.PHONY: clean

clean:
	$(RM) $(TARGET_EXEC)

-include $(DEPS)