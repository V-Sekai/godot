#pragma once

typedef struct MyState MyState;
#include "_generated/godot_elixir.h"
#include "main/main.h"
#include "os_linuxbsd.h"

#include <limits.h>

struct MyState {
    char *cwd = (char *)malloc(PATH_MAX);
	char *ret = nullptr;
};
