#pragma once

#ifdef GODOT_MODULE
#define GODOT_CPP_FUNC(name) name
#define GODOT_NAMESPACE
#define GODOT_OVERRIDE
#else
#define GODOT_CPP_FUNC(name) _##name
#define GODOT_NAMESPACE using namespace godot;
#define GODOT_OVERRIDE override
#endif
