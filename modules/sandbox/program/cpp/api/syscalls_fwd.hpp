#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
struct Object;
struct Variant;
using real_t = float;

#define EXTERN_SYSCALL(rval, name, ...) \
	extern "C" rval name(__VA_ARGS__);

EXTERN_SYSCALL(void, sys_print, const Variant *, size_t);
EXTERN_SYSCALL(void, sys_throw, const char *, size_t, const char *, size_t, const Variant *);
EXTERN_SYSCALL(unsigned, sys_callable_create, void (*)(), const Variant *, const void *, size_t);

inline __attribute__((noreturn)) void api_throw(std::string_view type, std::string_view msg, const Variant *srcVar = nullptr) {
	sys_throw(type.data(), type.size(), msg.data(), msg.size(), srcVar);
	__builtin_unreachable();
}

extern "C" __attribute__((noreturn)) void fast_exit();

// Helper method to call a method on any type that can be wrapped in a Variant
#define VMETHOD(name) \
	template <typename... Args> \
	inline Variant name(Args&&... args) { \
		return operator() (#name, std::forward<Args>(args)...); \
	}

#define METHOD(Type, name) \
	template <typename... Args> \
	inline Type name(Args&&... args) { \
		if constexpr (std::is_same_v<Type, void>) { \
			voidcall(#name, std::forward<Args>(args)...); \
		} else { \
			return operator() (#name, std::forward<Args>(args)...); \
		} \
	}

// Helpers for static method calls
#define SMETHOD(Type, name) METHOD(Type, name)
#define SVMETHOD(name) VMETHOD(name)
