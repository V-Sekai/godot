/**************************************************************************/
/*  libc.hpp                                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef LIBC_HPP
#define LIBC_HPP
#include <cstddef>
#include <cstdint>
#define _NOTHROW __attribute__((__nothrow__))

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noreturn)) void panic(const char *reason);

#ifdef __GLIBC__
#include <stdlib.h>
#include <string.h>
#else
extern void *memset(void *dest, int ch, size_t size);
extern void *memcpy(void *dest, const void *src, size_t size);
extern void *memmove(void *dest, const void *src, size_t size);
extern int memcmp(const void *ptr1, const void *ptr2, size_t n);
extern char *strcpy(char *dst, const char *src);
extern size_t strlen(const char *str);
extern int strcmp(const char *str1, const char *str2);
extern char *strcat(char *dest, const char *src);

extern void *malloc(size_t) _NOTHROW;
extern void *calloc(size_t, size_t) _NOTHROW;
extern void free(void *) _NOTHROW;
#endif

#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

#ifdef __cplusplus
}
#endif

extern "C" long sys_write(const void *data, size_t len);

inline void put_string(const char *string) {
	(void)sys_write(string, __builtin_strlen(string));
}

#endif // LIBC_HPP
