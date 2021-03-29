/*************************************************************************/
/*  config.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef CMARK_CONFIG_H
#define CMARK_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#define HAVE_STDBOOL_H

#ifdef HAVE_STDBOOL_H
#include <stdbool.h>
#elif !defined(__cplusplus)
typedef char bool;
#endif

#define HAVE___ATTRIBUTE__

#ifdef HAVE___ATTRIBUTE__
#define CMARK_ATTRIBUTE(list) __attribute__(list)
#else
#define CMARK_ATTRIBUTE(list)
#endif

#ifndef CMARK_INLINE
#if defined(_MSC_VER) && !defined(__cplusplus)
#define CMARK_INLINE __inline
#else
#define CMARK_INLINE inline
#endif
#endif

/* snprintf and vsnprintf fallbacks for MSVC before 2015,
   due to Valentin Milea http://stackoverflow.com/questions/2915672/
*/

#if defined(_MSC_VER) && _MSC_VER < 1900

#include <stdarg.h>
#include <stdio.h>

#define snprintf c99_snprintf
#define vsnprintf c99_vsnprintf

CMARK_INLINE int c99_vsnprintf(char *outBuf, size_t size, const char *format, va_list ap) {
	int count = -1;

	if (size != 0)
		count = _vsnprintf_s(outBuf, size, _TRUNCATE, format, ap);
	if (count == -1)
		count = _vscprintf(format, ap);

	return count;
}

CMARK_INLINE int c99_snprintf(char *outBuf, size_t size, const char *format, ...) {
	int count;
	va_list ap;

	va_start(ap, format);
	count = c99_vsnprintf(outBuf, size, format, ap);
	va_end(ap);

	return count;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
