/**************************************************************************/
/*  profiling.h                                                           */
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

#pragma once

#if defined(TRACY_ENABLE)

#include <tracy/Tracy.hpp>

#define ZN_PROFILER_ENABLED

#define ZN_PROFILE_SCOPE() ZoneScoped
#define ZN_PROFILE_SCOPE_NAMED(name) ZoneScopedN(name)
#define ZN_PROFILE_MARK_FRAME() FrameMark
#define ZN_PROFILE_SET_THREAD_NAME(name) tracy::SetThreadName(name)
#define ZN_PROFILE_PLOT(name, number) TracyPlot(name, number)
#define ZN_PROFILE_MESSAGE(message) TracyMessageL(message)
#define ZN_PROFILE_MESSAGE_DYN(message, size) TracyMessage(message, size)

#else

#define ZN_PROFILE_SCOPE()
// Name must be static const char* (usually string literal)
#define ZN_PROFILE_SCOPE_NAMED(name)
#define ZN_PROFILE_MARK_FRAME()
#define ZN_PROFILE_PLOT(name, number)
#define ZN_PROFILE_MESSAGE(message)
// Name must be const char*. An internal copy will be made so it can be temporary.
// Size does not include the terminating character.
#define ZN_PROFILE_MESSAGE_DYN(message, size)
// Name must be const char*. An internal copy will be made so it can be temporary.
#define ZN_PROFILE_SET_THREAD_NAME(name)

#endif

/*
To add Tracy support, clone it under thirdparty/tracy, and add the following lines in core/SCsub:

```
# tracy library
env.Append(CPPDEFINES="TRACY_ENABLE")
env_thirdparty.Append(CPPDEFINES="TRACY_ENABLE")
env_thirdparty.add_source_files(env.core_sources, ["#thirdparty/tracy/TracyClient.cpp"])
```
*/
