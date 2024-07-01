/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "core/object/class_db.h"
#include "speech.h"

#include "speech_decoder.h"
#include "speech_processor.h"

#include "thirdparty/jitter.h"

extern "C"
#ifdef __GNUC__
		__attribute__((noreturn))
#endif
		void
		celt_fatal(const char *str, const char *file, int line) {
	ERR_PRINT(String(str));
#if defined(_MSC_VER)
	_set_abort_behavior(0, _WRITE_ABORT_MSG);
#endif
	abort();
}

void initialize_speech_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_CLASS(PlaybackStats);
	GDREGISTER_CLASS(SpeechProcessor);
	GDREGISTER_CLASS(SpeechDecoder);
	GDREGISTER_CLASS(Speech);
	GDREGISTER_CLASS(JitterBuffer);
	GDREGISTER_CLASS(JitterBufferPacket);
}

void uninitialize_speech_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
