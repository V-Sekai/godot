/*************************************************************************/
/*  resonance_audio_wrapper.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RESONANCE_AUDIO_WRAPPER_H
#define RESONANCE_AUDIO_WRAPPER_H

#include "core/math/audio_frame.h"
#include "core/object.h"
#include "thirdparty/resonanceaudio/resonance_audio/api/resonance_audio_api.h"

struct AudioSourceId {
	vraudio::ResonanceAudioApi::SourceId id;
};

class ResonanceAudioWrapper : public Object {
	GDCLASS(ResonanceAudioWrapper, Object);

	static ResonanceAudioWrapper *singleton;

	vraudio::ResonanceAudioApi *resonance_api;

public:
	ResonanceAudioWrapper();

	static ResonanceAudioWrapper *get_singleton();

	AudioSourceId register_audio_source();
	void unregister_audio_source(AudioSourceId audio_source);

	void set_source_transform(AudioSourceId source, Transform source_transform);
	void set_head_transform(Transform head_transform);

	void push_source_buffer(AudioSourceId source, int num_frames, AudioFrame *frames);
	bool pull_listener_buffer(int num_frames, AudioFrame *frames);

	void set_source_attenuation(AudioSourceId source, float attenuation_linear);
};

#endif
