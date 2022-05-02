/*************************************************************************/
/*  resonance_audio_wrapper.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/error/error_macros.h"
#include "core/math/audio_frame.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/templates/list.h"
#include "core/templates/rid.h"
#include "core/templates/rid_owner.h"
#include "core/templates/set.h"
#include "core/variant/variant.h"

#include "thirdparty/resonanceaudio/resonance_audio/api/resonance_audio_api.h"

class AudioSourceId {
	vraudio::ResonanceAudioApi::SourceId id = -1;

public:
	AudioSourceId(vraudio::ResonanceAudioApi::SourceId p_id = -1) :
			id(p_id) {}
	vraudio::ResonanceAudioApi::SourceId get_id() const {
		return id;
	}
};
class AudioServer;
struct ResonanceAudioBus {
	RID self;

private:
	vraudio::ResonanceAudioApi *resonance_api = nullptr;

public:
	AudioSourceId register_audio_source() {
		resonance_api->CreateSoundObjectSource(vraudio::RenderingMode::kBinauralHighQuality);
		vraudio::ResonanceAudioApi::SourceId new_id = -1;
		resonance_api->SetSourceDistanceModel(
				new_id,
				vraudio::DistanceRolloffModel::kNone,
				/* min_distance= */ 0,
				/* max_distance= */ 0);
		AudioSourceId new_source = AudioSourceId(new_id);
		return new_source;
	}

	void unregister_audio_source(AudioSourceId audio_source) {
		if (audio_source.get_id() == -1) {
			return;
		}
		resonance_api->DestroySource(audio_source.get_id());
	}

	void set_source_transform(AudioSourceId source, Transform3D source_transform) {
		Quaternion source_rotation = Quaternion(source_transform.basis);
		resonance_api->SetSourcePosition(source.get_id(), source_transform.origin.x, source_transform.origin.y, source_transform.origin.z);
		resonance_api->SetSourceRotation(source.get_id(), source_rotation.x, source_rotation.y, source_rotation.z, source_rotation.w);
	}

	void set_head_transform(Transform3D head_transform) {
		Quaternion head_rotation = Quaternion(head_transform.basis);
		resonance_api->SetHeadPosition(head_transform.origin.x, head_transform.origin.y, head_transform.origin.z);
		resonance_api->SetHeadRotation(head_rotation.x, head_rotation.y, head_rotation.z, head_rotation.w);
	}

	void push_source_buffer(AudioSourceId source, int num_frames, AudioFrame *frames) {
		// Frames are just interleaved floats.
		resonance_api->SetInterleavedBuffer(source.get_id(), (const float *)frames, /* num_channels= */ 2, num_frames);
	}

	bool pull_listener_buffer(int num_frames, AudioFrame *frames) {
		// Frames are just interleaved floats.
		bool success = resonance_api->FillInterleavedOutputBuffer(/* num_channels= */ 2, num_frames, (float *)frames);
		if (!success) {
			// Zero out the array because Resonance Audio fills buffers with garbage on error under some circumstances.
			for (int32_t frame_i = 0; frame_i < num_frames; frame_i++) {
				frames[frame_i] = {};
			}
		}
		return success;
	}

	void set_source_attenuation(AudioSourceId source, float attenuation_linear) {
		resonance_api->SetSourceDistanceAttenuation(source.get_id(), attenuation_linear);
	}

	AudioSourceId register_stero_audio_source() {
		AudioSourceId new_source = AudioSourceId(resonance_api->CreateStereoSource(2));
		return new_source;
	}

	void set_linear_source_volume(AudioSourceId audio_source, real_t volume) {
		resonance_api->SetSourceVolume(audio_source.get_id(), volume);
	}
	_FORCE_INLINE_ void set_self(const RID &p_self) {
		self = p_self;
	}

	_FORCE_INLINE_ RID get_self() const {
		return self;
	}

	ResonanceAudioBus();
	~ResonanceAudioBus(){};
};

class ResonanceAudioServer : public Object {
	GDCLASS(ResonanceAudioServer, Object);

	static ResonanceAudioServer *singleton;

public:
	static ResonanceAudioServer *get_singleton() {
		return singleton;
	}

protected:
	static void _bind_methods() {
	}

private:
	RID_Owner<ResonanceAudioBus, true> bus_owner;
	RID master_bus;

public:
	RID create_bus() {
		RID ret = bus_owner.make_rid();
		ResonanceAudioBus *ptr = bus_owner.get_or_null(ret);
		ERR_FAIL_NULL_V(ptr, RID());
		ptr->set_self(ret);
		master_bus = ret;

		return ret;
	}

	AudioSourceId register_audio_source() {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			AudioSourceId source = ptr->register_audio_source();
			return source;
		}
		return AudioSourceId(-1);
	}
	void unregister_audio_source(AudioSourceId audio_source) {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			ptr->unregister_audio_source(audio_source);
			return;
		}
		return;
	}
	AudioSourceId register_stero_audio_source() {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			AudioSourceId source = ptr->register_stero_audio_source();
			return source;
		}
		return AudioSourceId(-1);
	}
	void set_source_transform(AudioSourceId audio_source, Transform3D source_transform) {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			ptr->set_source_transform(audio_source, source_transform);
			return;
		}
		return;
	}
	void set_head_transform(Transform3D head_transform) {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			ptr->set_head_transform(head_transform);
			return;
		}
		return;
	}
	void push_source_buffer(AudioSourceId source, int num_frames, AudioFrame *frames) {
		ResonanceAudioBus *bus = bus_owner.get_or_null(master_bus);
		if (bus) {
			bus->push_source_buffer(source, num_frames, frames);
			return;
		}
		return;
	}
	bool pull_listener_buffer(int num_frames, AudioFrame *frames) {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			return ptr->pull_listener_buffer(num_frames, frames);
		}
		return false;
	}

	void set_source_attenuation(AudioSourceId source, float attenuation_linear) {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			ptr->set_source_attenuation(source, attenuation_linear);
			return;
		}
		return;
	}

	void set_linear_source_volume(AudioSourceId audio_source, real_t volume) {
		ResonanceAudioBus *ptr = bus_owner.get_or_null(master_bus);
		if (ptr) {
			ptr->set_linear_source_volume(audio_source, volume);
			return;
		}
		return;
	}
	ResonanceAudioServer() {
		singleton = this;
		master_bus = create_bus();
	}
};

#endif
