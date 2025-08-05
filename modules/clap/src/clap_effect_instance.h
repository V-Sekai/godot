#pragma once

#include "godot_cpp/classes/audio_effect_instance.hpp"
#include "godot_cpp/classes/ref_counted.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/variant.hpp"

class ClapPluginHost;
using namespace godot;

class ClapAudioEffectInstance : public AudioEffectInstance {
	GDCLASS(ClapAudioEffectInstance, AudioEffectInstance)

protected:
	static void _bind_methods();

public:
	ClapPluginHost *host = nullptr;

	void _process(const void *p_src_buffer, godot::AudioFrame *p_dst_buffer, int32_t p_frame_count) override;
	bool _process_silence() const override;

	ClapAudioEffectInstance();
};
